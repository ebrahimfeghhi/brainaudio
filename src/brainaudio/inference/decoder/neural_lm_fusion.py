# Copyright (c) 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Language Model fusion for CTC beam search decoding."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn.functional as F

class NeuralLanguageModelFusion(ABC):
    """
    Base class for neural language model fusion during beam search.
    
    This class defines the interface for integrating neural LMs (e.g., GPT-2, LLaMA)
    into CTC beam search decoding. The LM is applied at word boundaries to rescore
    hypotheses based on linguistic context.
    
    Design:
        - Sparse fusion: Only applied at word boundaries (not every frame)
        - Phoneme-space beams: Beams remain in phoneme space, homophones scored separately
        - Aggregation: Multiple homophone scores combined via max or logsumexp
        - Before pruning: LM scores added before topk selection to guide search
    """
    
    def __init__(
        self, 
        weight: float = 0.3,
        homophone_aggregation: str = 'max',
        device: torch.device = None,
    ):
        """
        Initialize the neural LM fusion module.
        
        Args:
            weight: Weight for LM scores when combining with acoustic scores.
                   Typical range: 0.1-0.5. Higher values give LM more influence.
            homophone_aggregation: How to combine scores across homophones:
                - 'max': Take best homophone (recommended for 1-best decoding)
                - 'logsumexp': Bayesian average (better for N-best/confidence)
            device: Device for LM inference. If None, uses cuda if available.
        """
        self.weight = weight
        self.homophone_aggregation = homophone_aggregation
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_homophone_scores = False
        
        if homophone_aggregation not in ['max', 'logsumexp']:
            raise ValueError(f"homophone_aggregation must be 'max' or 'logsumexp', got {homophone_aggregation}")
    
    @abstractmethod
    def score_continuations(
        self, 
        contexts: List[str], 
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        """
        Score candidate words given context.
        
        This is the main method that subclasses must implement. It queries the
        neural LM to get log-probabilities for each candidate word continuation.
        
        Args:
            contexts: List of text contexts (one per beam).
                     Example: ["I saw my", "picnic with"]
            candidate_words: List of candidate word lists (one list per beam).
                           Example: [["aunt", "ant"], ["aunt", "ant"]]
                           
        Returns:
            scores: List of score lists matching structure of candidate_words.
                   Each score is a log-probability (negative values, higher is better).
                   Example: [[0.7, 0.01], [0.3, 0.6]] for the inputs above.
                   
        Notes:
            - Return scores in the same order as candidate_words
            - Use log-probabilities (not raw probs)
            - Can return approximate scores (e.g., from sampling) if exact scoring is slow
        """
        raise NotImplementedError("Subclasses must implement score_continuations()")
    
    def aggregate_homophone_scores(self, scores: List[float]) -> float:
        """
        Aggregate multiple homophone scores into a single score.
        
        Args:
            scores: List of log-probability scores for different homophones.
                   Example: [-0.357, -4.605] for ["aunt", "ant"]
                   
        Returns:
            Aggregated log-probability score.
            
        Examples:
            >>> # max aggregation (winner-take-all)
            >>> self.aggregate_homophone_scores([-0.357, -4.605])
            -0.357  # Best homophone wins
            
            >>> # logsumexp aggregation (Bayesian)
            >>> self.aggregate_homophone_scores([-0.357, -4.605])
            -0.342  # log(exp(-0.357) + exp(-4.605)) ≈ log(0.7 + 0.01)
        """
        if len(scores) == 1:
            return scores[0]
        
        if self.homophone_aggregation == 'max':
            return max(scores)
        elif self.homophone_aggregation == 'logsumexp':
            return torch.logsumexp(torch.tensor(scores), dim=0).item()
        else:
            raise ValueError(f"Unknown aggregation method: {self.homophone_aggregation}")
    
    def to(self, device: torch.device):
        """Move LM to specified device."""
        self.device = device
        return self


class DummyLMFusion(NeuralLanguageModelFusion):
    """
    Dummy LM fusion that returns uniform scores.
    Useful for testing the integration without actual LM overhead.
    """
    
    def score_continuations(
        self, 
        contexts: List[str], 
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        """Return uniform (zero) log-probability for all candidates."""
        return [[0.0 for _ in words] for words in candidate_words]


class HuggingFaceLMFusion(NeuralLanguageModelFusion):
    """
    Neural LM fusion using HuggingFace transformers (GPT-2, LLaMA, etc).
    
    Example usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> lm_fusion = HuggingFaceLMFusion(model, tokenizer, weight=0.3)
        >>> decoder = BatchedBeamCTCComputer(..., lm_fusion=lm_fusion)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        weight: float = 0.3,
        homophone_aggregation: str = 'max',
        device: torch.device = None,
        max_context_length: int = 512,
    ):
        """
        Initialize HuggingFace LM fusion.
        
        Args:
            model: HuggingFace causal LM model (e.g., GPT2LMHeadModel)
            tokenizer: Corresponding tokenizer
            weight: LM weight for fusion
            homophone_aggregation: 'max' or 'logsumexp'
            device: Device for inference
            max_context_length: Maximum context length in tokens (truncate if longer)
        """
        super().__init__(weight, homophone_aggregation, device)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_context_length = max_context_length
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        

    @torch.no_grad()
    def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> List[List[float]]:
        
        flat_texts = []
        flat_context_lens = []
        
        # 1. Flatten structure
        for context, candidates in zip(contexts, candidate_words):
            # Calculate context length roughly (approximation for indexing)
            # Note: add_special_tokens=False is safer if we want raw length
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            ctx_len = len(ctx_ids)
            
            for word in candidates:
                # 2. Fix Spacing Logic
                if not context:
                    full_text = word
                elif context.endswith(" ") or word.startswith(" "):
                    full_text = f"{context}{word}"
                else:
                    full_text = f"{context} {word}"
                
                flat_texts.append(full_text)
                flat_context_lens.append(ctx_len) 

        if not flat_texts:
            return []
    
        # 3. Batch Tokenize
        # Note: 'padding=True' is essential if lengths differ
        inputs = self.tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        input_ids = inputs.input_ids 
        attention_mask = inputs.attention_mask # padded tokens have a value of 0 

        # Forward Pass
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 

        # Shift Logits & Labels
        shift_logits = logits[..., :-1, :].contiguous() # B x (Seq Len - 1) x Vocab
        shift_labels = input_ids[..., 1:].contiguous() # B x (Seq Len - 1)
        
        
        # Log Softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather scores
        gathered_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1) # Beam Size x (Seq Len - 1)
        
        # Reconstruct and Sum
        final_scores = []
        flat_idx = 0
        
        for candidates in candidate_words:
            
            beam_scores = []
            
            for _ in candidates:
                
                start_idx = flat_context_lens[flat_idx] 
                
                # 4. Fix End Index Logic
                # We want to sum from the end of context to the end of the VALID sequence
                # -1 because gathered_probs is shifted
                total_valid_len = attention_mask[flat_idx].sum().item() - 1
                
                
                score = gathered_probs[flat_idx, start_idx:total_valid_len].sum().item()
                beam_scores.append(score)
                flat_idx += 1
                
            final_scores.append(beam_scores)

        return final_scores
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self
