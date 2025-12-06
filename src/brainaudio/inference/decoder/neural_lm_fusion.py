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
        self.max_context_length = max_context_length
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bos_token_id = self.tokenizer.bos_token_id
        if self.bos_token_id is None:
            self.bos_token_id = self.tokenizer.eos_token_id
        if self.bos_token_id is None:
            self.bos_token_id = self.tokenizer.pad_token_id
        if self.bos_token_id is None:
            raise ValueError("Tokenizer must define at least one of bos/eos/pad tokens")
    
    @torch.no_grad()
    def score_continuations(
        self, 
        contexts: List[str], 
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        
        """
        Docstring for score_continuations
        
        :param contexts: List of strings, each representing the context for a beam
        :type contexts: List[str]
        :param candidate_words: List of lists of candidate words for each beam
        :type candidate_words: List[List[str]]
        :return: List of lists of log-probabilities for each candidate word in each beam
        :rtype: List[List[float]]
        """
        
        print("CONTEXTS:", contexts)
        if not contexts:
            return []

        # 1. Flatten inputs for batching
        flat_texts = []
        meta_info = [] # Stores (beam_idx, word_start_index)

        for beam_idx, (context, words) in enumerate(zip(contexts, candidate_words)):
            if not words:
                continue

            prefix_ids = self.tokenizer.encode(context, add_special_tokens=False)
            prefix_len = len(prefix_ids) + 1  # +1 for manual BOS

            for word in words:
                full_text = f"{context} {word}" if context else word
                flat_texts.append(full_text)
                meta_info.append((beam_idx, prefix_len))

        if not flat_texts:
            return [[] for _ in candidate_words]

        # 2. Tokenize all beams + candidate words (manual BOS added later)
        inputs = self.tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(1, self.max_context_length - 1),
            add_special_tokens=False,
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        bos_column = torch.full(
            (input_ids.size(0), 1),
            self.bos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        
        input_ids = torch.cat([bos_column, input_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(bos_column), attention_mask], dim=1)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # 3. Forward Pass & Log Softmax
        outputs = self.model(input_ids)
        
        # Shape: [B, L, Vocab]
        all_log_probs = torch.log_softmax(outputs.logits, dim=-1)
        
        # 4. Shift and Gather
        # We want the probability of token at t, given context <t.
        # Logits at index [:-1] predict tokens at index [1:]
        
        # Align log probs and input_ids by shifting
        log_probs_shifted = all_log_probs[:, :-1, :] # Shape: [B, L-1, Vocab]
        target_ids = input_ids[:, 1:]                # Shape: [B, L-1]
        
        if len(contexts[0]) > 0:
            breakpoint()

        # Gather the log-prob of the ACTUAL token that appears in target_ids
        # gathered_probs[i, t] = score of token target_ids[i, t]
        gathered_probs = torch.gather(
            log_probs_shifted, 
            dim=2, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1) # Shape: [B, L-1]

        # 5. Reconstruct Scores
        beam_scores: List[List[float]] = [[] for _ in candidate_words]

        for batch_idx, (beam_idx, prefix_len) in enumerate(meta_info):
            # Calculate where the candidate word starts in the *shifted* array.
            # In input_ids, the new word starts at `prefix_len`.
            # In gathered_probs (which is shifted by 1), that corresponds to index `prefix_len - 1`.
            start = max(0, prefix_len - 1)
            
            # Calculate the end (ignoring padding)
            # -1 because gathered_probs is 1 shorter than input_ids
            valid_len = attention_mask[batch_idx].sum().item() - 1
            
            if start >= valid_len:
                # Context pushed word out of truncation window
                score = -float('inf')
            else:
                score = gathered_probs[batch_idx, start:valid_len].sum().item()

            beam_scores[beam_idx].append(score)

        return beam_scores
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self
