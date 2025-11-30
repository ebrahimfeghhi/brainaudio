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
    
    @torch.no_grad()
    def score_continuations(
        self, 
        contexts: List[str], 
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        """
        Score candidate words using the HuggingFace LM.
        
        Implementation:
            For each (context, candidate_word) pair:
            1. Tokenize: context + " " + candidate_word
            2. Get log-probs from LM
            3. Extract log-prob of the candidate word tokens
            4. Sum to get total log-prob of the word
        """
        if not contexts:
            return []

        beam_scores: List[List[float]] = [[] for _ in candidate_words]

        # Pre-tokenize contexts (used repeatedly per candidate word)
        encoded_contexts = [
            self.tokenizer.encode(ctx, add_special_tokens=False) if ctx else []
            for ctx in contexts
        ]
        
        if len(encoded_contexts[0]) > 1:
            
            breakpoint()
        space_token = self.tokenizer.encode(" ", add_special_tokens=False)

        flat_requests = []
        for beam_idx, (context, words) in enumerate(zip(contexts, candidate_words)):
            if not words:
                continue

            context_tokens = encoded_contexts[beam_idx]
            context_space_tokens = (
                self.tokenizer.encode(f"{context} ", add_special_tokens=False)
                if context else []
            )

            for word_idx, word in enumerate(words):
                full_text = f"{context} {word}" if context else word
                full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

                word_token_start = len(context_tokens)
                if context and full_tokens[word_token_start:word_token_start + 1] != space_token:
                    word_token_start = len(context_space_tokens)

                if len(full_tokens) > self.max_context_length:
                    offset = len(full_tokens) - self.max_context_length
                    full_tokens = full_tokens[offset:]
                    word_token_start = max(0, word_token_start - offset)

                flat_requests.append(
                    {
                        "beam_idx": beam_idx,
                        "word_idx": word_idx,
                        "tokens": full_tokens,
                        "word_start": word_token_start,
                    }
                )

        if not flat_requests:
            return beam_scores

        # Batch all context-word prompts into a single LM forward pass.
        max_len = max(len(req["tokens"]) for req in flat_requests)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full(
            (len(flat_requests), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for row, req in enumerate(flat_requests):
            tokens = torch.tensor(req["tokens"], dtype=torch.long, device=self.device)
            seq_len = tokens.size(0)
            input_ids[row, :seq_len] = tokens
            attention_mask[row, :seq_len] = True
            req["seq_len"] = seq_len

        outputs = self.model(input_ids, attention_mask=attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        for row, req in enumerate(flat_requests):
            seq_len = req["seq_len"]
            word_start = min(req["word_start"], seq_len)
            tokens = req["tokens"]
            score = 0.0
            for pos in range(word_start, seq_len):
                if pos == 0:
                    continue
                token_id = tokens[pos]
                score += log_probs[row, pos - 1, token_id].item()
            beam_scores[req["beam_idx"]].append(score)

        return beam_scores
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self
