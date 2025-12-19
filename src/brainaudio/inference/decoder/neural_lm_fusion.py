
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
from typing import List, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .lexicon_constraint import LexiconConstraint

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
        self.device = device
        
        if homophone_aggregation not in ['max', 'logsumexp']:
            raise ValueError(f"homophone_aggregation must be 'max' or 'logsumexp', got {homophone_aggregation}")
    
    @abstractmethod
    def score_continuations(
        self, 
        contexts: List[str], 
        candidate_words: List[List[str]]
    ) -> tuple[List[List[float]], List[List[int]]]:
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
            num_tokens: List of lists of token counts for each candidate word.
                   
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
    ) -> tuple[List[List[float]], List[List[int]]]:
        """Return uniform (zero) log-probability for all candidates and zero token counts."""
        scores = [[0.0 for _ in words] for words in candidate_words]
        num_tokens = [[0 for _ in words] for words in candidate_words]
        return scores, num_tokens


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
        word_insertion_bonus: float = 0.0, 
        lm_score: float = 1.0
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
        self.word_insertion_bonus = word_insertion_bonus
        self.lm_score = lm_score
        
        self.device = device if device is not None else next(self.model.parameters()).device
        
        # Move model to device and set to eval mode
        # Skip .to() for quantized models as they may already be on the correct device
        if device is not None:
            current_device = next(self.model.parameters()).device
            if current_device != self.device:
                self.model.to(self.device)
        self.model.eval()
        

    @torch.no_grad()
    def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> tuple[List[List[float]], List[List[int]]]:
        
        flat_texts = []
        # Store where the candidate word starts for each entry
        candidate_start_indices = [] 
        
        # 1. Prepare Texts
        for context, candidates in zip(contexts, candidate_words):
            
            # Robust Logic: Determine prefix length including Special Tokens
            # We assume the tokenizer adds BOS if configured to do so.
            # Note: We must replicate the 'add_special_tokens' behavior of the batch call.
            prefix_ids = self.tokenizer.encode(context, add_special_tokens=True)
            
            # The model predicts the NEXT token.
            # If prefix is [BOS, A, B], logits at last position predict C.
            # We want the scores starting FROM the prediction of the first candidate token.
            # Length of prefix is exactly the start index for the shifted logits.
            # e.g., Prefix len 3 ([BOS, A, B]). 
            # Shifted Labels: [A, B, C]. 
            # Indices: 0->A, 1->B, 2->C.
            # We want index 2 (C).
            # So start_idx = len(prefix_ids) - 1.
            start_idx = len(prefix_ids) - 1
            
            for word in candidates:
                # Construct full text
                if not context:
                    full_text = word.capitalize()
                elif context.endswith(" ") or word.startswith(" "):
                    full_text = f"{context}{word}"
                else:
                    full_text = f"{context} {word}"
                
                flat_texts.append(full_text)
                candidate_start_indices.append(start_idx)

        if not flat_texts:
            return [], []

        # 2. Batch Tokenize
        inputs = self.tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True 
        ).to(self.device)
        
        input_ids = inputs.input_ids 
        attention_mask = inputs.attention_mask 

        # 3. Forward Pass
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # 4. Shift and Gather
        # Logits[i] predicts Token[i+1]
        shift_logits = outputs.logits[..., :-1, :].contiguous() 
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # gathered_probs[b, t] is the log-prob of token t+1 given 0..t
        gathered_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1) 
        
        # 5. Extract Scores
        final_scores = []
        final_num_tokens = []
        flat_idx = 0
        
        for candidates in candidate_words:
            beam_scores = []
            num_tokens_beam = []
            
            for _ in candidates:
                start_idx = candidate_start_indices[flat_idx]
                
                # Find the end of the valid sequence (ignoring padding)
                # -1 because gathered_probs is 1 shorter than input
                valid_seq_len = attention_mask[flat_idx].sum().item() - 1
                
                # Safety clamp (in case context + candidate merged weirdly and became shorter than context)
                safe_start = min(start_idx, valid_seq_len)
                
                # Slice: From end of context -> End of word
                word_log_probs = gathered_probs[flat_idx, safe_start:valid_seq_len]
                
                score = word_log_probs.sum().item()
                
                # FIX: Return length of the *candidate word only*
                n_tokens = valid_seq_len - safe_start
                
                beam_scores.append(self.lm_score * score + self.word_insertion_bonus)
                
                flat_idx += 1
                
            final_scores.append(beam_scores)
            final_num_tokens.append(num_tokens_beam)

        return final_scores
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self
    
def apply_lm_fusion_post_selection(
    lm_fusion: NeuralLanguageModelFusion | None,
    lexicon: 'LexiconConstraint',
    beam_hyps: 'BatchedBeamHyps',
    blank_index: int,
    boundary_token: int,
    next_labels: torch.Tensor,
    prev_last_labels: torch.Tensor
) -> None:
    
    from .beam_helpers import (
        materialize_beam_transcript,
        collapse_ctc_sequence
    )
    
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    to_score = []  # List to store: (b, k, context_text, candidate_words)
    
    # --- PHASE 1: Identify beams that need LM Scoring ---
    for b in range(batch_size):
        for k in range(beam_size):
            
            # Skip empty or pruning beams
            if beam_hyps.scores[b, k] == float('-inf'):
                continue
            
            # Check if we just crossed a word boundary
            last_label = int(next_labels[b, k].item())
            prev_last_label = int(prev_last_labels[b, k].item())
            
            if last_label != boundary_token:
                continue
            if prev_last_label == boundary_token:
                continue
            
            # Get the sequence to find valid words
            seq_raw = materialize_beam_transcript(beam_hyps, b, k)
            seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)
            
            # Note: Assuming lexicon handles suffix extraction automatically as discussed
            _, at_boundary, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_ctc)
            
            if not at_boundary or not word_indices:
                continue
            
            # Get the list of text interpretations for this beam
            # context_text_tuples: List[Tuple[float, str]] where each tuple is (lm_score, text)
            # Currently this list has 1 element (will grow up to K when homophones are encountered)
            context_text_tuples = beam_hyps.context_texts[b][k]
            candidate_words = [lexicon.word_list[idx] for idx in word_indices]

            to_score.append((b, k, context_text_tuples, candidate_words))

    if not to_score:
        return

    # --- PHASE 2: Batch LM Scoring ---
    # We need to score EACH text interpretation from each beam against all its homophones.
    # Example: beam (b=0, k=3) has 2 text interpretations and 3 candidate homophones
    #          -> we need 2 LM calls for this beam (one per text interpretation)
    #
    # Flatten all texts into a single list for one batched LM call:
    # - flat_contexts[i] = text string to use as context
    # - flat_candidates[i] = list of candidate words to score
    # - beam_mapping[i] = (b, k, tuple_idx) to know where to put the results back
    flat_contexts = []
    flat_candidates = []
    beam_mapping = []  # Maps flat index -> (batch_idx, beam_idx, tuple_idx)

    for (b, k, context_text_tuples, candidate_words) in to_score:
        for tuple_idx, (lm_score, text) in enumerate(context_text_tuples):
            flat_contexts.append(text)
            flat_candidates.append(candidate_words)
            beam_mapping.append((b, k, tuple_idx))

    # Single batched LM call - scores all contexts against their candidate words
    all_lm_scores = lm_fusion.score_continuations(flat_contexts, flat_candidates)

    # Reorganize results back by (b, k) for easy lookup
    # scores_by_beam[(b, k)][tuple_idx] = list of scores for each candidate word
    scores_by_beam = {}
    for flat_idx, (b, k, tuple_idx) in enumerate(beam_mapping):
        key = (b, k)
        if key not in scores_by_beam:
            scores_by_beam[key] = {}
        scores_by_beam[key][tuple_idx] = all_lm_scores[flat_idx]
    
    # --- PHASE 3: Update Beams ---
    # For each beam, combine each text interpretation with each homophone candidate.
    # Keep top K candidates (sorted by LM score for this word).
    # Update beam's acoustic score by adding the best candidate's LM score.
    num_homophone_beams = beam_hyps.num_homophone_beams

    for (b, k, context_text_tuples, candidate_words) in to_score:
        base_score = beam_hyps.scores[b, k].item()
        lm_scores_dict = scores_by_beam[(b, k)]

        # Collect all (lm_score, text) candidates by combining each text with each homophone
        # Accumulate LM scores so we track full sequence probability
        all_candidates = []
        for tuple_idx, (prev_lm_score, prev_text) in enumerate(context_text_tuples):
            lm_scores_for_tuple = lm_scores_dict[tuple_idx]
            for word, word_lm_score in zip(candidate_words, lm_scores_for_tuple):
                # Accumulate: new score = previous accumulated LM + this word's LM score
                new_lm_score = prev_lm_score + word_lm_score
                w_add = word.capitalize() if not prev_text.strip() else word
                new_text = f"{prev_text} {w_add}".strip()
                all_candidates.append((new_lm_score, new_text))

        # Sort by accumulated LM score (descending) and keep top K
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        new_tuples = all_candidates[:num_homophone_beams]

        # Update beam score: add delta between new and old best LM scores
        # beam_hyps.scores contains acoustic + previous LM, so we add only the new LM contribution
        old_best_lm_score = context_text_tuples[0][0] if context_text_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]
        lm_score_delta = new_best_lm_score - old_best_lm_score
        beam_hyps.scores[b, k] = base_score + lm_score_delta

        # Update context_texts with new tuples
        beam_hyps.context_texts[b][k] = new_tuples

        # Update hash based on best text (index 0)
        beam_hyps.context_texts_hash[b][k] = hash(new_tuples[0][1])

   