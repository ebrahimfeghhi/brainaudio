
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


def _is_fresh_word_boundary(
    seq_raw: torch.Tensor,
    boundary_token: int | None,
    blank_index: int,
) -> bool:
    """
    Determine if seq_raw just reached a word boundary for the FIRST time.
    
    We want to trigger LM scoring exactly once per completed word. A word boundary
    is marked by the boundary_token (e.g., '|' silence marker). We trigger when:
    - The LAST token (not last non-blank) is exactly the boundary token, AND  
    - The PREVIOUS token was NOT the boundary token (to avoid re-triggering on repeats)
    
    This prevents double-counting when:
    - Blank tokens follow boundary: [..., |, BLANK] → last is BLANK, not |, so skip
    - Repeated boundaries: [..., |, |] → prev is |, so skip
    
    Args:
        seq_raw: Raw token sequence (with blanks and repeats)
        boundary_token: The word boundary token ID (e.g., silence '|')
        blank_index: The CTC blank token ID (unused but kept for API consistency)
        
    Returns:
        True if this is a fresh word boundary that should trigger LM scoring
    """
    if boundary_token is None or len(seq_raw) == 0:
        return False
    
    last_token = int(seq_raw[-1].item())
    
    # Must end with the boundary token
    if last_token != boundary_token:
        return False
    
    # If this is the first token, it's a fresh boundary
    if len(seq_raw) == 1:
        return True
    
    # Only trigger if previous token was NOT also the boundary (avoid re-triggering on repeats)
    prev_token = int(seq_raw[-2].item())
    return prev_token != boundary_token


def _can_complete_word_with_boundary(
    seq_ctc: list[int],
    lexicon: 'LexiconConstraint',
) -> tuple[bool, list[int]]:
    """
    Check if appending boundary_token to seq_ctc would complete a valid word.
    
    Args:
        seq_ctc: Collapsed CTC sequence (no blanks/repeats)
        lexicon: Lexicon constraint for word lookup
        
    Returns:
        Tuple of (can_complete, word_indices) where word_indices are the indices
        of words that would be completed.
    """
    boundary_token = getattr(lexicon, "word_boundary_token", None)
    if boundary_token is None:
        return False, []
    
    # Simulate appending boundary token
    seq_with_boundary = seq_ctc + [boundary_token]
    
    # Check if this completes a word
    _, at_boundary, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_with_boundary)
    
    return at_boundary and len(word_indices) > 0, word_indices


def apply_lm_fusion_pre_selection(
    lm_fusion: NeuralLanguageModelFusion | None,
    lexicon: 'LexiconConstraint',
    log_probs: torch.Tensor,
    beam_hyps: 'BatchedBeamHyps',
    beam_size: int,
    blank_index: int,
    curr_batch_size: int,
    token_to_symbol: dict | None = None,
    word_insertion_bonus: float = 0.0,
) -> torch.Tensor:
    """
    Apply neural LM fusion BEFORE beam selection by boosting the boundary token.
    
    This is the correct timing for LM fusion: we want the LM score to influence
    WHETHER a beam emits '|' (completing a word), not just what happens after.
    
    For each beam that could complete a word by emitting '|':
    1. Determine what word(s) would be completed (homophones)
    2. Score those candidates with the LM given the context
    3. Add the LM score to log_probs[b, k, boundary_token]
    
    This way, the LM score influences the topk selection that follows.
    
    Args:
        lm_fusion: NeuralLanguageModelFusion instance for scoring, or None to skip
        lexicon: LexiconConstraint for word boundary detection
        log_probs: [B, beam_size, V] current log probabilities (modified in-place)
        beam_hyps: BatchedBeamHyps containing sequences up to current timestep
        beam_size: Number of beams per batch element
        blank_index: Index of the CTC blank token
        curr_batch_size: Current batch size
        token_to_symbol: Optional dict mapping token ID to phoneme symbol
        
    Returns:
        log_probs: [B, beam_size, V] with LM scores added to boundary token
    """
    from .beam_helpers import (
        materialize_beam_transcript,
        collapse_ctc_sequence,
        decode_sequence_to_text,
        log_lm_watchlist_scores,
    )
    
    if lm_fusion is None:
        return log_probs
    
    # Get token_to_symbol mapping for text decoding
    if token_to_symbol is None and hasattr(lexicon, 'token_to_symbol'):
        token_to_symbol = lexicon.token_to_symbol
    
    boundary_token = getattr(lexicon, "word_boundary_token", None)
    if boundary_token is None:
        return log_probs
    
    for b in range(curr_batch_size):
        # Collect beams that could complete a word by emitting boundary_token
        # Each entry: (beam_idx, word_indices, context_text, candidate_words)
        completable_beams: list[tuple[int, list, str, list[str]]] = []

        for k in range(beam_size):
            # Get the raw and collapsed sequences for this beam
            seq_raw = materialize_beam_transcript(beam_hyps, b, k)
            seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)
            
            # Skip if beam is empty
            if len(seq_ctc) == 0:
                continue
            
            # Skip if we already just emitted a boundary (would be a repeat)
            # This prevents boosting '|' when we're already at a word boundary
            if seq_ctc[-1] == boundary_token:
                continue
            
            # Check if emitting '|' would complete a valid word
            can_complete, word_indices = _can_complete_word_with_boundary(seq_ctc, lexicon)
            if not can_complete:
                continue
            
            # Build context text (all completed words so far, NOT including the word we're about to complete)
            # The word we're completing is still in progress, so we want context before it
            context_text = decode_sequence_to_text(
                token_sequence=seq_ctc,
                lexicon=lexicon,
                token_to_symbol=token_to_symbol,
                exclude_last_word=True,  # Exclude the partial word we're about to complete
            )
            
            # Get the candidate words (homophones) that would be completed
            candidate_words = [lexicon.word_list[idx] for idx in word_indices]
            
            completable_beams.append((k, word_indices, context_text, candidate_words))

        # Batch score all completable beams together for efficiency
        if not completable_beams:
            continue
            
        contexts = [info[2] for info in completable_beams]
        candidate_lists = [info[3] for info in completable_beams]
        
        # Get LM scores for all candidate words
        all_lm_scores = lm_fusion.score_continuations(contexts, candidate_lists)
        
        # Add LM scores to the boundary token for each beam
        for (k, word_indices, context_text, candidate_words), lm_scores in \
                zip(completable_beams, all_lm_scores):
            
            # Aggregate scores across homophones (e.g., "to"/"too"/"two")
            combined_score = lm_fusion.aggregate_homophone_scores(lm_scores)
            
            # Debug logging if enabled
            log_lm_watchlist_scores(
                candidate_words=candidate_words,
                combined_score=combined_score,
                context=context_text,
            )
            
            if getattr(lm_fusion, "log_homophone_scores", False):
                print(
                    f"  [{b},{k}] context='{context_text}' completing={candidate_words} "
                    f"lm_score={combined_score:.4f} adjustment={lm_fusion.weight * combined_score:.4f}"
                )
            
            # Add weighted LM log-probability to the boundary token.
            #
            # LM scores are log-probabilities (negative, closer to 0 = more likely).
            # With positive weight, this PENALIZES completing words (more for unlikely):
            #   - common word (lm≈-2): small penalty → completing still viable
            #   - rare word (lm≈-15): large penalty → continuing preferred
            #
            # This behavior may seem counterintuitive. If you want to BOOST likely
            # words instead of penalizing unlikely ones, use a NEGATIVE weight
            # (though this changes the semantics significantly).
            #
            # TODO: Consider alternative formulations:
            #   1. Relative scoring: (lm_score - baseline) where baseline = avg log-prob
            #   2. Reward formulation: -lm_score (but then rare words get big rewards)
            #   3. Apply LM to all tokens (standard shallow fusion)
            log_probs[b, k, boundary_token] += lm_fusion.weight * combined_score
            # Apply word insertion bonus (helps balance LM penalty)
            log_probs[b, k, boundary_token] += word_insertion_bonus
    
    return log_probs


# Keep the old function name as an alias for backward compatibility
def apply_lm_fusion(
    lm_fusion: NeuralLanguageModelFusion | None,
    lexicon: 'LexiconConstraint',
    log_probs: torch.Tensor,
    beam_hyps: 'BatchedBeamHyps',
    beam_size: int,
    blank_index: int,
    curr_batch_size: int,
    token_to_symbol: dict | None = None,
    word_insertion_bonus: float = 0.0,
) -> torch.Tensor:
    """
    Apply neural language model fusion. Delegates to apply_lm_fusion_pre_selection.
    
    See apply_lm_fusion_pre_selection for full documentation.
    """
    return apply_lm_fusion_pre_selection(
        lm_fusion=lm_fusion,
        lexicon=lexicon,
        log_probs=log_probs,
        beam_hyps=beam_hyps,
        beam_size=beam_size,
        blank_index=blank_index,
        curr_batch_size=curr_batch_size,
        token_to_symbol=token_to_symbol,
        word_insertion_bonus=word_insertion_bonus,
    )
    


def apply_lm_fusion_post_selection(
    lm_fusion: NeuralLanguageModelFusion | None,
    lexicon: 'LexiconConstraint',
    beam_hyps: 'BatchedBeamHyps',
    blank_index: int,
    boundary_token: int,
    next_labels: torch.Tensor,
    prev_last_labels: torch.Tensor,
    token_to_symbol: dict | None = None,
    word_insertion_bonus: float = 0.0,
) -> None:
    
    from .beam_helpers import (
        materialize_beam_transcript,
        collapse_ctc_sequence,
        decode_sequence_to_text
    )
    
    """
    Apply LM fusion after top-k pruning and recombination.
    For each active beam, if the last emitted token is the boundary token and the previous token was not,
    apply LM fusion to the beam's score (once per word completion).
    Optionally add a word insertion bonus.
    Modifies beam_hyps.scores in-place.
    """
    
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    for b in range(batch_size):
        
        for k in range(beam_size):
            # Only consider active beams
            
            if beam_hyps.scores[b, k] == float('-inf'):
                continue
            
            last_label = int(next_labels[b, k].item())
            prev_last_label = int(prev_last_labels[b, k].item())
            
            if last_label != boundary_token:
                continue
            
            if prev_last_label == boundary_token:
                continue  # Already applied fusion for this word

            # Reconstruct the full token sequence for this beam
            seq_raw = materialize_beam_transcript(beam_hyps, b, k)
            seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)

            # Check if this is a valid word completion
            _, at_boundary, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_ctc)
            
            if not at_boundary or not word_indices:
                print("Something is wrong")
                breakpoint()
                continue

            # Build context text (all words before the just-completed one)
            context_text = decode_sequence_to_text(
                token_sequence=seq_ctc,
                lexicon=lexicon,
                token_to_symbol=token_to_symbol,
                exclude_last_word=True,
            )
            
            candidate_words = [lexicon.word_list[idx] for idx in word_indices]

            # Score with LM
            lm_scores = lm_fusion.score_continuations([context_text], [candidate_words])[0]
            combined_score = lm_fusion.aggregate_homophone_scores(lm_scores)

            if getattr(lm_fusion, "log_homophone_scores", False):
                print(
                    f"[POST] [{b},{k}] context='{context_text}' completing={candidate_words} "
                    f"lm_score={combined_score:.4f} adjustment={lm_fusion.weight * combined_score:.4f}"
                )

            # Apply LM fusion and word insertion bonus to the beam score
            beam_hyps.scores[b, k] += lm_fusion.weight * combined_score
            beam_hyps.scores[b, k] += word_insertion_bonus