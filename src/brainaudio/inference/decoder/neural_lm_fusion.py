
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
import truecase

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
                   Example: [[-0.3, -4.6], [-1.2, -0.5]] for the inputs above.

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
        scores = [[0.0 for _ in words] for words in candidate_words]
        return scores


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
        scoring_chunk_size: int = 32,
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
            scoring_chunk_size: Maximum number of sequences to score in one forward pass.
                Reduces peak memory usage at the cost of slightly more compute. Default 32.
        """
        super().__init__(weight, homophone_aggregation, device)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_context_length = max_context_length
        self.word_insertion_bonus = word_insertion_bonus
        self.scoring_chunk_size = scoring_chunk_size
        
        self.device = device if device is not None else next(self.model.parameters()).device

        print(f"[HuggingFaceLMFusion] word_insertion_bonus={self.word_insertion_bonus}, weight={self.weight}")

        # Move model to device and set to eval mode
        # Skip .to() for quantized models as they may already be on the correct device
        if device is not None:
            current_device = next(self.model.parameters()).device
            if current_device != self.device:
                self.model.to(self.device)
        self.model.eval()
        

    @torch.no_grad()
    def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> List[List[float]]:
        """
        Score candidate words given contexts using chunked processing to limit memory usage.

        Processes sequences in chunks of size `self.scoring_chunk_size` to prevent
        OOM errors when scoring many sequences at once.
        """
        
        flat_texts = []
        # Store where the candidate word starts for each entry
        candidate_start_indices = []

        # 1. Prepare Texts
        for context, candidates in zip(contexts, candidate_words):
            for word in candidates:
                # Construct full text
                if not context:
                    full_text = word
                elif context.endswith(" ") or word.startswith(" "):
                    full_text = f"{context}{word}"
                else:
                    full_text = f"{context} {word}"

                # Apply truecase for proper capitalization before LLM scoring
                full_text = truecase.get_true_case(full_text)

                # Extract the truecased context (everything before the last word)
                # This ensures start_idx is computed from the same tokenization as full_text
                truecased_context = full_text.rsplit(" ", 1)[0] if " " in full_text else ""

                # Compute start_idx from the truecased context
                prefix_ids = self.tokenizer.encode(truecased_context, add_special_tokens=True)
                start_idx = len(prefix_ids) - 1

                flat_texts.append(full_text)
                candidate_start_indices.append(start_idx)

        if not flat_texts:
            return []

        # 2. Collect scores for each flat text (will be filled in by chunks)
        flat_scores = [0.0] * len(flat_texts)

        # 3. Process in chunks to limit memory usage
        chunk_size = self.scoring_chunk_size
        num_sequences = len(flat_texts)

        for chunk_start in range(0, num_sequences, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_sequences)
            chunk_texts = flat_texts[chunk_start:chunk_end]
            chunk_start_indices = candidate_start_indices[chunk_start:chunk_end]

            # Tokenize this chunk
            inputs = self.tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Forward pass for this chunk
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Shift and Gather
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)

            gathered_probs = torch.gather(
                log_probs,
                dim=2,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Extract scores for this chunk
            for i, (start_idx, attn) in enumerate(zip(chunk_start_indices, attention_mask)):
                valid_seq_len = attn.sum().item() - 1
                safe_start = min(start_idx, valid_seq_len)
                word_log_probs = gathered_probs[i, safe_start:valid_seq_len]
                score = word_log_probs.sum().item()
                flat_scores[chunk_start + i] = self.weight * score + self.word_insertion_bonus

            # Free memory immediately after processing each chunk
            del outputs, log_probs, gathered_probs, shift_logits, shift_labels
            del inputs, input_ids, attention_mask
            torch.cuda.empty_cache()

        # 4. Reshape flat_scores back to match candidate_words structure
        final_scores = []
        flat_idx = 0
        for candidates in candidate_words:
            beam_scores = []
            for _ in candidates:
                beam_scores.append(flat_scores[flat_idx])
                flat_idx += 1
            final_scores.append(beam_scores)

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
    prev_last_labels: torch.Tensor,
    homophone_prune_threshold: float | None = 10.0,
    frame_idx: Optional[int] = None
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
                new_text = f"{prev_text} {word}".strip()
                all_candidates.append((new_lm_score, new_text))
                
                #if word == "irish" or word == "heirs":
                #    print(f"Text: {prev_text}, Candidate word: {word}")
                #    print(f"Frame idx: {frame_idx}, Word: {word}, LM score: {word_lm_score}, Acoustic score: {base_score-prev_lm_score}")
                    # print(f"Debug: Scored 'royal' for beam (b={b}, k={k}), prev_text='{prev_text}', new_text='{new_text}', word_lm_score={word_lm_score}, new_lm_score={new_lm_score}")
                    # breakpoint()

        # Sort by accumulated LM score (descending)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Prune candidates that are too far below the best
        # This saves memory/compute by not tracking very unlikely homophones
        if homophone_prune_threshold is not None and all_candidates:
            best_score = all_candidates[0][0]
            all_candidates = [c for c in all_candidates if best_score - c[0] <= homophone_prune_threshold]

        # Keep top K (after pruning)
        new_tuples = all_candidates[:num_homophone_beams]

        # Update beam score using formula: score = acoustic_score + lm_score
        # Extract acoustic component by subtracting old LM, then add new LM
        old_best_lm_score = context_text_tuples[0][0] if context_text_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]
        acoustic_score = base_score - old_best_lm_score
        beam_hyps.scores[b, k] = acoustic_score + new_best_lm_score


        # Update context_texts with new tuples
        beam_hyps.context_texts[b][k] = new_tuples

        # Update hash based on best text (index 0)
        beam_hyps.context_texts_hash[b][k] = hash(new_tuples[0][1])


def apply_lm_end_of_sentence_scoring(
    lm_fusion: NeuralLanguageModelFusion | None,
    beam_hyps: 'BatchedBeamHyps',
) -> None:
    """
    Add end-of-sentence probability to beam scores.

    For each beam, scores the probability of a period "." following the
    current context text and adds it to the beam score. This helps the LM
    prefer complete, well-formed sentences.

    Args:
        lm_fusion: The neural LM fusion module (or None to skip).
        beam_hyps: The beam hypotheses to update in-place.
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape

    # Collect all contexts that need EOS scoring
    contexts = []
    beam_indices = []  # Track (batch_idx, beam_idx) for each context

    for b in range(batch_size):
        for k in range(beam_size):
            # Skip pruned beams
            if beam_hyps.scores[b, k] == float('-inf'):
                continue

            # Get the best text interpretation for this beam
            context_tuples = beam_hyps.context_texts[b][k]
            if not context_tuples:
                continue

            # Use the best (first) text interpretation
            _, text = context_tuples[0]

            # Skip if text already ends with a period
            if text.rstrip().endswith('.'):
                continue

            contexts.append(text)
            beam_indices.append((b, k))

    if not contexts:
        return

    # Score "." for all contexts in one batched call
    candidate_words = [["."]] * len(contexts)
    eos_scores = lm_fusion.score_continuations(contexts, candidate_words)

    # Add EOS scores to beam scores
    for (b, k), scores in zip(beam_indices, eos_scores):
        eos_score = scores[0]  # Only one candidate (".")
        beam_hyps.scores[b, k] += eos_score
