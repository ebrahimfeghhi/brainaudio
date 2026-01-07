# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import List, Optional, Union
from typing import Any

import numpy as np
import torch

from .lexicon_constraint import LexiconConstraint
from .batched_beam_decoding_utils import BatchedBeamHyps
from .asr_confidence_utils import ConfidenceMethodMixin
from .optional_cuda_graphs import WithOptionalCudaGraphs
from .cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from .enum import PrettyStrEnum
from .nemo_stubs import logging
from .neural_lm_fusion import NeuralLanguageModelFusion
from .context_biasing import GPUBoostingTreeModel

HAVE_LM_FUSION = False

try:
    from cuda.bindings import runtime as cudart
    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


NON_EXISTENT_LABEL_VALUE = -1
INACTIVE_SCORE = float("-inf")


class BacthedBeamCTCState:
    """
    State for Batched Beam Search for CTC models. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    beam_size: int  # (maximum) length of internal storage for beam dimension
    blank_index: int  # the index of the blank token

    decoder_outputs: torch.Tensor  # logprobs from decoder
    decoder_output_lengths: torch.Tensor  # lengths of the decoder outputs (i.e. max time for each utterance)
    last_timesteps: torch.Tensor  # last time step for each utterance (used to check if the decoding is finished)

    vocab: torch.Tensor  # vocabulary of the model. Constant
    vocab_blank_mask: torch.Tensor  # mask for blank token in the vocabulary. Constant

    curr_frame_idx: torch.Tensor  # current frame index for each utterance (used to check if the decoding is finished)
    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')

    batched_hyps: BatchedBeamHyps  # batched hypotheses - decoding result
    
      # fusion models related fields
    fusion_models: Optional[List["NGramGPULanguageModel"]] = None  # Forward reference to avoid circular import
    fusion_states_list: Optional[List[torch.Tensor]] = None
    fusion_states_candidates_list: Optional[List[torch.Tensor]] = None

    # fusion models related fields

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_time: int,
        vocab_size: int,
        device: torch.device,
        float_dtype: torch.dtype,
        blank_index: int,
        num_homophone_beams: int = 1,
    ):
        """
        Args:
            batch_size: batch size for encoder output storage
            beam_size: beam size for decoder output storage
            max_time: maximum time for encoder output storage
            vocab_size: vocabulary size of the model including blank
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            blank_index: index of the blank symbol
            num_homophone_beams: number of text interpretations (homophones) to track per beam
        """

        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_time = max_time
        self.blank_index = blank_index
        self.vocab_size = vocab_size
        self.num_homophone_beams = num_homophone_beams

        self.NON_EXISTENT_LABEL = torch.tensor(NON_EXISTENT_LABEL_VALUE, device=self.device, dtype=torch.long)
        self.BLANK_TENSOR = torch.tensor(self.blank_index, device=self.device, dtype=torch.long)
        self.INACTIVE_SCORE = torch.tensor(INACTIVE_SCORE, device=self.device, dtype=float_dtype)

        # Storage for all decoder log probs the CUDA graph might ever replay.
        self.decoder_outputs = torch.zeros(
            (self.batch_size, self.max_time, self.vocab_size),
            dtype=float_dtype,
            device=self.device,
        )
        self.decoder_output_lengths = torch.zeros(
            (self.batch_size, self.beam_size), dtype=torch.long, device=self.device
        )
        self.last_timesteps = torch.zeros((self.batch_size, self.beam_size), dtype=torch.long, device=self.device)

        self.vocab = torch.arange(self.vocab_size, device=self.device, dtype=torch.long)
        self.vocab_blank_mask = torch.eq(self.vocab, self.blank_index)

        # Running pointers that let us resume decoding when the outer Python code replays a graph.
        self.curr_frame_idx = torch.zeros([self.beam_size], device=self.device, dtype=torch.long)
        self.active_mask = torch.zeros((batch_size, self.beam_size), device=self.device, dtype=torch.bool)
        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self.blank_index,
            init_length=max_time + 1,
            device=device,
            float_dtype=float_dtype,
            model_type='ctc',
        )

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsBatchedBeamCTC:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    _before_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _after_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


class BatchedBeamCTCComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Batched beam search implementation for CTC models.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_beam_batch_conditional_ctc.cu"

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsBatchedBeamCTC]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[BacthedBeamCTCState]
    def __init__(
        self,
        blank_index: int,
        beam_size: int,
        return_best_hypothesis: bool = True,
        preserve_alignments=False,
        compute_timestamps: bool = False,
        beam_beta: float = 0.0,
        beam_blank_penalty: float = 0.0,
        beam_threshold: float = 10.0,
        allow_cuda_graphs: bool = True,
        lexicon: LexiconConstraint = None,
        lm_fusion: NeuralLanguageModelFusion = None,
        lm_beam_limit: Optional[int] = None,
        num_homophone_beams: int = 1,
        homophone_prune_threshold: Optional[float] = 10.0,
        fusion_models: Optional[List] = None,
        fusion_models_alpha: Optional[List[float]] = None, 
        kv_cache: Optional[Any] = None
    ):
        """
        Init method.
        Args:
            blank_index: index of blank symbol.
            beam_size: beam size.
            return_best_hypothesis: whether to return the best hypothesis or N-best hypotheses.
            preserve_alignments: if alignments are needed. Defaults to False.
            compute_timestamps: if timestamps are needed. Defaults to False.
            beam_beta: bonus for extending beams (emitting new non-blank, non-repeat tokens).
            beam_blank_penalty: penalty subtracted from blank emissions.
            beam_threshold: threshold for pruning candidates.
            allow_cuda_graphs: whether to allow CUDA graphs. Defaults to True.
            lexicon: optional LexiconConstraint to constrain beam search to valid words. Defaults to None.
            lm_fusion: optional NeuralLanguageModelFusion for word-level LM rescoring. Defaults to None.
            lm_beam_limit: optional cap on how many beams per batch element receive neural LM scoring at a word boundary.
            num_homophone_beams: number of text interpretations (homophones) to track per beam. Defaults to 1.
            homophone_prune_threshold: if set, prune homophone candidates whose LM score is more than this
                many log-prob points below the best candidate. Defaults to 10.0. Set to None to disable.
        """

        super().__init__()
        self._blank_index = blank_index

        self.beam_size = beam_size
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps
        self.allow_cuda_graphs = allow_cuda_graphs
        self.return_best_hypothesis = return_best_hypothesis

        self.beam_beta = beam_beta
        self.beam_blank_penalty = beam_blank_penalty
        self.beam_threshold = beam_threshold

        assert not self.preserve_alignments, "Preserve aligments is not supported"

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        self.cuda_graphs_mode = None
        self.maybe_enable_cuda_graphs()

        self.lexicon = lexicon
        self.lm_fusion = lm_fusion
        self.lm_beam_limit = lm_beam_limit
        self.num_homophone_beams = num_homophone_beams
        self.homophone_prune_threshold = homophone_prune_threshold

        # Warn if lm_fusion is used without lexicon
        if self.lm_fusion is not None and self.lexicon is None:
            logging.warning("lm_fusion is enabled but lexicon is None. Neural LM fusion requires lexicon for word boundary detection.")

        # Frame-level fusion models (e.g., phoneme n-gram LM)
        self.fusion_models = fusion_models or []
        self.fusion_models_alpha = fusion_models_alpha or [1.0] * len(self.fusion_models)

        self.kv_cache = kv_cache

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.state = None

    def maybe_enable_cuda_graphs(self) -> bool:
        """Enable CUDA graphs if conditions met"""
        if self.cuda_graphs_mode is not None:
            # CUDA graphs are already enabled
            return False

        if not self.allow_cuda_graphs:
            self.cuda_graphs_mode = None
        else:
            # cuda graphs are allowed
            # check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                logging.warning(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_GRAPHS
        self.reset_cuda_graphs_state()
        return self.cuda_graphs_mode is not None

    def disable_cuda_graphs(self) -> bool:
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if self.cuda_graphs_mode is None:
            # nothing to disable
            return False
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()
        return True

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    @torch.no_grad()
    def batched_beam_search_torch(
        self, decoder_outputs: torch.Tensor, decoder_output_lengths: torch.Tensor
    ) -> BatchedBeamHyps:
        """
        Pure PyTorch implementation of the batched beam search algorithm.

        Args:
            decoder_outputs (torch.Tensor): Tensor of shape [B, T, V+1], where B is the batch size,
                T is the maximum sequence length, and V is the vocabulary size. The tensor contains log-probabilities.
            decoder_output_lengths (torch.Tensor): Tensor of shape [B], contains lengths of each sequence in the batch.
        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """

        curr_batch_size, curr_max_time, vocab_size = decoder_outputs.shape

        lexicon_state = None
        if self.lexicon is not None and getattr(self.lexicon, "supports_state_tracking", False):
            # lexicon_state is of shape (batch_size, beam_size)
            lexicon_state = self.lexicon.initialize_state(
                batch_size=curr_batch_size,
                beam_size=self.beam_size,
                device=decoder_outputs.device,
            )

        # All tensors below stay on the same device/dtype as logits so we avoid syncs later.

        # Create vocabulary index tensor [0, 1, 2, ..., V] to use for vectorized operations
        # This lets us check token properties without loops
        vocab = torch.arange(vocab_size, device=decoder_outputs.device, dtype=torch.long)
        
        # Precompute a boolean mask marking where the blank token sits in the vocab
        # Shape: [V+1], True only at position self._blank_index
        vocab_blank_mask = vocab == self._blank_index

        # Initialize the data structure that will hold all beam hypotheses for all utterances
        # BatchedBeamHyps manages:
        #   - scores: [B, beam_size] current cumulative log-prob for each beam
        #   - last_label: [B, beam_size] the last emitted (non-blank) token for each beam
        #   - sequences: the full token sequences for each hypothesis
        # init_length is the max possible sequence length (one token per frame)
        batched_beam_hyps = BatchedBeamHyps(
            batch_size=curr_batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=curr_max_time + 1,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
            model_type='ctc',
            num_homophone_beams=self.num_homophone_beams,
            score_combination='max', 
            kv_cache=self.kv_cache
        )
        
         # init fusion models
        if self.fusion_models is not None:
            fusion_states_list = []
            fusion_states_candidates_list = []
            for fusion_model in self.fusion_models:
                fusion_model.to(decoder_outputs.device)
                fusion_states_list.append(
                    fusion_model.get_init_states(batch_size=curr_batch_size * self.beam_size, bos=True)
                )
                fusion_states_candidates_list.append(None)

        
        # Main decoding loop: process one acoustic frame at a time
        for frame_idx in range(curr_max_time):
            
            # active_mask: [B, beam_size] - True if this utterance hasn't ended yet at this frame
            # decoder_output_lengths[b] tells us the actual (unpadded) length of utterance b
            # If frame_idx >= length, that utterance is done and we shouldn't update its beams
            active_mask = frame_idx < decoder_output_lengths.unsqueeze(1)

            # repeated_mask: [B, beam_size, V+1] - True where a vocab token equals the last emitted token
            # batched_beam_hyps.last_label is [B, beam_size], we broadcast to compare against all vocab
            # This identifies which tokens would be *repetitions* of the previous non-blank token
            repeated_mask = batched_beam_hyps.last_label[:, :, None] == vocab[None, None, :]
            
            # repeated_or_blank_mask: [B, beam_size, V+1] - True for blanks OR repeated tokens
            # CTC rule: emitting blank or repeating the last token doesn't add a new symbol
            repeated_or_blank_mask = repeated_mask | vocab_blank_mask[None, None, :]

            # step 2.1: getting the log probs and updating with fusion scores
            # log probs is of shape batch size [B, beam_size, V+1]
            log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).repeat(1, self.beam_size, 1)
            log_probs += batched_beam_hyps.scores.unsqueeze(-1) # add current beam scores to every token in our beam
            
                # step 2.2: updating non-blank and non-repeating token scores with `beam_beta`
            #if frame_idx == curr_max_time - 5:  # First padding frame
            #    breakpoint()
            # step 2.2: updating non-blank and non-repeating token scores with `beam_beta`

                
            log_probs = torch.where(repeated_or_blank_mask, log_probs, log_probs + self.beam_beta)

            # step 2.2.1: apply blank penalty
            log_probs = torch.where(vocab_blank_mask[None, None, :], log_probs - self.beam_blank_penalty, log_probs)

            # step 2.2.5: apply lexicon constraints if provided
            # This masks out tokens that would create invalid words according to the lexicon
            if self.lexicon is not None:
                # Vectorized lexicon maintains internal state (prefix) per beam
                if lexicon_state is not None:
                    lexicon_mask = self.lexicon.get_constraint_mask_with_state(
                        state=lexicon_state,
                        vocab_size=vocab_size,
                        last_labels=batched_beam_hyps.last_label,
                    )
                else:
                    lexicon_mask = self.lexicon.get_constraint_mask(
                        sequences=batched_beam_hyps.transcript_wb,
                        last_labels=batched_beam_hyps.last_label,
                        vocab_size=vocab_size,
                    )
                
                # Set invalid tokens to -inf so they won't be selected by topk
                log_probs = torch.where(lexicon_mask, log_probs, INACTIVE_SCORE)
                
            
            # ==================== STEP 2.2: FUSION MODEL (N-GRAM LM) SCORING ====================
            # This section integrates phoneme-level language model scores into the beam search.
            # The LM provides P(next_phoneme | history) to bias the search toward likely phoneme sequences.

            if self.fusion_models is not None:
                for fusion_idx, fusion_model in enumerate(self.fusion_models):

                    # ----- ADVANCE THE LM: Get scores and next states for ALL possible phonemes -----
                    # fusion_states_list[fusion_idx] contains the current LM state for each beam.
                    # Shape: [B * beam_size] (flattened). Each state encodes the n-gram history.
                    #
                    # fusion_model.advance() returns:
                    #   - fusion_scores: log P(phoneme | state) for each of the 40 phonemes
                    #     Shape: [B * beam_size, 40] — one score per phoneme per beam
                    #   - fusion_states_candidates: the LM state AFTER emitting each phoneme
                    #     Shape: [B * beam_size, 40] — one next-state per phoneme per beam
                    #     Example: if current state represents history "AH K", then:
                    #       fusion_states_candidates[:, 0] = state for "AH K AA"
                    #       fusion_states_candidates[:, 1] = state for "AH K AE"
                    #       ... and so on for all 40 phonemes
                    fusion_scores, fusion_states_candidates = fusion_model.advance(
                        states=fusion_states_list[fusion_idx].view(-1)
                    )

                    # ----- ZERO OUT LM SCORES FOR REPEATED TOKENS -----
                    # In CTC, emitting the same token twice in a row (e.g., "AA AA") does NOT produce
                    # two phonemes — it just extends the duration of one phoneme. The LM should not
                    # score these repeats because no new phoneme is being added to the sequence.
                    #
                    # repeated_mask: [B, beam_size, vocab_size] — True where token == last_token
                    # repeated_mask[..., 1:]: [B, beam_size, 40] — same but excluding blank (index 0)
                    #
                    # torch.where(condition, value_if_true, value_if_false):
                    #   - Where repeated_mask is True (repeat) → use 0 (no LM contribution)
                    #   - Where repeated_mask is False (new phoneme) → use actual LM score
                    #
                    # fusion_scores is reshaped from [B*beam, 40] to [B, beam, 40] to match repeated_mask
                    fusion_scores = torch.where(
                        repeated_mask[..., 1:], 0, fusion_scores.view(curr_batch_size, self.beam_size, -1)
                    )

                    # ----- ADD WEIGHTED LM SCORES TO ACOUSTIC SCORES -----
                    # log_probs: [B, beam_size, vocab_size] = [B, beam, 41] — acoustic model scores
                    # log_probs[..., 1:]: [B, beam, 40] — non-blank tokens only (CTC indices 1-40)
                    #
                    # Final score = acoustic_score + alpha * lm_score
                    # where alpha (fusion_models_alpha) controls how much the LM influences decoding.
                    # Higher alpha = LM has more influence; lower alpha = trust acoustic model more.
                    #
                    # Note: We don't add LM scores to blank (index 0) because blank doesn't represent
                    # a phoneme — it's just a CTC mechanism for handling timing/alignment.
                    log_probs[..., 1:] += self.fusion_models_alpha[fusion_idx] * fusion_scores.view(
                        curr_batch_size, self.beam_size, -1
                    )

                    # ----- STORE CANDIDATE STATES FOR LATER -----
                    # After topk selection (step 2.3), we'll need to know what LM state to transition
                    # to based on which token was actually selected. We store all 40 possible next
                    # states here and will pick the right one in step 2.4.
                    fusion_states_candidates_list[fusion_idx] = fusion_states_candidates

            """
                step 2.3: getting `beam_size` best candidates
                log_probs.view(curr_batch_size, -1) reshapes log_probs to [B, beam_size * (V+1)]
            """
            
            next_scores, next_candidates_indices = torch.topk(
                log_probs.view(curr_batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            
            next_indices = next_candidates_indices // vocab_size # indices of beams being extended
            next_labels = next_candidates_indices % vocab_size # label indices
                    
            # step 2.3: pruning candidates with threshold `
            batch_next_scores = next_scores.view(curr_batch_size, -1)
            max_next_score = batch_next_scores.max(dim=-1, keepdim=True).values
            batch_next_scores.masked_fill_(batch_next_scores <= max_next_score - self.beam_threshold, INACTIVE_SCORE)
            next_scores = batch_next_scores.view(curr_batch_size, self.beam_size)

            inactive_candidate_mask = torch.isinf(next_scores) & (next_scores < 0)
            
            if inactive_candidate_mask.any():
                
                next_labels = torch.where(
                    inactive_candidate_mask,
                    next_labels.new_full(next_labels.shape, NON_EXISTENT_LABEL_VALUE),
                    next_labels,
                )
                next_scores = torch.where(
                    inactive_candidate_mask,
                    next_scores.new_full(next_scores.shape, INACTIVE_SCORE),
                    next_scores,
                )
            
            prev_last_labels = torch.gather(batched_beam_hyps.last_label, dim=-1, index=next_indices)

            # ==================== STEP 2.4: UPDATE FUSION MODEL (LM) STATES ====================
            # After topk selection, we know which beams survived and what tokens they emitted.
            # Now we need to update the LM state for each beam to reflect the newly emitted phoneme.
            #
            # Key insight: The LM state only changes when a NEW phoneme is emitted. It does NOT
            # change for blanks or repeated tokens (CTC merging behavior).

            if self.fusion_models is not None:

                # ----- DETERMINE WHICH BEAMS SHOULD PRESERVE THEIR LM STATE -----
                # last_labels: the previous token for each selected beam's parent
                # Shape: [B, beam_size]
                # We use next_indices to look up which parent beam each new beam came from.
                last_labels = torch.gather(batched_beam_hyps.last_label, dim=-1, index=next_indices)

                # blank_mask: True if this beam emitted a blank token
                # Shape: [B, beam_size]
                # Blank = CTC's way of saying "no phoneme at this frame"
                blank_mask = next_labels == self._blank_index

                # repeating_mask: True if this beam emitted the same token as its parent's last token
                # Shape: [B, beam_size]
                # In CTC, "AA AA" collapses to single "AA" — no new phoneme emitted
                repeating_mask = next_labels == last_labels

                # preserve_state_mask: True if we should KEEP the old LM state (not update it)
                # Shape: [B, beam_size]
                # We preserve state when:
                #   - repeating_mask: repeated token (CTC merge, no new phoneme)
                #   - blank_mask: blank emission (no phoneme)
                #   - ~active_mask: utterance is finished
                #   - inactive_candidate_mask: beam was pruned (has -inf score, next_labels = -1)
                preserve_state_mask = repeating_mask | blank_mask | ~active_mask | inactive_candidate_mask

                # ----- CONVERT CTC INDICES TO LM INDICES -----
                # CTC vocabulary: [0=blank, 1=AA, 2=AE, ..., 40=SIL] — 41 tokens
                # LM vocabulary:  [0=AA, 1=AE, ..., 39=SIL] — 40 tokens (no blank)
                #
                # Mapping: CTC index i → LM index (i - 1)
                # Example: CTC index 1 (AA) → LM index 0
                #          CTC index 40 (SIL) → LM index 39
                #
                # For blanks (CTC index 0), we use 0 as a placeholder. This value won't actually
                # be used because preserve_state_mask will be True for blanks.
                next_labels_lm = torch.where(blank_mask, 0, next_labels - 1)

                # Clamp to valid range [0, 39] to prevent index-out-of-bounds errors.
                # This handles edge cases like inactive beams with invalid label values.
                next_labels_lm = next_labels_lm.clamp(min=0)

                # ----- GATHER AND UPDATE LM STATES FOR EACH FUSION MODEL -----
                for fusion_idx, fusion_model in enumerate(self.fusion_models):

                    # --- Step A: Reshape candidate states for gathering ---
                    # fusion_states_candidates_list[fusion_idx] was computed in step 2.2.
                    # Original shape: [B * beam_size, 40] — flat batch, one row per beam, 40 next-states
                    #
                    # next_indices tells us which PARENT BEAM each selected candidate came from.
                    # Shape: [B, beam_size] — values are in range [0, beam_size-1]
                    #
                    # To gather along the beam dimension, we need to expand next_indices to match
                    # the last dimension (40 phonemes).
                    # next_indices_extended: [B, beam_size, 40]
                    next_indices_extended = next_indices[:, :, None].expand(
                        curr_batch_size, self.beam_size, fusion_states_candidates_list[fusion_idx].shape[-1]
                    )

                    # Reshape from [B*beam, 40] to [B, beam, 40] for gathering
                    fusion_states_candidates = fusion_states_candidates_list[fusion_idx].view(
                        curr_batch_size, self.beam_size, -1
                    )

                    # --- Step B: Gather candidate states from parent beams ---
                    # For each selected beam, get the 40 candidate next-states from its PARENT beam.
                    # Before: fusion_states_candidates[b, parent_beam, phoneme] = state after emitting phoneme
                    # After:  fusion_states_candidates[b, new_beam, phoneme] = state after emitting phoneme
                    #         (now indexed by new beam position, but containing parent's candidates)
                    #
                    # Example: If new beam 0 came from parent beam 3, then after this gather:
                    #   fusion_states_candidates[0, 0, :] = old fusion_states_candidates[0, 3, :]
                    fusion_states_candidates = torch.gather(
                        fusion_states_candidates, dim=1, index=next_indices_extended
                    )

                    # --- Step C: Get previous LM states (for preserve_state_mask cases) ---
                    # fusion_states_list[fusion_idx]: current LM states, shape [B * beam_size]
                    # Reshape to [B, beam_size] and gather using next_indices to get parent states.
                    # These are the states we'll use when preserve_state_mask is True.
                    fusion_states_prev = torch.gather(
                        fusion_states_list[fusion_idx].view(curr_batch_size, self.beam_size), dim=1, index=next_indices
                    )

                    # --- Step D: Select the correct next state based on emitted token ---
                    # fusion_states_candidates: [B, beam_size, 40] — 40 possible next states per beam
                    # next_labels_lm: [B, beam_size] — which phoneme was actually emitted (LM index)
                    #
                    # We gather along the last dimension to pick the state corresponding to the
                    # actually emitted phoneme.
                    # next_labels_lm.unsqueeze(-1): [B, beam_size, 1] — index for gather
                    # After gather: [B, beam_size, 1] → squeeze → [B, beam_size]
                    #
                    # Example: If beam emitted phoneme with LM index 5, we get:
                    #   fusion_states[b, beam] = fusion_states_candidates[b, beam, 5]
                    fusion_states = torch.gather(
                        fusion_states_candidates, dim=-1, index=next_labels_lm.unsqueeze(-1)
                    ).squeeze(-1)

                    # --- Step E: Apply preserve_state_mask to decide final state ---
                    # Where preserve_state_mask is True (blank/repeat/inactive):
                    #   → Keep the previous state (fusion_states_prev)
                    # Where preserve_state_mask is False (new phoneme emitted):
                    #   → Use the new state (fusion_states)
                    #
                    # Finally, flatten back to [B * beam_size] for the next frame.
                    fusion_states_list[fusion_idx] = torch.where(
                        preserve_state_mask, fusion_states_prev, fusion_states
                    ).view(-1)

            
            # step 2.5: masking inactive hypotheses, updating + recombining batched beam hypoteses
            next_labels = torch.where(active_mask, next_labels, NON_EXISTENT_LABEL_VALUE)

            if lexicon_state is not None:
                # gather parent lexicon states for the selected beams
                parent_states = torch.gather(lexicon_state, dim=1, index=next_indices)
                # update lexicon states based on the emitted labels
                lexicon_state = self.lexicon.update_state(
                    parent_state=parent_states,
                    emitted_labels=next_labels,
                    prev_last_labels=prev_last_labels,
                )
                lexicon_state = torch.where(inactive_candidate_mask, parent_states, lexicon_state)
                
            batched_beam_hyps.add_results_(next_indices, next_labels, next_scores)
            batched_beam_hyps.recombine_hyps_(is_last_step= curr_max_time-1 == frame_idx)

        
            # Apply LM fusion post-selection (after pruning and recombination)
            if self.lm_fusion is not None and self.lexicon is not None:

                from .neural_lm_fusion import apply_lm_fusion_post_selection

                boundary_token = getattr(self.lexicon, "word_boundary_token", None)

                apply_lm_fusion_post_selection(
                    lm_fusion=self.lm_fusion,
                    lexicon=self.lexicon,
                    beam_hyps=batched_beam_hyps,
                    blank_index=self._blank_index,
                    boundary_token=boundary_token,
                    next_labels=next_labels,
                    prev_last_labels=prev_last_labels,
                    homophone_prune_threshold=self.homophone_prune_threshold,
                    frame_idx=frame_idx
                )

     
         # step 3: updating fusoin scores with eos scores
        if self.fusion_models is not None:
            for fusion_idx, fusion_model in enumerate(self.fusion_models):
                # only GPUBoostingTreeModel does not support eos scores for CTC models by default
                if not isinstance(fusion_model, GPUBoostingTreeModel):
                    eos_score = fusion_model.get_final(fusion_states_list[fusion_idx]).view(
                        batched_beam_hyps.scores.shape
                    )
                    batched_beam_hyps.scores += eos_score * self.fusion_models_alpha[fusion_idx]
                    
           # After decoding is complete, add end-of-sentence probability (with incomplete word handling)
        if self.lm_fusion is not None:
            print("Applying LM end-of-sentence scoring...")
            from .neural_lm_fusion import apply_lm_end_of_sentence_with_incomplete_word
            apply_lm_end_of_sentence_with_incomplete_word(
                lm_fusion=self.lm_fusion,
                lexicon=self.lexicon,
                beam_hyps=batched_beam_hyps,
                blank_index=self._blank_index,
            )
            

        return batched_beam_hyps
   
    
    def batched_beam_search_cuda_graphs(
        self,
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ) -> BatchedBeamHyps:
        """
        Cuda-Graphs implementation of the batched beam search algorithm.

        Args:
            decoder_outputs (torch.Tensor): Tensor of shape [B, T, V+1], where B is the batch size,
                T is the maximum sequence length, and V is the vocabulary size. The tensor contains log-probabilities.
            decoder_output_lengths (torch.Tensor): Tensor of shape [B], contains lengths of each sequence in the batch.
        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """

        assert self.cuda_graphs_mode is not None

        curr_batch_size, curr_max_time, _ = decoder_outputs.shape

        if torch.is_autocast_enabled():
            decoder_outputs = decoder_outputs.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(decoder_outputs):
            self._graph_reinitialize(decoder_outputs, decoder_output_lengths)

        # set length to zero for elements outside the current batch
        self.state.decoder_output_lengths.fill_(0)
        # copy (projected) encoder output and lenghts
        self.state.decoder_outputs[:curr_batch_size, :curr_max_time, ...].copy_(decoder_outputs)
        self.state.decoder_output_lengths[:curr_batch_size].copy_(decoder_output_lengths.unsqueeze(-1))
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs._before_process_batch.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs._process_batch.replay()
            self.separate_graphs._after_process_batch.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_process_batch()
            while self.state.active_mask_any.item():
                self._process_batch()
            self._after_process_batch()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        return self.state.batched_hyps

    @classmethod
    def _create_process_batch_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

        extern "C" __global__
        void loop_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"loop_conditional", cls.CUDA_PROGRAM_NAME)

    def _graph_reinitialize(
        self,
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ):
        """
        Reinitializes the graph state for the Beam Search computation.
        This method sets up the internal state required for the decoding process, including initializing
        decoder outputs, decoder states, and optional n-gram language model states. It also handles CUDA
        graph compilation based on the specified mode.
        Args:
            encoder_output_projected (torch.Tensor): The projected encoder output tensor of shape
                (batch_size, max_time, encoder_dim).
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch.
        Raises:
            NotImplementedError: If an unsupported CUDA graph mode is specified.
        """

        batch_size, max_time, vocab_size = decoder_outputs.shape

        # Create (or grow) the persistent buffers that CUDA graphs will keep reusing across batches.
        self.state = BacthedBeamCTCState(
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            vocab_size=vocab_size,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
            blank_index=self._blank_index,
        )

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self._full_graph_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsBatchedBeamCTC()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._before_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._after_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._after_process_batch()

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        self.full_graph = torch.cuda.CUDAGraph()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._before_process_batch()
            capture_status, _, graph, _, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )

            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            loop_kernel = self._create_process_batch_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            loop_args = np.array(
                [loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            # loop while there are active utterances
            with with_conditional_node(loop_kernel, loop_args, loop_conditional_handle, device=self.state.device):
                self._process_batch()

            self._after_process_batch()

    def _before_process_batch(self):
        """
        Clears state and setups fusion models.
        """
        # step 1.1: reset state
        self.state.batched_hyps.clear_()
        self.state.curr_frame_idx.fill_(0)

        # maximum time step for each utterance
        torch.sub(self.state.decoder_output_lengths, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.decoder_output_lengths, 0, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _process_batch(self):
        """
        Performs a decoding step.
        """
        # Detect label repetitions and blanks so we can apply CTC constraints without re-materializing strings.
        repeated_mask = self.state.batched_hyps.last_label[:, :, None] == self.state.vocab[None, None, :]
        repeated_or_blank_mask = repeated_mask | self.state.vocab_blank_mask[None, None, :]

        # step 2.1: getting the log probs and updating with fusion scores
        log_probs = self.state.decoder_outputs.index_select(dim=1, index=self.state.curr_frame_idx)
        log_probs += self.state.batched_hyps.scores[:, :, None]

        # step 2.2: updating non-blank and non-repeating token scores with `beam_beta`
        log_probs = torch.where(repeated_or_blank_mask, log_probs, log_probs + self.beam_beta)

        # step 2.3: getting `beam_size` best candidates
        # Flatten beam/vocab so a single topk gives the best extensions per utterance.
        next_scores, next_candidates_indices = torch.topk(
            log_probs.view(self.state.batch_size, -1), k=self.beam_size, largest=True, sorted=True
        )
        next_indices = next_candidates_indices // self.state.vocab_size
        next_labels = next_candidates_indices % self.state.vocab_size

        # step 2.3: pruning candidates with threshold `beam_threshold`
        batch_next_scores = next_scores.view(self.state.batch_size, -1)
        max_next_score = batch_next_scores.max(dim=-1, keepdim=True).values
        # Drop beams that fall outside the pruning window; keeps search width manageable on long vocab tails.
        batch_next_scores.masked_fill_(batch_next_scores <= max_next_score - self.beam_threshold, INACTIVE_SCORE)
        next_scores.view(self.state.batch_size, self.beam_size, -1)

        # step 2.5: masking inactive hypotheses, updating + recombining batched beam hypoteses
        torch.where(self.state.active_mask, next_labels, self.state.NON_EXISTENT_LABEL, out=next_labels)
        self.state.batched_hyps.add_results_no_checks_(next_indices, next_labels, next_scores)
        self.state.batched_hyps.recombine_hyps_()

        # step 2.6: updating frame idx and active masks
        self.state.curr_frame_idx.add_(1)
        torch.greater_equal(self.state.last_timesteps, self.state.curr_frame_idx, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _after_process_batch(self):
        """Hook for any per-batch cleanup after CUDA decoding (currently unused)."""
        return None

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> BatchedBeamHyps:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            return self.batched_beam_search_cuda_graphs(decoder_outputs=x, decoder_output_lengths=out_len)

        return self.batched_beam_search_torch(decoder_outputs=x, decoder_output_lengths=out_len)
        
        
