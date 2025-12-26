"""
Beam search decoder with lexicon constraint.
Ported from NVIDIA NeMo with full CUDA graphs support.
"""

from .lexicon_constraint import LexiconConstraint, VectorizedLexiconConstraint
from .ctc_batched_beam_decoding import BatchedBeamCTCComputer
from .batched_beam_decoding_utils import BatchedBeamHyps
from .beam_helpers import (
    materialize_beam_transcript,
    format_beam_phonemes,
    strip_ctc,
    decode_beam_texts,
    decode_best_beams,
    collapse_ctc_sequence,
    decode_sequence_to_text,
    pick_device,
    load_log_probs,
    apply_ctc_rules,
    load_token_to_phoneme_mapping,
    load_phoneme_to_word_mapping,
    compute_wer,
    log_lm_watchlist_scores,
)
from .rnnt_utils import Hypothesis, NBestHypotheses
from .neural_lm_fusion import NeuralLanguageModelFusion, HuggingFaceLMFusion

__all__ = [
    'LexiconConstraint',
    'VectorizedLexiconConstraint',
    'BatchedBeamCTCComputer',
    'materialize_beam_transcript',
    'format_beam_phonemes',
    'strip_ctc',
    'decode_beam_texts',
    'decode_best_beams',
    'collapse_ctc_sequence',
    'decode_sequence_to_text',
    'pick_device',
    'load_log_probs',
    'BatchedBeamHyps',
    'Hypothesis',
    'NBestHypotheses',
    'NeuralLanguageModelFusion',
    'HuggingFaceLMFusion',
    'apply_ctc_rules',
    'load_token_to_phoneme_mapping',
    'load_phoneme_to_word_mapping',
    'compute_wer',
    'log_lm_watchlist_scores',
]
