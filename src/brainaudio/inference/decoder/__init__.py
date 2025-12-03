"""
Beam search decoder with lexicon constraint.
Ported from NVIDIA NeMo with full CUDA graphs support.
"""

from .lexicon_constraint import LexiconConstraint, VectorizedLexiconConstraint
from .ctc_batched_beam_decoding import BatchedBeamCTCComputer, materialize_beam_transcript
from .batched_beam_decoding_utils import BatchedBeamHyps
from .rnnt_utils import Hypothesis, NBestHypotheses
from .neural_lm_fusion import NeuralLanguageModelFusion, HuggingFaceLMFusion, DummyLMFusion
from .decode_utils import apply_ctc_rules, load_token_to_phoneme_mapping, load_phoneme_to_word_mapping, compute_wer

__all__ = [
    'LexiconConstraint',
    'VectorizedLexiconConstraint',
    'BatchedBeamCTCComputer', 
    'materialize_beam_transcript',
    'BatchedBeamHyps',
    'Hypothesis',
    'NBestHypotheses',
    'NeuralLanguageModelFusion',
    'HuggingFaceLMFusion',
    'DummyLMFusion',
    'apply_ctc_rules',
    'load_token_to_phoneme_mapping',
    'load_phoneme_to_word_mapping',
    'compute_wer'
]
