"""Minimal stubs for NeMo dependencies."""

import logging as _logging

# Use Python's standard logging
logging = _logging


class PrettyStrEnum(str):
    """Simple enum replacement."""
    pass


class WithOptionalCudaGraphs:
    """Stub for CUDA graphs support - not needed for basic functionality."""
    pass


class ConfidenceMethodMixin:
    """Stub for confidence scoring - not needed for basic beam search."""
    pass


class NGramGPULanguageModel:
    """Stub for n-gram LM - not using this feature."""
    pass


class GPUBoostingTreeModel:
    """Stub for context biasing - not using this feature."""
    pass


class LanguageModelFusion:
    """Stub for LM fusion - not using this feature."""
    pass


class Hypothesis:
    """Hypothesis data class for beam search results."""
    def __init__(self, score: float, y_sequence: list, text: str = "", dec_state=None, timestep: list = None):
        self.score = score
        self.y_sequence = y_sequence if y_sequence is not None else []
        self.text = text
        self.dec_state = dec_state
        self.timestep = timestep if timestep is not None else []


class NBestHypotheses:
    """Container for n-best hypotheses."""
    def __init__(self, n_best_hypotheses: list = None):
        self.n_best_hypotheses = n_best_hypotheses if n_best_hypotheses is not None else []


# CUDA Python utilities stubs
def skip_cuda_python_context_manager():
    """Stub for CUDA context manager."""
    from contextlib import nullcontext
    return nullcontext()


def run_only_if_cuda_python_available(func):
    """Decorator stub - just return the function as-is."""
    return func
