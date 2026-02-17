"""
Default configuration for CTC beam search decoder.
Edit this file to change default hyperparameters.
"""

# =============================================================================
# PATHS
# =============================================================================
PATHS = {
    "tokens": "/home/ebrahim/data2/brain2text/lm/units_pytorch.txt",
    "lexicon": "/home/ebrahim/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt",
    "word_lm": "/home/ebrahim/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm",
    "transcripts_val": "/home/ebrahim/data2/brain2text/b2t_24/transcripts_val_cleaned.pkl",
    "lora_adapter_1b": "/home/ebrahim/brainaudio/finetune_llm/llama-3.2-1b-hf-finetuned-normalized",
    "lora_adapter_3b": "/home/ebrahim/brainaudio/finetune_llm/llama-3.2-3b-hf-finetuned-normalized",
    "results_dir": "../results/b2t_24",
    "results_test_dir": "../results/test_files/b2t_24",
}

# =============================================================================
# LLM SETTINGS
# =============================================================================
LLM = {
    "model": "meta-llama/Llama-3.2-1B",
    "llm_weight": 1.2,
    "ngram_rescore_weight": 0.0,
    "lm_rescore_interval": 10,
    "scoring_chunk_size": 256,
}

# =============================================================================
# ACOC / CTC SETTINGS
# =============================================================================

ACOUSTIC = {
    "temperature": 1.0,
    "acoustic_scale": 0.6,
}

# =============================================================================
# BEAM SEARCH SETTINGS
# =============================================================================
BEAM_SEARCH = {
    "beam_size": 900,
    "num_homophone_beams": 3,
    "beam_prune_threshold": 18.0,
    "homophone_prune_threshold": 4.0,
    "beam_beta": 1.5,
    "word_boundary_bonus": 1.0,
    "alpha_ngram": 0.8,
    "top_k": 10,
    "score_combination": "max",
}

# =============================================================================
# DEVICE SETTINGS
# =============================================================================
DEVICE = {
    "device": "cuda:0",
}
