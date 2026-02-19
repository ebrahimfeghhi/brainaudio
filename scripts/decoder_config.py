"""
Default configuration for CTC beam search decoder.
Edit this file to change default hyperparameters.
"""

base_path = "/home/ebrahim/"

# =============================================================================
# DATASET SELECTOR  ("b2t_24" or "b2t_25")
# =============================================================================
DATASET = "b2t_25"

# =============================================================================
# PER-DATASET CONFIGS
# =============================================================================
_B2T_24 = {
    "transcripts_val": f"{base_path}/data2/brain2text/b2t_24/transcripts_val_cleaned.pkl",
    "results_dir": "../results/transformer_24",
    "results_test_dir": "../results/test_files/transformer_24",
    "beam_size": 1000,
    "beam_prune_threshold": 22.0,
    "alpha_ngram": 0.8,
    "acoustic_scale": 0.6,
    "lm_rescore_interval": 10,
}

_B2T_25 = {
    "transcripts_val": f"{base_path}/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl",
    "results_dir": "../results/transformer_25",
    "results_test_dir": "../results/test_files/transformer_25",
    "beam_size": 900,
    "beam_prune_threshold": 18.0,
    "alpha_ngram": 1.0,
    "acoustic_scale": 0.4,
    "lm_rescore_interval": 15,
}

_DS = {"b2t_24": _B2T_24, "b2t_25": _B2T_25}[DATASET]

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
    "lora_adapter_270m": "/home/ebrahim/brainaudio/finetune_llm/gemma-3-270m-hf-finetuned-normalized",
    "lora_adapter_360m": "/home/ebrahim/brainaudio/finetune_llm/smollm-360m-hf-finetuned-normalized",
    "lora_adapter_2b": "/home/ebrahim/brainaudio/finetune_llm/granite-3.3-2b-hf-finetuned-normalized",
    "lora_adapter_8b": "/home/ebrahim/brainaudio/finetune_llm/",
    "results_dir": "../results/b2t_24",
    "results_test_dir": "../results/test_files/b2t_24",
}

# =============================================================================
# LLM SETTINGS
# =============================================================================
LLM = {
    "model": "meta-llama/Llama-3.2-1B", # google/gemma-3-270m, HuggingFaceTB/SmolLM-360M, ibm-granite/granite-3.3-2b-base
    "llm_weight": 1.2,
    "ngram_rescore_weight": 0.0,
    "lm_rescore_interval": _DS["lm_rescore_interval"],
    "scoring_chunk_size": 256,
}

# =============================================================================
# ACOUSTIC / CTC SETTINGS
# =============================================================================
ACOUSTIC = {
    "temperature": 1.0,
    "acoustic_scale": _DS["acoustic_scale"],
}

# =============================================================================
# BEAM SEARCH SETTINGS
# =============================================================================
BEAM_SEARCH = {
    "beam_size": _DS["beam_size"],
    "num_homophone_beams": 3,
    "beam_prune_threshold": _DS["beam_prune_threshold"],
    "homophone_prune_threshold": 4.0,
    "beam_beta": 1.5,
    "word_boundary_bonus": 1.0,
    "alpha_ngram": _DS["alpha_ngram"],
    "top_k": 20,
    "score_combination": "max",
}

# =============================================================================
# DEVICE SETTINGS
# =============================================================================
DEVICE = {
    "device": "cuda:0",
}
