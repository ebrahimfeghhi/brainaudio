# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
from brainaudio.inference.load_model_generate_logits import load_transformer_model, generate_and_save_logits
import os
from typing import Optional, Dict
import csv
import json

# --- Configuration ---
MODEL_NAME_TEMPLATE = "neurips_b2t_25_causal_transformer_v4_prob_1_seed_{seed}"
SEEDS = [0]

local_model_folder = "b2t_25" # folder the model is stored
modelWeightsFiles = "modelWeights_PER_25" # "modelWeights_PER_24"

DEVICE = "cuda:0"
PARTITION = 'test'

EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 20}, {"chunk_size": 1, "context_sec": 17.5}, {"chunk_size": 1, "context_sec": 15}, 
        {"chunk_size": 1, "context_sec": 12.5}, {"chunk_size": 1, "context_sec": 10}, 
        {"chunk_size": 1, "context_sec": 7.5}, {"chunk_size": 1, "context_sec": 5}] # Test Config
#EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 20}]

if modelWeightsFiles == "modelWeights_PER_25":

    MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_25/trial_level_data/manifest.json"]
    SAVE_PATHS = {0:'/home/ebrahim/data2/brain2text/b2t_25/logits/'}
    PARTICIPANT_IDS = [0]


if modelWeightsFiles == "modelWeights_PER_24":

    if "gru" in MODEL_NAME_TEMPLATE:
        
        print("LOADING NON LOGGED DATA")
        MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_24/trial_level_data/manifest.json"]
        SAVE_PATHS = {0:'/home/ebrahim/data2/brain2text/b2t_24/logits/'}
        PARTICIPANT_IDS = [0]

    else:

        print("LOADING LOGGED DATA")
        MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_24/trial_level_data_log/manifest.json"]
        SAVE_PATHS = {0:'/home/ebrahim/data2/brain2text/b2t_24/logits/'}
        PARTICIPANT_IDS = [0]

    



def _format_eval_tag(cfg: Optional[Dict[str, Optional[int]]]) -> str:
    if cfg is None:
        return "default"
    chunk = cfg.get("chunk_size")
    context = cfg.get("context_sec")
    chunk_str = "full" if chunk is None else str(chunk)
    context_str = "full" if context is None else str(context)
    return f"chunk_{chunk_str}_context_{context_str}"

PER_RESULTS_DIR = "/home/ebrahim/brainaudio/results/per_results"

def main():

    per_by_config = {}  # {eval_tag: {seed: per}}
    all_seed_pers = []  # (tag, seed, pid, per) for final summary

    for seed in SEEDS:
        MODEL_NAME = MODEL_NAME_TEMPLATE.format(seed=seed)
        LOAD_MODEL_FOLDER = f"/home/ebrahim/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"

        print(f"\n{'='*60}")
        print(f"Running seed {seed}: {MODEL_NAME}")
        print(f"{'='*60}")

        resolved_save_paths = {}
        for idx, value in SAVE_PATHS.items():
            resolved_path = f"{value}{MODEL_NAME}"
            os.makedirs(resolved_path, exist_ok=True)
            resolved_save_paths[idx] = resolved_path

        eval_configs = EVAL_CONFIGS or [None]
        for eval_cfg in eval_configs:
            tag = _format_eval_tag(eval_cfg)
            print(f"=== Running eval config: {tag} ===")

            model, args = load_transformer_model(
                LOAD_MODEL_FOLDER,
                DEVICE,
                modelWeightsFile=modelWeightsFiles,
                eval_chunk_config=eval_cfg,
            )

            per_dict = generate_and_save_logits(
                model=model,
                config=args,
                partition=PARTITION,
                device=DEVICE,
                manifest_paths=MANIFEST_PATHS,
                save_paths=resolved_save_paths,
                participant_ids=PARTICIPANT_IDS,
                chunk_config=eval_cfg,
            )

            if per_dict is not None:
                for pid, per in per_dict.items():
                    per_by_config.setdefault(tag, {})[str(seed)] = round(per, 6)
                    all_seed_pers.append((tag, seed, pid, round(per, 6)))

    # Save PER results: one JSON per model, nested by eval config then seed
    if per_by_config:
        os.makedirs(PER_RESULTS_DIR, exist_ok=True)
        model_base_name = MODEL_NAME_TEMPLATE.replace("_seed_{seed}", "")
        per_out_path = os.path.join(PER_RESULTS_DIR, f"{model_base_name}.json")
        with open(per_out_path, "w") as f:
            json.dump(per_by_config, f, indent=2)
        print(f"\nPER results saved to: {per_out_path}")

    print("\n--- Logit generation complete. ---")
    print("\n=== PER Summary Across Seeds ===")
    for tag, seed, pid, per in all_seed_pers:
        print(f"  config={tag}  seed={seed}  participant={pid}  PER={per}")
    per_values = [per for _, _, _, per in all_seed_pers]
    print(f"\nPER list: {per_values}")

if __name__ == "__main__":
    main()
