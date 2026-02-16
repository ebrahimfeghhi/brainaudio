# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
from brainaudio.inference.load_model_generate_logits import load_transformer_model, generate_and_save_logits
import os
from typing import Optional, Dict
import csv

# --- Configuration ---
MODEL_NAME = "neurips_b2t_24_chunked_transformer_seed_0"
local_model_folder = "b2t_24" # folder the model is stored
modelWeightsFiles = "modelWeights_PER_24" # "modelWeights_PER_24"


LOAD_MODEL_FOLDER = f"/home/ebrahim/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"  
DEVICE = "cuda:0"   
PARTITION = 'val'

# Optionally evaluate multiple chunk configs per run. Use None to keep the
# checkpoint's stored eval config. Add dicts like {"chunk_size": 5, "context_chunks": 50}.
# EVAL_CONFIGS = [
#     {"chunk_size": 1, "context_sec": 5},
#     {"chunk_size": 1, "context_sec": 10},
#     {"chunk_size": 1, "context_sec": 20},
#     {"chunk_size": 1, "context_sec": None},

#     {"chunk_size": 5, "context_sec": 5},
#     {"chunk_size": 5, "context_sec": 10},
#     {"chunk_size": 5, "context_sec": 20},
#     {"chunk_size": 5, "context_sec": None},

#     {"chunk_size": 10, "context_sec": 5},
#     {"chunk_size": 10, "context_sec": 10},
#     {"chunk_size": 10, "context_sec": 20},
#     {"chunk_size": 10, "context_sec": None},

#     {"chunk_size": 20, "context_sec": 5},
#     {"chunk_size": 20, "context_sec": 10},
#     {"chunk_size": 20, "context_sec": 20},
#     {"chunk_size": 20, "context_sec": None},

#     {"chunk_size": None, "context_sec": None},
# ] # Val Configs

EVAL_CONFIGS = [{"chunk_size": 5, "context_sec": None},] # Test Config

if modelWeightsFiles == "modelWeights_PER_25":
    
    MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_25/trial_level_data/manifest.json"]
    SAVE_PATHS = {0:'/home/ebrahim/data2/brain2text/b2t_25/logits/'}
    PARTICIPANT_IDS = [0]
    
    
if modelWeightsFiles == "modelWeights_PER_24":
    
    MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_24/trial_level_data/manifest.json"]
    SAVE_PATHS = {0:'/home/ebrahim/data2/brain2text/b2t_24/logits/'}
    PARTICIPANT_IDS = [0]
    
    
if modelWeightsFiles == "modelWeights_PER":
    
    MANIFEST_PATHS = ["/home/ebrahim/data2/brain2text/b2t_25/trial_level_data/manifest.json", 
                      "/home/ebrahim/data2/brain2text/b2t_24/trial_level_data/manifest.json"]
    SAVE_PATHS = { 0:'/home/ebrahim/data2/brain2text/b2t_25/logits/test/',
                    1:'/home/ebrahim/data2/brain2text/b2t_24/logits/'}
    PARTICIPANT_IDS = [0, 1]
    
def _format_eval_tag(cfg: Optional[Dict[str, Optional[int]]]) -> str:
    if cfg is None:
        return "default"
    chunk = cfg.get("chunk_size")
    context = cfg.get("context_chunks")
    chunk_str = "full" if chunk is None else str(chunk)
    context_str = "full" if context is None else str(context)
    return f"chunk_{chunk_str}_context_{context_str}"

def main():

    resolved_save_paths = {}
    for idx, value in SAVE_PATHS.items():
        resolved_path = f"{value}{MODEL_NAME}"
        os.makedirs(resolved_path, exist_ok=True)
        resolved_save_paths[idx] = resolved_path

    results = []

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
                chunk_size = eval_cfg.get("chunk_size")
                context_sec = eval_cfg.get("context_sec")
                results.append({
                    "model_name": MODEL_NAME,
                    "chunk_size": "full" if chunk_size is None else chunk_size,
                    "context_sec": "full" if context_sec is None else context_sec,
                    "participant_id": pid,
                    "per": round(per, 6),
                })
            per_out_path = os.path.join(resolved_save_paths[pid], f"per_results_{modelWeightsFiles[-2:]}.csv")
            with open(per_out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    print("--- Logit generation complete. ---")

if __name__ == "__main__":
    main()