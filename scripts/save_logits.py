# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
from brainaudio.inference.load_model_generate_logits import load_transformer_model, generate_and_save_logits
import os
from typing import Optional, Dict

# --- Configuration ---
MODEL_NAME = "tm_transformer"
local_model_folder = "b2t_25"
modelWeightsFiles = ["modelWeights_WER_25"]


LOAD_MODEL_FOLDER = f"/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"  
DEVICE = "cuda:1"   
PARTITION = 'val'

# Optionally evaluate multiple chunk configs per run. Use None to keep the
# checkpoint's stored eval config. Add dicts like {"chunk_size": 5, "context_chunks": 50}.
EVAL_CONFIGS = [
    {"chunk_size": 5, "context_chunks": 30},
    {"chunk_size": 5, "context_chunks": 25},
    {"chunk_size": 5, "context_chunks": 20},
]


if local_model_folder == "b2t_25":
    
    MANIFEST_PATHS = ["/data2/brain2text/b2t_25/trial_level_data/manifest.json"]
    SAVE_PATHS = {0:'/data2/brain2text/b2t_25/logits/'}
    PARTICIPANT_IDS = [0]
    
    
if local_model_folder == "b2t_24":
    
    MANIFEST_PATHS = ["/data2/brain2text/b2t_24/trial_level_data/manifest.json"]
    SAVE_PATHS = {0:'/data2/brain2text/b2t_24/logits/'}
    PARTICIPANT_IDS = [0]
    
    
if local_model_folder == "b2t_combined":
    
    MANIFEST_PATHS = ["/data2/brain2text/b2t_25/trial_level_data/manifest.json", 
                      "/data2/brain2text/b2t_24/trial_level_data/manifest.json"]
    SAVE_PATHS = { 0:'/data2/brain2text/b2t_25/logits/',
                    1:'/data2/brain2text/b2t_24/logits/'}
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

    eval_configs = EVAL_CONFIGS or [None]
    for eval_cfg in eval_configs:
        tag = _format_eval_tag(eval_cfg)
        print(f"=== Running eval config: {tag} ===")

        model, args = load_transformer_model(
            LOAD_MODEL_FOLDER,
            DEVICE,
            modelWeightsFile=modelWeightsFiles[0],
            eval_chunk_config=eval_cfg,
        )

        generate_and_save_logits(
            model=model,
            config=args,
            partition=PARTITION,
            device=DEVICE,
            manifest_paths=MANIFEST_PATHS,
            save_paths=resolved_save_paths,
            participant_ids=PARTICIPANT_IDS,
            chunk_config=eval_cfg,
        )
    print("--- Logit generation complete. ---")

if __name__ == "__main__":
    main()