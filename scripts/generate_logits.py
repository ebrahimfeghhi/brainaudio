# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
from brainaudio.inference.load_model_generate_logits import load_transformer_model, generate_and_save_logits
import os
from typing import Optional, Dict
import csv
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Configuration ---
MODEL_NAME_TEMPLATES = [
    "neurips_b2t_25_causal_transformer_day_specific_softsign_seed_{seed}",
    "neurips_b2t_25_causal_transformer_day_specific_seed_{seed}",
]
SEEDS = [0]
MODEL_TYPE = "transformer"
local_model_folder = "b2t_25"  # folder the model is stored
modelWeightsFilesList = ["modelWeights_PER_25"]

DEVICE = "cuda:0"
PARTITION = 'val'

# Transformer Chunking Configs
# EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 20}, {"chunk_size": 1, "context_sec": 17.5}, {"chunk_size": 1, "context_sec": 15},
#        {"chunk_size": 1, "context_sec": 12.5}, {"chunk_size": 1, "context_sec": 10},
#        {"chunk_size": 1, "context_sec": 7.5}, {"chunk_size": 1, "context_sec": 5}] # Test Config
if local_model_folder == "b2t_24":
    EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 7.5}]
elif local_model_folder == "b2t_25":
    EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 20}]

MANIFEST_PATHS = [f"/home/ebrahim/data2/brain2text/{local_model_folder}/trial_level_data/manifest.json"]
BASE_SAVE_PATH = f'/home/ebrahim/data2/brain2text/{local_model_folder}/logits/'


def _format_eval_tag(cfg: Optional[Dict[str, Optional[int]]]) -> str:
    if cfg is None:
        return "default"
    chunk = cfg.get("chunk_size")
    context = cfg.get("context_sec")
    chunk_str = "full" if chunk is None else str(chunk)
    context_str = "full" if context is None else str(context)
    return f"chunk_{chunk_str}_context_{context_str}"

PER_RESULTS_DIR = "../results/per_results"
OUTPUT_LOG_PATH = f"output_logging/day_specific_log_{local_model_folder}.md"

def write_md_log(all_seed_pers):
    """Write a markdown summary of PER results to OUTPUT_LOG_PATH."""
    import datetime
    os.makedirs(os.path.dirname(OUTPUT_LOG_PATH), exist_ok=True)

    lines = [
        "# Day-Specific Transformer — Validation PER",
        f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        f"_Partition: val | Weights: modelWeights_PER_24 | Chunk: 1s / 7.5s context_",
        "",
    ]

    FAMILY_LABELS = {
        "softsign_transformer": "Day-Specific Softsign Transformer",
        "transformer": "Day-Specific Linear Transformer",
    }

    # Group by model family
    from collections import defaultdict
    by_family = defaultdict(list)
    for template, wfile, tag, seed, per in all_seed_pers:
        family = template.replace("neurips_b2t_24_chunked_unidirectional_day_specific_", "").replace("_seed_{seed}", "")
        by_family[family].append((seed, per))

    for family, entries in sorted(by_family.items()):
        label = FAMILY_LABELS.get(family, family)
        entries_sorted = sorted(entries, key=lambda x: x[0])
        pers = [p for _, p in entries_sorted]
        mean_per = sum(pers) / len(pers)

        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Seed | PER |")
        lines.append("|------|-----|")
        for seed, per in entries_sorted:
            lines.append(f"| {seed} | {per:.6f} |")
        lines.append(f"| **mean** | **{mean_per:.6f}** |")
        lines.append("")

    with open(OUTPUT_LOG_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown log saved to: {OUTPUT_LOG_PATH}")


def main():

    all_seed_pers = []  # (model_template, weights_file, tag, seed, per) for final summary

    for model_template in MODEL_NAME_TEMPLATES:
        for modelWeightsFiles in modelWeightsFilesList:
            # Derive a short suffix for the save directory (PER or PER_24)
            weights_suffix = modelWeightsFiles.replace("modelWeights_", "")

            per_by_config = {}

            for seed in SEEDS:
                MODEL_NAME = model_template.format(seed=seed)
                LOAD_MODEL_FOLDER = f"/home/ebrahim/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"

                print(f"\n{'='*60}")
                print(f"Template: {model_template}")
                print(f"Weights:  {modelWeightsFiles}")
                print(f"Seed {seed}: {MODEL_NAME}")
                print(f"{'='*60}")

                # Save to a directory that encodes both model name and weights file
                save_dir = f"{BASE_SAVE_PATH}{MODEL_NAME}_{weights_suffix}"
                os.makedirs(save_dir, exist_ok=True)
                if MODEL_TYPE == "transformer":
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

                        per = generate_and_save_logits(
                            model=model,
                            config=args,
                            partition=PARTITION,
                            device=DEVICE,
                            manifest_paths=MANIFEST_PATHS,
                            save_path=save_dir,
                            chunk_config=eval_cfg,
                        )

                        if per is not None:
                            per_by_config.setdefault(tag, {})[str(seed)] = round(per, 6)
                            all_seed_pers.append((model_template, modelWeightsFiles, tag, seed, round(per, 6)))

            # Save PER results per (model_template, weights_file)
            if per_by_config:
                os.makedirs(PER_RESULTS_DIR, exist_ok=True)
                model_base_name = model_template.replace("_seed_{seed}", "")
                per_out_path = os.path.join(PER_RESULTS_DIR, f"{model_base_name}_{weights_suffix}.json")
                with open(per_out_path, "w") as f:
                    json.dump(per_by_config, f, indent=2)
                print(f"\nPER results saved to: {per_out_path}")

    print("\n--- Logit generation complete. ---")
    print("\n=== PER Summary Across All Models ===")
    for template, wfile, tag, seed, per in all_seed_pers:
        short_template = template.replace("neurips_b2t_24_chunked_unidirectional_day_specific_", "")
        print(f"  model={short_template}  weights={wfile}  config={tag}  seed={seed}  PER={per}")
    per_values = [per for *_, per in all_seed_pers]
    print(f"\nPER list: {per_values}")

    write_md_log(all_seed_pers)

if __name__ == "__main__":
    main()
