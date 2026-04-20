# File: generate_logits.py
# Purpose: Run model inference and save logits for downstream decoding.
from brainaudio.inference.load_model_generate_logits import load_transformer_model, load_gru_model, generate_and_save_logits
import os
from typing import Optional, Dict
import json

# ---- Edit these fields before running ----
MODEL_NAME_TEMPLATES = [
    "neurips_b2t_25_causal_transformer_seed_{seed}",
]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
local_model_folder = "b2t_25"   # "b2t_24" or "b2t_25"
modelWeightsFilesList = ["modelWeights_PER_25"]
PARTITION = "val"               # "val" or "test"
DEVICE = "cuda:0"
# ------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.split(":")[-1]

MODEL_TYPE = "transformer"  # "transformer" or "gru"

# Chunk size and context window for Transformer eval (not used for GRU).
# B2T '24 best: chunk_size=1, context_sec=7.5
# B2T '25 best: chunk_size=1, context_sec=20
if local_model_folder == "b2t_24":
    EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 7.5}]
elif local_model_folder == "b2t_25":
    EVAL_CONFIGS = [{"chunk_size": 1, "context_sec": 20}]


if modelWeightsFiles == "modelWeights_PER_25":

    MANIFEST_PATHS = [f"{BASE_DIR}/data2/brain2text/b2t_25/trial_level_data/manifest.json"]
    SAVE_PATH = f"{BASE_DIR}/data2/brain2text/b2t_25/logits/"


if modelWeightsFiles == "modelWeights_PER_24":

    if "gru" in MODEL_NAME_TEMPLATE:
        
        print("LOADING NON LOGGED DATA")
        MANIFEST_PATHS = [f"{BASE_DIR}/data2/brain2text/b2t_24/data/trial_level_data/manifest.json"]
        SAVE_PATH = f"{BASE_DIR}/data2/brain2text/b2t_24/logits/"

    else:

        print("LOADING LOGGED DATA")
        MANIFEST_PATHS = [f"{BASE_DIR}/data2/brain2text/b2t_24/data/trial_level_data_log/manifest.json"]
        SAVE_PATH = f"{BASE_DIR}/data2/brain2text/b2t_24/logits/"



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
    for template, _, _, seed, per in all_seed_pers:
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

    per_by_config = {}  # {eval_tag: {seed: per}}
    all_seed_pers = []  # (tag, seed, pid, per) for final summary

    for seed in SEEDS:
        MODEL_NAME = MODEL_NAME_TEMPLATE.format(seed=seed)
        LOAD_MODEL_FOLDER = f"{BASE_DIR}/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"

        print(f"\n{'='*60}")
        print(f"Running seed {seed}: {MODEL_NAME}")
        print(f"{'='*60}")

        resolved_save_path = f"{SAVE_PATH}{MODEL_NAME}"
        os.makedirs(resolved_save_path, exist_ok=True)


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

                elif MODEL_TYPE == "gru":
                    tag = "default"
                    model, args = load_gru_model(
                        LOAD_MODEL_FOLDER,
                        DEVICE,
                        modelWeightsFile=modelWeightsFiles,
                    )

                    per = generate_and_save_logits(
                        model=model,
                        config=args,
                        partition=PARTITION,
                        device=DEVICE,
                        manifest_paths=MANIFEST_PATHS,
                        save_path=save_dir,
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
    print("\n=== PER Summary Across Seeds ===")
    for tag, seed, per in all_seed_pers:
        print(f"  config={tag}  seed={seed}  PER={per}")
    per_values = [per for _, _, per in all_seed_pers]
    print(f"\nPER list: {per_values}")

if __name__ == "__main__":
    main()
