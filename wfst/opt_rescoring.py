from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import numpy as np
from llm_utils import cer_with_gpt2_decoder, gpt2_lm_decode
import pandas as pd
from brainaudio.inference.eval_metrics import _cer_and_wer
from brainaudio.inference.eval_metrics import clean_string
import torch

# ---- Config ----
dataset = "b2t_24"
val = True

model_paths = [
    "time_masked_transformer/nbest_wfst_seed_1_neurips.pkl",
]
save_names = [
    "time_masked_transformer_seed_1_neurips",
]
assert len(model_paths) == len(save_names), "model_paths and save_names must be the same length"

print(f"RUNNING USING {dataset} settings, val={val}")

model_name = "facebook/opt-6.7b"

# ---- Load LLM ----
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

if dataset == "b2t_25":
    print("Loading model in 16-bit with automatic device placement...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
elif dataset == "b2t_24":
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

# Sanity check
inputs = llm_tokenizer("The future of AI is", return_tensors="pt").to(llm.device)
outputs = llm.generate(**inputs, max_new_tokens=50)
print(llm_tokenizer.decode(outputs[0], skip_special_tokens=True))

# ---- Dataset-specific config ----
if dataset == 'b2t_25':
    acoustic_scale = 0.325
    llm_weight = 0.55
    wfst_outputs_path = "/home/ebrahim/data2/brain2text/b2t_25/wfst_outputs/"
    saved_dir = "/home/ebrahim/data2/brain2text/b2t_25/opt_outputs/"
    ground_truth = pd.read_pickle("/home/ebrahim/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")
else:
    acoustic_scale = 0.5
    llm_weight = 0.5
    wfst_outputs_path = "/home/ebrahim/data2/brain2text/b2t_24/wfst_outputs/"
    saved_dir = "/home/ebrahim/data2/brain2text/b2t_24/opt_outputs/"
    ground_truth = pd.read_pickle("/home/ebrahim/data2/brain2text/b2t_24/transcripts_val_cleaned.pkl")

# ---- Rescoring loop ----
for mn, save_name in zip(model_paths, save_names):

    nbest_path = f"{wfst_outputs_path}{mn}"

    with open(nbest_path, mode='rb') as f:
        nbest = pickle.load(f)

    best_hyp_all = []

    for idx, nbest_trial in enumerate(nbest):
        if idx % 100 == 0:
            print(f"Index: {idx}")

        best_hyp = gpt2_lm_decode(
            llm,
            llm_tokenizer,
            nbest_trial,
            acoustic_scale,
            lengthPenlaty=0,
            alpha=llm_weight,
            returnConfidence=False
        )

        best_hyp_all.append(best_hyp.strip())

    if dataset == "b2t_25":
        best_hyp_all_cleaned = [clean_string(hyp) for hyp in best_hyp_all]
        df = pd.DataFrame({
            'id': range(len(best_hyp_all_cleaned)),
            'text': best_hyp_all_cleaned
        })
        df.to_csv(saved_dir + f"{save_name}_llm_outs.csv", index=False)

    else:
        with open(saved_dir + f"{save_name}_llm_outs.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(best_hyp_all) + "\n")

    if val:
        metrics = _cer_and_wer(best_hyp_all, ground_truth)
        print(f"Model: {save_name}")
        print(f"CER: {metrics[0]:.4f}, WER: {metrics[1]:.4f}")

breakpoint()