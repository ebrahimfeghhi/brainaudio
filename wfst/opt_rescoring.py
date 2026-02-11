from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import numpy as np
from llm_utils import cer_with_gpt2_decoder, gpt2_lm_decode
import pandas as pd
from brainaudio.inference.eval_metrics import _cer_and_wer
from brainaudio.inference.eval_metrics import clean_string
import torch

dataset = "b2t_25"

model_name = "facebook/opt-6.7b"

# Load tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

if dataset == "b2t_25":

    print("Loading model in 16-bit with automatic device placement...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
elif dataset == "b2t_24":
        
        # Load model in 8-bit with automatic device placement
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

# Example: Generate from a prompt
inputs = llm_tokenizer("The future of AI is", return_tensors="pt").to(llm.device)
outputs = llm.generate(**inputs, max_new_tokens=50)
print(llm_tokenizer.decode(outputs[0], skip_special_tokens=True))

if dataset == 'b2t_25':
    acoustic_scale = 0.325
    ground_truth = pd.read_pickle("/home/ebrahim/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")
    llm_weight = 0.55
else:
    acoustic_scale = 0.5
    llm_weight = 0.5
    
wer = []

for i in range(0,1,1):

    nbest_path = f"/home/ebrahim/data2/brain2text/b2t_24/wfst_outputs/time_masked_transformer/nbest_wfst_rescore_True_acoustic_scale_0.5_test_seed_{i}.pkl"
    model_outputs_path = None

    with open(nbest_path, mode = 'rb') as f:
        nbest = pickle.load(f)

    breakpoint()

    if model_outputs_path is not None:
        model_outputs = np.load(model_outputs_path, allow_pickle=True) 
        
        for i in range(len(model_outputs['transcriptions'])):
            new_trans = [ord(c) for c in model_outputs['transcriptions'][i]] + [0]
            model_outputs['transcriptions'][i] = np.array(new_trans)
            
        # Rescore nbest outputs with LLM
        llm_out = cer_with_gpt2_decoder(
            llm,
            llm_tokenizer,
            nbest[:],
            acoustic_scale,
            model_outputs,
            outputType="speech_sil",
            returnCI=True,
            lengthPenalty=0,
            alpha=llm_weight,
        )
        
    else:
        
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
            
            best_hyp_all.append(best_hyp)

    best_hyp_all_cleaned = [clean_string(hyp) for hyp in best_hyp_all]

    # wer.append(metrics[1])#
    df = pd.DataFrame({                                                                                                                                                                                                 
        'id': range(len(best_hyp_all_cleaned)),
        'text': best_hyp_all_cleaned
    })

    df.to_csv(f'/home/ebrahim/brainaudio/results/b2t_24_test_results/tm_best_hyp_all_cleaned_output_test_b2t_24_seed_{i}.csv', index=False)

#np.save("/home/ebrahim/brainaudio/wfst/wer_values_5gram_opt", wer)