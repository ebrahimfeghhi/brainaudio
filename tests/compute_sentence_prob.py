import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import truecase

def compute_log_likelihood_manual(sentence, model_id="meta-llama/Llama-3.2-3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    # 1. Prepare inputs
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 2. Shift for Causal Prediction
    # Logits at index `i` predict the token at `i+1`
    shift_logits = logits[:, :-1, :]  # All tokens except last (predictions)
    shift_labels = input_ids[:, 1:]   # All tokens except first (targets)

    # 3. Log Softmax
    # Converts raw logits into log-probabilities for the entire vocabulary
    # dim=-1 applies it across the vocabulary dimension
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # 4. Gather the Log-Probs of the Actual Tokens
    # We need to pick the specific log-prob for the token that actually appeared.
    # gather expects indices to have the same dimensions as the source, so we unsqueeze.
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))

    # 5. Sum them up
    # This is the log likelihood of the entire sequence
    sentence_log_likelihood = target_log_probs.sum().item()
    
    print(f"\nProcessing sentence: '{sentence}'")
    running_sum = 0.0
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # We skip the first token (BOS) for printing because it has no prediction score
    for i, log_prob in enumerate(target_log_probs[0]):
        val = log_prob.item()
        running_sum += val
        # The token being predicted is at index i+1
        token_str = tokens[i+1]
        print(f"Token: {token_str:<15} | LogProb: {val:.4f} | Running Sum: {running_sum:.4f}")

    return running_sum


# --- Usage ---
if __name__ == "__main__":
    sentence = "helps them understand the."
    ll = compute_log_likelihood_manual(sentence)
    print(f"Log Likelihood: {ll:.4f}")