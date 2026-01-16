import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def compute_sentence_log_probs(model, tokenizer, sentences):
    """
    Computes the total log probability of a list of sentences using a Causal LM.
    Processing is done in a single batch (parallel).
    """
    
    # 1. Tokenize as a batch
    # We set padding_side='right' for generation usually, but for scoring, 
    # standard padding works as long as we use the attention mask correctly.
    tokenizer.pad_token = tokenizer.eos_token  # Llama usually has no pad token by default
    
    inputs = tokenizer(
        sentences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)

    # 2. Forward Pass (Compute Logits for all tokens in parallel)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. Shift Logits and Labels
    # The model predicts the NEXT token. So logits at position N predict token at N+1.
    # We slice off the last logit (no next token to predict) and the first label (no prediction for it).
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    shift_mask = inputs.attention_mask[:, 1:].contiguous()

    # 4. Compute Log Probabilities
    # We use CrossEntropyLoss to get the log prob of the *correct* token at each step.
    # reduction='none' gives us the loss per token.
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # View(-1) flattens the batch for the loss function
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )

    # Reshape back to [batch_size, seq_len]
    token_losses = token_losses.view(shift_labels.size())

    # 5. Mask and Sum
    # CrossEntropy returns positive Loss (-log_prob). We want Log Prob, so we negate it.
    # We also multiply by the mask to ignore padding tokens.
    token_log_probs = -token_losses * shift_mask
    
    # Sum across the sequence length to get total sentence log prob
    sentence_log_probs = token_log_probs.sum(dim=1)

    return sentence_log_probs

def main():
    model_id = "meta-llama/Llama-3.2-3B"  # Ensure you have access
    
    print(f"Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load in 16-bit precision to save memory/speed up
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure you ran 'huggingface-cli login' and have access to Llama 3.2.")
        return

    sentences = [
        "He is also a member of the real iris addendum",
        "He is also a member of the royal iris academy",
    ]

    print("\nComputing log probabilities...")
    scores = compute_sentence_log_probs(model, tokenizer, sentences)

    print("-" * 60)
    for sent, score in zip(sentences, scores):
        print(f"Sentence: {sent}")
        print(f"Log Prob: {score.item():.4f}")
        print("-" * 60)

    # Comparison Logic
    diff = scores[1] - scores[0]
    print(f"\nDifference: {diff.item():.4f}")
    if scores[1] > scores[0]:
        print("Result: The model prefers the Capitalized version.")
    else:
        print("Result: The model prefers the lowercase version.")

if __name__ == "__main__":
    main()