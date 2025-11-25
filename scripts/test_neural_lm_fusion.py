"""
Test script for neural LM fusion with CTC beam search.

This script demonstrates how to use the neural LM fusion feature and
measures its impact on decoding quality (WER).

Usage:
    python test_neural_lm_fusion.py [num_trials]
    
Example:
    python test_neural_lm_fusion.py 10  # Test on 10 utterances
"""

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer, 
    LexiconConstraint,
    HuggingFaceLMFusion,
    DummyLMFusion,
)
import numpy as np
import torch
import pandas as pd
import sys

# Configuration
LANGUAGE_MODEL_PATH = "/data2/brain2text/lm/"
TOKENS_TXT = f"{LANGUAGE_MODEL_PATH}units_pytorch.txt"
WORDS_TXT = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
LOGITS_PATH = "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"

# Number of trials to run (default 2, can be overridden by command line)
NUM_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

def apply_ctc_rules(ids):
    """Apply CTC rules: remove blanks (0) and merge consecutive repeats."""
    if hasattr(ids, 'cpu'):
        ids = ids.cpu().numpy()
    
    clean_ids = []
    prev_id = None
    
    for id_val in ids:
        if id_val == 0:  # Skip blank
            prev_id = None
            continue
        if id_val == prev_id:  # Skip repeats
            continue
        clean_ids.append(int(id_val))
        prev_id = id_val
    
    return clean_ids


def load_token_to_phoneme_mapping(tokens_file):
    """Load token ID -> phoneme symbol mapping."""
    token_to_symbol = {}
    with open(tokens_file, 'r') as f:
        for idx, line in enumerate(f):
            token_to_symbol[idx] = line.strip()
    return token_to_symbol


def load_phoneme_to_word_mapping(lexicon_file):
    """Build phoneme sequence -> word mapping from lexicon."""
    phoneme_to_word = {}
    with open(lexicon_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phonemes = tuple(p for p in parts[1:] if p != '|')
            phoneme_to_word[phonemes] = word
    return phoneme_to_word


def run_beam_search(
    logits_batched, 
    logits_lengths, 
    lexicon,
    lm_fusion=None,
    beam_size=10,
    allow_cuda_graphs=False,
):
    """Run beam search with optional LM fusion."""
    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=allow_cuda_graphs,
    )
    
    result = decoder(logits_batched, logits_lengths)
    return result


def decode_results(result, lexicon, token_to_symbol, phoneme_to_word):
    """Decode beam search results to text."""
    decoded_texts = []
    
    for b in range(result.transcript_wb.shape[0]):
        seq = result.transcript_wb[b, 0]
        seq_filtered = seq[seq >= 0]
        score = result.scores[b, 0].item()
        
        print(f"\n   DEBUG Utterance {b}:")
        print(f"     Raw sequence length: {len(seq_filtered)}")
        print(f"     First 10 tokens: {seq_filtered[:10].tolist()}")
        
        if score > float('-inf') and len(seq_filtered) > 0:
            # Apply CTC rules and decode
            ids_no_blanks = apply_ctc_rules(seq_filtered)
            print(f"     After CTC rules: {len(ids_no_blanks)} tokens")
            print(f"     Tokens after CTC: {ids_no_blanks[:10]}")
            
            # Convert to symbols
            symbols = [token_to_symbol.get(t, f"UNK{t}") for t in ids_no_blanks]
            print(f"     Symbols: {symbols[:20]}")
            
            # Try to decode with lexicon
            word_alts = lexicon.decode_sequence_to_words(
                ids_no_blanks, 
                token_to_symbol, 
                phoneme_to_word,
                return_alternatives=False
            )
            print(f"     Decoded: '{word_alts}'")
            decoded_texts.append(word_alts)
        else:
            decoded_texts.append("<NO_DECODE>")
    
    return decoded_texts


def compute_wer(hypothesis, reference):
    """Simple WER computation."""
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    
    # Simple Levenshtein distance
    d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]
    
    for i in range(len(hyp_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(hyp_words)][len(ref_words)] / max(len(ref_words), 1)


def main():
    print("=" * 80)
    print("Neural LM Fusion Test")
    print("=" * 80)
    print(f"\nTesting on {NUM_TRIALS} utterances")
    
    # Load data
    print("\n1. Loading data...")
    model_logits = np.load(LOGITS_PATH)
    ground_truth_transcript = pd.read_pickle('/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl')
    
    # Load first N trials
    logits_list = []
    lengths_list = []
    
    for i in range(NUM_TRIALS):
        logits_i = model_logits[f"arr_{i}"]
        logits_list.append(logits_i)
        lengths_list.append(logits_i.shape[0])
    
    # Pad to same time dimension
    max_time = max(lengths_list)
    padded_logits = []
    
    for logits_i in logits_list:
        if logits_i.shape[0] < max_time:
            pad_width = ((0, max_time - logits_i.shape[0]), (0, 0))
            logits_i = np.pad(logits_i, pad_width, mode='constant', constant_values=0)
        padded_logits.append(logits_i)
    
    # Batch and convert to torch
    logits_batched = torch.from_numpy(np.stack(padded_logits, axis=0)).to('cuda:0')
    logits_lengths = torch.from_numpy(np.array(lengths_list)).to('cuda:0')
    
    print(f"   Loaded {NUM_TRIALS} utterances")
    print(f"   Logits shape: {logits_batched.shape}")
    
    # Load lexicon
    print("\n2. Loading lexicon...")
    lexicon = LexiconConstraint.from_file_paths(
        tokens_file=TOKENS_TXT,
        lexicon_file=WORDS_TXT,
        device=torch.device('cuda:0'),
    )
    token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)
    phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)
    print(f"   Lexicon loaded: {len(phoneme_to_word)} words")
    
    # Test 1: Baseline (no LM fusion)
    print("\n3. Running baseline beam search (no LM fusion)...")
    result_baseline = run_beam_search(
        logits_batched, 
        logits_lengths, 
        lexicon,
        lm_fusion=None,
        allow_cuda_graphs=False,
    )
    decoded_baseline = decode_results(result_baseline, lexicon, token_to_symbol, phoneme_to_word)
    
    print("\n   Baseline results (detailed):")
    for i, (decoded, gt) in enumerate(zip(decoded_baseline, ground_truth_transcript[:NUM_TRIALS])):
        wer = compute_wer(decoded, gt)
        print(f"\n   Utterance {i}:")
        print(f"     Ground truth: {gt}")
        print(f"     Decoded:      {decoded}")
        print(f"     Score:        {result_baseline.scores[i, 0].item():.2f}")
        print(f"     Seq length:   {(result_baseline.transcript_wb[i, 0] >= 0).sum().item()}")
        print(f"     WER:          {wer:.2%}")
    
    # Compute average WER
    wers_baseline = [compute_wer(decoded, gt) for decoded, gt in zip(decoded_baseline, ground_truth_transcript[:NUM_TRIALS])]
    avg_wer_baseline = np.mean(wers_baseline)
    print(f"\n   Average Baseline WER: {avg_wer_baseline:.2%}")
    print(f"   Min WER: {np.min(wers_baseline):.2%}, Max WER: {np.max(wers_baseline):.2%}")
    
    print("\n" + "=" * 80)
    print("Baseline test complete!")
    print("=" * 80)
    return  # Exit early for debugging
    
    # Test 2: Dummy LM fusion (should be same as baseline)
    print("\n4. Running with dummy LM fusion (should match baseline)...")
    dummy_lm = DummyLMFusion(weight=0.3)
    result_dummy = run_beam_search(
        logits_batched, 
        logits_lengths, 
        lexicon,
        lm_fusion=dummy_lm,
        allow_cuda_graphs=False,
    )
    decoded_dummy = decode_results(result_dummy, lexicon, token_to_symbol, phoneme_to_word)
    
    wers_dummy = [compute_wer(decoded, gt) for decoded, gt in zip(decoded_dummy, ground_truth_transcript[:NUM_TRIALS])]
    avg_wer_dummy = np.mean(wers_dummy)
    print(f"   Average Dummy LM WER: {avg_wer_dummy:.2%} (should match baseline)")
    
    # Test 3: Real LM fusion (Gemma-3-270M) - optional, requires transformers
    try:
        print("\n5. Loading Gemma-3-270M (tiny, fast LM) for neural LM fusion...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        
        # Check for HF token
        hf_token = os.environ.get('HF_TOKEN', None)
        if hf_token:
            print("   Found HF_TOKEN in environment")
        
        # Use Gemma-3-270M - very small and fast
        model_name = "google/gemma-3-270m"
        print(f"   Loading {model_name}...")
        print("   Note: If you get 'gated repo' error, run: huggingface-cli login")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for speed
            device_map="cuda:0",
            token=hf_token,  # Pass token if available
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        
        lm_fusion = HuggingFaceLMFusion(
            model=model,
            tokenizer=tokenizer,
            weight=0.3,
            homophone_aggregation='max',
            device=torch.device('cuda:0'),
        )
        
        print("   Gemma-3-270M loaded successfully")
        print("\n6. Running beam search with Gemma-3-270M LM fusion...")
        
        result_lm = run_beam_search(
            logits_batched, 
            logits_lengths, 
            lexicon,
            lm_fusion=lm_fusion,
            allow_cuda_graphs=False,
        )
        decoded_lm = decode_results(result_lm, lexicon, token_to_symbol, phoneme_to_word)
        
        print("\n   Gemma-3-270M LM results:")
        for i, (decoded, gt) in enumerate(zip(decoded_lm, ground_truth_transcript[:NUM_TRIALS])):
            wer = compute_wer(decoded, gt)
            print(f"   Utterance {i}: WER={wer:.2%}")
        
        # Compute average WER
        wers_lm = [compute_wer(decoded, gt) for decoded, gt in zip(decoded_lm, ground_truth_transcript[:NUM_TRIALS])]
        avg_wer_lm = np.mean(wers_lm)
        print(f"\n   Average Gemma-3-270M WER: {avg_wer_lm:.2%}")
        
        print("\n7. Summary:")
        print(f"   {'Method':<20} {'Avg WER':<15} {'Improvement':<15}")
        print(f"   {'-'*50}")
        print(f"   {'Baseline':<20} {avg_wer_baseline:<15.2%} {'-':<15}")
        improvement = (avg_wer_baseline - avg_wer_lm) / max(avg_wer_baseline, 1e-10)
        print(f"   {'Gemma-3-270M':<20} {avg_wer_lm:<15.2%} {improvement:<15.2%}")
        
        print("\n8. Per-utterance comparison:")
        print(f"   {'Utt':<5} {'Baseline WER':<15} {'Gemma WER':<15} {'Improvement':<15}")
        print(f"   {'-'*50}")
        for i in range(NUM_TRIALS):
            wer_base = wers_baseline[i]
            wer_lm = wers_lm[i]
            improvement = (wer_base - wer_lm) / max(wer_base, 1e-10)
            print(f"   {i:<5} {wer_base:<15.2%} {wer_lm:<15.2%} {improvement:<15.2%}")
        
    except ImportError:
        print("\n5. Skipping Gemma-3-270M test (transformers not installed)")
        print("   Install with: pip install transformers")
    except Exception as e:
        print(f"\n5. Error loading Gemma-3-270M: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
