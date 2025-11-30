"""
Test script for neural LM fusion with CTC beam search.

This script demonstrates how to use the neural LM fusion feature and
measures its impact on decoding quality (WER).

Usage:
import argparse
    python test_neural_lm_fusion.py [num_trials]
    
Example:
    python test_neural_lm_fusion.py 10  # Test on 10 utterances
"""

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer, 
    LexiconConstraint,
    VectorizedLexiconConstraint,
    HuggingFaceLMFusion,
)
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

# Configuration
LANGUAGE_MODEL_PATH = "/data2/brain2text/lm/"
TOKENS_TXT = f"{LANGUAGE_MODEL_PATH}units_pytorch.txt"
WORDS_TXT = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
LOGITS_PATH = "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test neural LM fusion on a subset of trials")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of utterances to decode (default: 1)")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size to use for decoding (default: 1)")
    parser.add_argument("--start-trial", type=int, default=1, help="Starting trial index from validation set (default: 1)")
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run decoding without neural LM for comparison",
    )
    return parser.parse_args()

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
    blank_index: int = 0,
):
    """Run beam search with optional LM fusion."""
    decoder = BatchedBeamCTCComputer(
        blank_index=blank_index,
        beam_size=beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=allow_cuda_graphs,
    )
    
    result = decoder(logits_batched, logits_lengths)
    return result


def decode_beam_outputs(
    result,
    lexicon,
    token_to_symbol,
    phoneme_to_word,
    max_beams: int = 1,
    verbose: bool = False,
):
    """Decode up to max_beams hypotheses per utterance."""

    all_decoded = []
    num_beams_available = result.transcript_wb.shape[1]
    beams_to_decode = min(max_beams, num_beams_available)

    for b in range(result.transcript_wb.shape[0]):
        beam_entries = []
        for k in range(beams_to_decode):
            seq = result.transcript_wb[b, k]
            seq_filtered = seq[seq >= 0]
            score = result.scores[b, k].item()

            if verbose:
                print(f"\n   DEBUG Utterance {b} Beam {k}:")
                print(f"     Raw sequence length: {len(seq_filtered)}")
                print(f"     First 10 tokens: {seq_filtered[:10].tolist()}")

            if score > float('-inf') and len(seq_filtered) > 0:
                ids_no_blanks = apply_ctc_rules(seq_filtered)

                if verbose:
                    print(f"     After CTC rules: {len(ids_no_blanks)} tokens")
                    print(f"     Tokens after CTC: {ids_no_blanks[:10]}")

                phoneme_text = " ".join(token_to_symbol.get(t, f"UNK{t}") for t in ids_no_blanks)
                if lexicon is not None and phoneme_to_word is not None:
                    word_alts = lexicon.decode_sequence_to_words(
                        token_ids=ids_no_blanks,
                        token_to_symbol=token_to_symbol,
                        lexicon_word_map=phoneme_to_word,
                        return_alternatives=True,
                    )
                    primary_words = [alts[0] if alts else word for word, alts in word_alts]
                    decoded_text = " ".join(primary_words)
                else:
                    decoded_text = phoneme_text

                beam_entries.append(
                    {
                        "beam_index": k,
                        "score": score,
                        "text": decoded_text,
                        "tokens": ids_no_blanks,
                        "phonemes": phoneme_text,
                    }
                )

                if verbose:
                    print(f"     Decoded: '{decoded_text}'")
            else:
                beam_entries.append(
                    {
                        "beam_index": k,
                        "score": score,
                        "text": "<NO_DECODE>",
                        "tokens": [],
                        "phonemes": "",
                    }
                )

        all_decoded.append(beam_entries)

    return all_decoded


def score_sentence_with_lm(text: str, tokenizer, model, device: torch.device) -> float:
    """Compute log-probability of a sentence under the HuggingFace LM."""
    if not text.strip():
        return float('-inf')

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2:
        return 0.0

    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits.float()
    log_probs = F.log_softmax(logits, dim=-1)

    score = 0.0
    for idx in range(1, len(tokens)):
        token_id = tokens[idx]
        score += log_probs[0, idx - 1, token_id].item()
    return score


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
    args = parse_args()
    num_trials = max(1, args.num_trials)
    requested_beam = max(1, args.beam_size)
    beam_size = 1
    if requested_beam != 1:
        print(f"Forcing beam size to 1 (requested {requested_beam}) for this experiment.")
    start_trial = max(0, args.start_trial)
    run_baseline = args.run_baseline
    trial_indices = list(range(start_trial, start_trial + num_trials))
    print("=" * 80)
    print("Neural LM Fusion Test")
    print("=" * 80)
    print(f"\nTesting on {num_trials} utterances starting at trial {start_trial} (beam size {beam_size})")
    
    # Load data
    print("\n1. Loading data...")
    model_logits = np.load(LOGITS_PATH)
    ground_truth_transcript = pd.read_pickle('/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl')
    gt_subset = ground_truth_transcript[start_trial:start_trial + num_trials]
    
    # Load first N trials
    logits_list = []
    lengths_list = []
    
    for i in trial_indices:
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
    
    print(f"   Loaded {num_trials} utterances")
    print(f"   Logits shape: {logits_batched.shape}")
    
    # Load lexicon
    print("\n2. Loading lexicon...")
    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=TOKENS_TXT,
        lexicon_file=WORDS_TXT,
        device=torch.device('cuda:0'),
    )
    token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)
    phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)
    print(f"   Lexicon loaded: {len(phoneme_to_word)} words")
    decoded_baseline = []
    wers_baseline = []
    avg_wer_baseline = None
    step_idx = 3

    if run_baseline:
        print("\n3. Running baseline beam search (no lexicon constraint)...")
        result_baseline = run_beam_search(
            logits_batched, 
            logits_lengths, 
            lexicon=None,
            lm_fusion=None,
            beam_size=beam_size,
            allow_cuda_graphs=False,
            blank_index=0,
        )
        baseline_beams = decode_beam_outputs(
            result_baseline,
            lexicon=None,
            token_to_symbol=token_to_symbol,
            phoneme_to_word=None,
            max_beams=min(beam_size, 10),
            verbose=False,
        )
        decoded_baseline = [beams[0]["text"] if beams else "<NO_DECODE>" for beams in baseline_beams]

        print("\n   Baseline beam outputs:")
        for trial_id, beams, gt in zip(trial_indices, baseline_beams, gt_subset):
            print(f"\n   Trial {trial_id} (ground truth: {gt})")
            for rank, beam_info in enumerate(beams, 1):
                print(f"     Beam {rank:02d} | CTC={beam_info['score']:.2f}")
                print(f"        Phonemes: {beam_info['phonemes']}")
                print(f"        Text:     {beam_info['text']}")

        wers_baseline = [compute_wer(decoded, gt) for decoded, gt in zip(decoded_baseline, gt_subset)]
        avg_wer_baseline = np.mean(wers_baseline)
        print(f"\n   Baseline average WER (phoneme text vs transcript): {avg_wer_baseline:.2%}")
        
        print("\n" + "=" * 80)
        print("Baseline decoding (no lexicon) complete.")
        print("=" * 80)
        step_idx = 4
    else:
        print("\n3. Skipping baseline beam search (pass --run-baseline to enable).")
    
    # Test 2: Real LM fusion (Gemma-3-270M) - optional, requires transformers
    try:
        print(f"\n{step_idx}. Loading Gemma-3-270M (tiny, fast LM) for neural LM fusion...")
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
        lm_fusion.log_homophone_scores = True
        
        print("   Gemma-3-270M loaded successfully")
        print(f"\n{step_idx + 1}. Running beam search with Gemma-3-270M LM fusion...")
        
        result_lm = run_beam_search(
            logits_batched, 
            logits_lengths, 
            lexicon,
            lm_fusion=lm_fusion,
            beam_size=beam_size,
            allow_cuda_graphs=False,
            blank_index=lexicon.blank_index,
        )
        max_display_beams = min(10, beam_size)
        decoded_lm_beams = decode_beam_outputs(
            result_lm,
            lexicon,
            token_to_symbol,
            phoneme_to_word,
            max_beams=max_display_beams,
            verbose=False,
        )
        decoded_lm = [beams[0]["text"] if beams else "<NO_DECODE>" for beams in decoded_lm_beams]
        
        lm_device = getattr(lm_fusion, "device", torch.device('cuda:0'))
        print("\n   Gemma-3-270M LM beam outputs:")
        for trial_id, beam_list in zip(trial_indices, decoded_lm_beams):
            print(f"\n   Utterance {trial_id}:")
            for rank, beam_info in enumerate(beam_list, 1):
                text = beam_info["text"]
                ctc_score = beam_info["score"]
                lm_logprob = score_sentence_with_lm(text, tokenizer, model, lm_device) if text != "<NO_DECODE>" else float('-inf')
                phoneme_seq = beam_info.get("phonemes", "")
                print(
                    f"     Beam {rank:02d}: LM logprob={lm_logprob:.2f}, "
                    f"CTC score={ctc_score:.2f}\n"
                    f"        Phonemes: {phoneme_seq}\n"
                    f"        Text:     {text}"
                )
        
        # Compute average WER
        wers_lm = [compute_wer(decoded, gt) for decoded, gt in zip(decoded_lm, gt_subset)]
        avg_wer_lm = np.mean(wers_lm)
        print(f"\n   Average Gemma-3-270M WER: {avg_wer_lm:.2%}")
        
        print(f"\n{step_idx + 2}. Summary:")
        print(f"   {'Method':<20} {'Avg WER':<15} {'Improvement':<15}")
        print(f"   {'-'*50}")
        if run_baseline and avg_wer_baseline is not None:
            print(f"   {'Baseline':<20} {avg_wer_baseline:<15.2%} {'-':<15}")
            improvement = (avg_wer_baseline - avg_wer_lm) / max(avg_wer_baseline, 1e-10)
            print(f"   {'Gemma-3-270M':<20} {avg_wer_lm:<15.2%} {improvement:<15.2%}")
        else:
            print(f"   {'Gemma-3-270M':<20} {avg_wer_lm:<15.2%} {'(baseline disabled)':<15}")
        
        if run_baseline and wers_baseline:
            print(f"\n{step_idx + 3}. Per-utterance comparison:")
            print(f"   {'Utt':<5} {'Baseline WER':<15} {'Gemma WER':<15} {'Improvement':<15}")
            print(f"   {'-'*50}")
            for trial_id, wer_base, wer_lm in zip(trial_indices, wers_baseline, wers_lm):
                improvement = (wer_base - wer_lm) / max(wer_base, 1e-10)
                print(f"   {trial_id:<5} {wer_base:<15.2%} {wer_lm:<15.2%} {improvement:<15.2%}")
        
    except ImportError:
        print(f"\n{step_idx}. Skipping Gemma-3-270M test (transformers not installed)")
        print("   Install with: pip install transformers")
    except Exception as e:
        print(f"\n{step_idx}. Error loading Gemma-3-270M: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
