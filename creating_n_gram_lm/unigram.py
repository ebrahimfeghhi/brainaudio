import os
import re
import sys
from collections import Counter
from tqdm import tqdm

# ================= CONFIGURATION =================
# Update these paths to match where your files actually are
ALLOWLIST_PATH = "./wordlist_697k.txt"
C4_PATH = "./lm_training_data/raw_c4.txt"
SUBTITLES_PATH = "./lm_training_data/raw_subtitles.txt"
OUTPUT_VOCAB_PATH = "./vocab_200k_cased.txt"

TOP_K = 200_000

# Explicitly keep punctuation as tokens (per the paper's methodology)
PUNCTUATION = {".", ",", "!", "?", "'", '"'}
# =================================================

def load_allowlist(path):
    print(f"Loading allowlist from {path}...")
    allowlist = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Store as lowercase for case-insensitive lookup
                allowlist.add(line.strip().lower())
    except FileNotFoundError:
        print(f"Error: Allowlist file not found at {path}")
        sys.exit(1)
        
    print(f"Allowlist loaded. {len(allowlist)} unique base words.")
    return allowlist

def tokenize(text):
    """
    Simple whitespace and punctuation tokenizer.
    Keeps casing intact.
    """
    # This regex separates words and punctuation, keeping them distinct
    return re.findall(r"\w+(?:'\w+)?|[.,!?;]", text)

def count_corpus(file_path, allowlist):
    print(f"Counting tokens in {file_path}...")
    counts = Counter()
    total_tokens = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing", unit="lines"):
                tokens = tokenize(line)
                
                for token in tokens:
                    # 1. Always keep Punctuation
                    if token in PUNCTUATION:
                        counts[token] += 1
                        total_tokens += 1
                        continue
                    
                    # 2. Check Allowlist (Case-Insensitive)
                    # We count the CASED version (e.g., "The") if "the" is in allowlist
                    if token.lower() in allowlist:
                        counts[token] += 1
                        total_tokens += 1
                        
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return Counter(), 0

    print(f"Finished {os.path.basename(file_path)}. Valid tokens: {total_tokens:,}")
    return counts, total_tokens

def main():
    # 1. Load the filter list
    valid_words = load_allowlist(ALLOWLIST_PATH)
    
    # 2. Process datasets
    # Note: If memory is tight, we could optimize this, but for 200k vocab
    # the Counters fit easily in RAM even if processing GBs of text.
    c4_counts, c4_total = count_corpus(C4_PATH, valid_words)
    sub_counts, sub_total = count_corpus(SUBTITLES_PATH, valid_words)
    
    print("\nCalculating Mixture Probabilities...")
    
    # Get all unique cased words found in either dataset
    all_unique_words = set(c4_counts.keys()) | set(sub_counts.keys())
    
    word_scores = {}
    
    # Linear Mixture Model: P(w) = 0.5 * P_c4(w) + 0.5 * P_sub(w)
    # This gives equal voting rights to the Conversational data (Subtitles)
    # vs General English (C4), preventing the larger vocab of C4 from dominating.
    for word in tqdm(all_unique_words, desc="Scoring"):
        prob_c4 = c4_counts[word] / c4_total if c4_total > 0 else 0
        prob_sub = sub_counts[word] / sub_total if sub_total > 0 else 0
        
        mixture_prob = 0.5 * prob_c4 + 0.5 * prob_sub
        word_scores[word] = mixture_prob

    # 3. Sort and Select Top K
    print(f"\nSorting and selecting top {TOP_K} words...")
    # Sort by score descending
    sorted_vocab = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
    top_vocab = sorted_vocab[:TOP_K]
    
    # 4. Save to file
    print(f"Saving vocabulary to {OUTPUT_VOCAB_PATH}...")
    with open(OUTPUT_VOCAB_PATH, 'w', encoding='utf-8') as f:
        for word, _ in top_vocab:
            f.write(word + "\n")
            
    # Preview
    print("\nTop 10 Words in Vocabulary:")
    for i in range(min(10, len(top_vocab))):
        print(f"{i+1}. {top_vocab[i][0]}")

    print("\nDone.")

if __name__ == "__main__":
    main()