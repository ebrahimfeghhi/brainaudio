import re
import tqdm
import nltk
from datasets import load_dataset
import os

# --- SETTINGS ---
TARGET_SIZE_MB = 20
OUTPUT_FILE = "truecase_training_corpus.txt"
LEXICON_PATH = "/data2/brain2text/lm/lexicon_phonemes.txt"
MAX_OOV_RATIO = 0  # Strict: Drop sentence if any word is not in lexicon

# DEBUG SETTINGS
PRINT_DROPPED = True        
MAX_DROPPED_PRINT = 20      

# Download the sentence splitter model
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_lexicon(path):
    """
    Loads the lexicon file into a set of uppercase words.
    """
    print(f"Loading lexicon from {path}...")
    valid_words = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    # Lexicon is usually UPPERCASE (e.g., CMU Dict)
                    word = parts[0].upper()
                    valid_words.add(word)
        print(f"Lexicon loaded: {len(valid_words)} words.")
        return valid_words
    except FileNotFoundError:
        print(f"ERROR: Lexicon not found at {path}")
        exit(1)

def check_sentence_quality(sentence, lexicon, max_oov_ratio):
    """
    Returns (is_clean, oov_ratio, oov_words_list)
    
    Checks 1: Are the words in the lexicon? (Validity check)
    Checks 2: Is the casing distribution 'normal'? (Heuristic check)
    """
    # Regex to find words (preserving case for now)
    # Matches words like "Don't", "Apple", "test"
    words_cased = re.findall(r"\b[a-zA-Z]+(?:'[a-z]+)?\b", sentence)
    
    if not words_cased:
        return False, 1.0, []
        
    # --- 1. LEXICON CHECK ---
    # We convert to UPPER just for the lookup, but we don't modify the original sentence
    oov_words = []
    for w in words_cased:
        if w.upper() not in lexicon:
            oov_words.append(w)
    
    total_words = len(words_cased)
    oov_ratio = len(oov_words) / total_words
    
    if oov_ratio > max_oov_ratio:
        return False, oov_ratio, oov_words

    # --- 2. CASING HEURISTICS (New for Truecasing) ---
    # We want to drop sentences that are ALL CAPS (like headers) 
    # or all lowercase (garbage), as they confuse the truecaser.
    
    upper_count = sum(1 for w in words_cased if w.isupper())
    # If > 80% of words are ALL CAPS, it's likely a header or shouting -> Drop it
    if upper_count / total_words > 0.8:
        return False, 0.0, ["<TOO_MANY_CAPS>"]

    return True, oov_ratio, oov_words

def clean_and_format(text):
    # REMOVED: text = text.upper() (We need case information!)
    
    # Allow a-z (lowercase), A-Z, spaces, and standard punctuation
    text = re.sub(r"[^a-zA-Z\s.,'?]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) < 5:
        return []

    # Sentence split works better on cased text
    sentences = nltk.sent_tokenize(text)
    return sentences

# --- MAIN EXECUTION ---
print(f"Streaming OpenWebText for Truecasing (Target: {TARGET_SIZE_MB}MB)...")

lexicon = load_lexicon(LEXICON_PATH)
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    total_bytes = 0
    bar = tqdm.tqdm(total=TARGET_SIZE_MB * 1024 * 1024, unit='B', unit_scale=True)
    
    dropped_sentences = 0
    kept_sentences = 0
    printed_dropped_count = 0
    
    print("\n--- DROPPED SENTENCE LOG ---")
    
    for sample in dataset:
        sentences = clean_and_format(sample['text'])
        
        for sent in sentences:
            if len(sent) < 10: continue # Slightly higher min length for context
            
            is_clean, ratio, oov_words = check_sentence_quality(sent, lexicon, MAX_OOV_RATIO)
            
            if is_clean:
                # Write the CASED sentence
                f.write(sent + "\n")
                
                line_bytes = len(sent.encode('utf-8')) + 1
                total_bytes += line_bytes
                bar.update(line_bytes)
                kept_sentences += 1
            else:
                dropped_sentences += 1
                
                # DEBUG PRINTING
                if PRINT_DROPPED and printed_dropped_count < MAX_DROPPED_PRINT:
                    print(f"\n[DROPPED] Ratio: {ratio:.2f} ({len(oov_words)} bad/OOV words)")
                    print(f"SENT: {sent}")
                    print(f"Reasons: {oov_words}")
                    printed_dropped_count += 1
                    
                    if printed_dropped_count == MAX_DROPPED_PRINT:
                        print("\n... (Limit reached, suppressing further dropped logs) ...\n")
        
        if total_bytes > TARGET_SIZE_MB * 1024 * 1024:
            break

print(f"\nDone! Saved TRUECASED training text to {OUTPUT_FILE}")
print(f"Stats: Kept {kept_sentences} | Dropped {dropped_sentences}")