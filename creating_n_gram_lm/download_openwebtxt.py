import re
import tqdm
import nltk
from datasets import load_dataset
import os

# --- SETTINGS ---
TARGET_SIZE_MB = 20000  
OUTPUT_FILE = "lm_corpus_cleaned_large.txt"
LEXICON_PATH = "/data2/brain2text/lm/lexicon_phonemes.txt"
MAX_OOV_RATIO = 0

# DEBUG SETTINGS
PRINT_DROPPED = True        # Set to False to silence output
MAX_DROPPED_PRINT = 20      # Only print the first 20 failures to avoid flooding console

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
                    word = parts[0].upper()
                    valid_words.add(word)
        print(f"Lexicon loaded: {len(valid_words)} words.")
        return valid_words
    except FileNotFoundError:
        print(f"ERROR: Lexicon not found at {path}")
        exit(1)

def check_sentence_oov(sentence, lexicon, max_oov_ratio):
    """
    Returns (is_clean, oov_ratio, oov_words_list)
    """
    words = re.findall(r"\b[A-Z]+(?:'[A-Z]+)?\b", sentence)
    
    if not words:
        return False, 1.0, []
        
    oov_words = []
    for w in words:
        if w not in lexicon:
            oov_words.append(w)
    
    total_words = len(words)
    oov_ratio = len(oov_words) / total_words
    
    is_clean = oov_ratio <= max_oov_ratio
    return is_clean, oov_ratio, oov_words

def clean_and_format(text):
    text = text.upper()
    text = re.sub(r"[^A-Z\s.,'?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) < 5:
        return []

    sentences = nltk.sent_tokenize(text)
    return sentences

# --- MAIN EXECUTION ---
print(f"Streaming OpenWebText (Target: {TARGET_SIZE_MB}MB)...")

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
            if len(sent) < 5: continue
            
            is_clean, ratio, oov_words = check_sentence_oov(sent, lexicon, MAX_OOV_RATIO)
            
            if is_clean:
                f.write(sent + "\n")
                line_bytes = len(sent.encode('utf-8')) + 1
                total_bytes += line_bytes
                bar.update(line_bytes)
                kept_sentences += 1
            else:
                dropped_sentences += 1
                
                # DEBUG PRINTING
                if PRINT_DROPPED and printed_dropped_count < MAX_DROPPED_PRINT:
                    print(f"\n[DROPPED] Ratio: {ratio:.2f} ({len(oov_words)} OOV words)")
                    print(f"SENT: {sent}")
                    print(f"OOVs: {oov_words}")
                    printed_dropped_count += 1
                    
                    if printed_dropped_count == MAX_DROPPED_PRINT:
                        print("\n... (Limit reached, suppressing further dropped logs) ...\n")
        
        if total_bytes > TARGET_SIZE_MB * 1024 * 1024:
            break

print(f"\nDone! Saved cleaned text to {OUTPUT_FILE}")
print(f"Stats: Kept {kept_sentences} | Dropped {dropped_sentences}")