import re
import tqdm
import nltk
from datasets import load_dataset
import os
import sys

# --- SETTINGS ---
# Adjust size as needed (e.g., 500MB for a solid model, 20MB for a quick test)
TARGET_SIZE_MB = 20  
OUTPUT_FILE = "truecase_training_corpus.txt"

# Your specific lexicon path
LEXICON_PATH = "/data2/brain2text/lm/lexicon_phonemes.txt"

# STRICTNESS: 0 = Drop sentence if even ONE word is missing from lexicon.
# Recommended for ASR to prevent "impossible" LM predictions.
MAX_OOV_RATIO = 0  

# DEBUG SETTINGS
PRINT_DROPPED = True        
MAX_DROPPED_PRINT = 10      

# Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
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
                    # Lexicon is standardly UPPERCASE in ASR (e.g. CMU Dict)
                    # We store it as such for case-insensitive lookup
                    word = parts[0].upper()
                    valid_words.add(word)
        print(f"Lexicon loaded: {len(valid_words)} words.")
        return valid_words
    except FileNotFoundError:
        print(f"ERROR: Lexicon not found at {path}")
        print("Please check the path or generate a lexicon first.")
        sys.exit(1)

def check_sentence_quality(sentence, lexicon, max_oov_ratio):
    """
    Returns (is_clean, oov_ratio, oov_words_list)
    """
    # Regex to find words while ignoring punctuation
    # Matches "Don't", "Apple", "test"
    words_cased = re.findall(r"\b[a-zA-Z]+(?:'[a-z]+)?\b", sentence)
    
    if not words_cased:
        return False, 1.0, []
        
    # --- 1. LEXICON CHECK ---
    oov_words = []
    for w in words_cased:
        # Check against lexicon in UPPERCASE, but don't change original text
        if w.upper() not in lexicon:
            oov_words.append(w)
    
    total_words = len(words_cased)
    oov_ratio = len(oov_words) / total_words
    
    if oov_ratio > max_oov_ratio:
        return False, oov_ratio, oov_words

    # --- 2. CASING HEURISTICS ---
    # Filter out ALL CAPS lines (headers) or garbage
    upper_count = sum(1 for w in words_cased if w.isupper())
    
    # If > 80% of words are ALL CAPS, it's likely a header/shouting -> Drop it
    if len(words_cased) > 1 and (upper_count / total_words > 0.8):
        return False, 0.0, ["<TOO_MANY_CAPS>"]

    return True, oov_ratio, oov_words

def clean_and_format(text):
    # Keep A-Z, a-z, spaces, and standard punctuation. 
    # CRITICAL: Do NOT .upper() or .lower() here. We need the case!
    text = re.sub(r"[^a-zA-Z\s.,'?]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) < 5:
        return []

    # Use NLTK to split into sentences (handles "Mr." vs "." well)
    sentences = nltk.sent_tokenize(text)
    return sentences

# --- MAIN EXECUTION ---
def main():
    print(f"Streaming OpenWebText for Truecasing (Target: {TARGET_SIZE_MB}MB)...")

    lexicon = load_lexicon(LEXICON_PATH)
    
    # Streaming ensures we don't need 50GB RAM
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        total_bytes = 0
        # Progress bar estimates based on bytes
        bar = tqdm.tqdm(total=TARGET_SIZE_MB * 1024 * 1024, unit='B', unit_scale=True, desc="Filtering")
        
        dropped_sentences = 0
        kept_sentences = 0
        printed_dropped_count = 0
        
        if PRINT_DROPPED:
            print("\n--- DROPPED SENTENCE SAMPLE ---")
        
        for sample in dataset:
            sentences = clean_and_format(sample['text'])
            
            for sent in sentences:
                if len(sent) < 15: continue # Skip very short snippets
                
                is_clean, ratio, oov_words = check_sentence_quality(sent, lexicon, MAX_OOV_RATIO)
                
                if is_clean:
                    # Write the Clean, CASED sentence
                    f.write(sent + "\n")
                    
                    line_bytes = len(sent.encode('utf-8')) + 1
                    total_bytes += line_bytes
                    bar.update(line_bytes)
                    kept_sentences += 1
                else:
                    dropped_sentences += 1
                    
                    # DEBUG LOGGING
                    if PRINT_DROPPED and printed_dropped_count < MAX_DROPPED_PRINT:
                        print(f"[DROP] Ratio: {ratio:.2f} | Reason: {oov_words[:3]}... | Sent: {sent[:60]}...")
                        printed_dropped_count += 1
                        if printed_dropped_count == MAX_DROPPED_PRINT:
                            print("... (Suppressing further dropped logs) ...")
            
            if total_bytes > TARGET_SIZE_MB * 1024 * 1024:
                break

    print(f"\n\nDone! Saved TRUECASED training text to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Stats: Kept {kept_sentences:,} | Dropped {dropped_sentences:,}")
    print("Next Step: Use this file to train your KenLM!")

if __name__ == "__main__":
    main()