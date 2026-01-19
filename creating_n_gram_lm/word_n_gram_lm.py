import os
import shutil
import gzip
import string
import requests
import nltk
from datasets import load_dataset

# Ensure NLTK sentence tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ==========================================
# CONFIGURATION SECTION
# ==========================================

# MODE: "TEST" (Quick verification) or "PROD" (Full 12B word training)
MODE = "PROD"  

# OOV filtering threshold (0.0 = strict; 0.1 = allow 10% OOV).
MAX_OOV_RATIO = 0.00

if MODE == "TEST":
    SUBSET_SIZE = 100_000      # Lines per dataset
    NGRAM_ORDER = 3            
    PRUNE_ARGS = "0 1"         
    OUTPUT_NAME = "test_cased"
    USE_TEMP_DISK = False      
    
else:
    # PROD SETTINGS
    SUBSET_SIZE = None
    NGRAM_ORDER = 4
    PRUNE_ARGS = "0 1 1 1"     # Prune singleton 2-grams, 3-grams, and 4-grams
    OUTPUT_NAME = "final_cased_12B"
    USE_TEMP_DISK = True

# Whether to keep the cleaned corpus .txt file after training
KEEP_CORPUS = True

# PATHS (Verified based on your previous logs)
BIN_LMPLZ = "/home/ebrahim/kenlm/build/bin/lmplz"
BIN_BUILD_BINARY = "/home/ebrahim/kenlm/build/bin/build_binary"
TEMP_DIR = "./kenlm_tmp"
VOCAB_FILE = "/data2/brain2text/lm/words.txt" 

# URL for OpenSubtitles (Direct from OPUS)
OPENSUB_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_vocabulary(vocab_path):
    """Load vocabulary set from file for OOV checking."""
    print(f"   -> Loading vocabulary from {vocab_path}...")
    vocab = set()
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    vocab.add(word)
        print(f"      Loaded {len(vocab):,} words.")
        return vocab
    except FileNotFoundError:
        print(f"[ERROR] Vocab file not found at {vocab_path}. OOV filter disabled.")
        return None

# Characters to strip from words for OOV checking
PUNCT_TO_STRIP = string.punctuation.replace("'", "").replace("-", "")

def strip_punctuation(word):
    return word.strip(PUNCT_TO_STRIP)

def clean_sentence_text(text):
    """
    CRITICAL FIX: Remove <s> and </s> tags. 
    These cause KenLM to crash with 'FormatLoadException'.
    """
    return text.replace("<s>", "").replace("</s>", "").strip()

def split_into_sentences(text):
    """Split text into sentences using NLTK."""
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = []
    for s in sentences:
        s = clean_sentence_text(s) # Sanitize here
        if len(s) > 2:
            cleaned_sentences.append(s)
    return cleaned_sentences

def get_oov_ratio(text, vocab):
    """Get the ratio of OOV words in text."""
    if vocab is None: return 0.0
    words = text.lower().split()
    if not words: return 0.0
    
    oov_count = 0
    total_count = 0
    for word in words:
        clean_word = strip_punctuation(word)
        if clean_word:
            total_count += 1
            if clean_word not in vocab:
                oov_count += 1
    return oov_count / total_count if total_count > 0 else 0.0

def exceeds_oov_threshold(text, vocab, max_ratio=MAX_OOV_RATIO):
    return get_oov_ratio(text, vocab) > max_ratio

def stream_opensubtitles_manual(url, file_handle, vocab, limit=None):
    """Streams OpenSubtitles directly from OPUS."""
    print(f"   -> Downloading & Streaming OpenSubtitles from source...")
    count = 0
    skipped = 0
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with gzip.open(r.raw, 'rt', encoding='utf-8') as f_in:
            for line in f_in:
                # 1. Clean Tags
                line = clean_sentence_text(line)
                if len(line) > 2:
                    # 2. Split & Filter
                    for sentence in split_into_sentences(line):
                        if exceeds_oov_threshold(sentence, vocab):
                            skipped += 1
                        else:
                            file_handle.write(sentence + "\n")
                            count += 1
                    
                    if (count + skipped) % 500_000 == 0:
                        print(f"      Processed {(count + skipped) // 1_000}k sentences...", end='\r')
                
                if limit and count >= limit:
                    break
    print(f"\n      Finished OpenSubtitles: {count} kept, {skipped} skipped.")

def stream_hf_dataset(dataset_name, config, split, file_handle, text_col, vocab, limit):
    """Safe wrapper for Hugging Face datasets."""
    print(f"   -> Streaming {dataset_name}...")
    try:
        try:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        except:
            ds = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"      [ERROR] Skipped {dataset_name}. Reason: {e}")
        return

    count = 0
    skipped = 0
    for row in ds:
        text = row[text_col]
        if isinstance(text, str):
            # 1. Clean Newlines & Tags
            text = clean_sentence_text(text.replace('\n', ' '))
            if len(text) > 2:
                # 2. Split & Filter
                for sentence in split_into_sentences(text):
                    if exceeds_oov_threshold(sentence, vocab):
                        skipped += 1
                    else:
                        file_handle.write(sentence + "\n")
                        count += 1
                if (count + skipped) % 500_000 == 0:
                    print(f"      Processed {(count + skipped) // 1_000}k sentences...", end='\r')
        if limit and count >= limit:
            break
    print(f"\n      Finished {dataset_name}: {count} kept, {skipped} skipped.")

# ==========================================
# MAIN
# ==========================================

def main():
    if USE_TEMP_DISK:
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    corpus_file = f"{OUTPUT_NAME}.txt"
    arpa_file = f"{OUTPUT_NAME}.arpa"
    binary_file = f"{OUTPUT_NAME}.binary"

    print(f"== STARTING {MODE} RUN ==")
    
    # Load vocabulary
    vocab = load_vocabulary(VOCAB_FILE)
    
    # 1. AGGREGATE DATA
    print(">>> Phase 1: Aggregating Data...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        # A. OpenSubtitles
        stream_opensubtitles_manual(OPENSUB_URL, f, vocab, SUBSET_SIZE)
        # B. WikiText-103
        stream_hf_dataset("wikitext", "wikitext-103-raw-v1", "train", f, "text", vocab, SUBSET_SIZE)
        # C. OpenWebText
        stream_hf_dataset("Skylion007/openwebtext", None, "train", f, "text", vocab, SUBSET_SIZE)

    # 2. TRAIN
    print("\n>>> Phase 2: Training N-gram Model...")
    # Note: No 'sed' pipe needed here because Phase 1 already cleaned the tags!
    cmd_train = (
        f"{BIN_LMPLZ} -o {NGRAM_ORDER} --prune {PRUNE_ARGS} "
        f"-S 80% -T {TEMP_DIR} "
        f"< {corpus_file} > {arpa_file}"
    )
    print(f"Running: {cmd_train}")
    
    ret = os.system(cmd_train)
    if ret != 0:
        print("[CRITICAL ERROR] lmplz failed. Check disk space or memory.")
        return

    # 3. BINARIZE
    print("\n>>> Phase 3: Binarizing...")
    cmd_bin = f"{BIN_BUILD_BINARY} {arpa_file} {binary_file}"
    os.system(cmd_bin)

    # 4. CLEANUP
    print("\n>>> Cleanup...")
    # if os.path.exists(arpa_file): os.remove(arpa_file)  # Keep for entropy pruning later
    if USE_TEMP_DISK and os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    
    if not KEEP_CORPUS and os.path.exists(corpus_file):
        os.remove(corpus_file)

    print("-" * 30)
    print(f"DONE! Model: {binary_file}")

if __name__ == "__main__":
    main()