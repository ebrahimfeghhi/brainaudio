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

# OOV filtering threshold (0.0 = strict, only in-vocab; 0.1 = allow up to 10% OOV words)
MAX_OOV_RATIO = 0.10

if MODE == "TEST":
    SUBSET_SIZE = 100_000      # Lines per dataset
    NGRAM_ORDER = 3            # Smaller n-gram for speed
    PRUNE_ARGS = "0 1"         # Prune singleton trigrams
    OUTPUT_NAME = "test_cased"
    USE_TEMP_DISK = False      
    
else:
    # PROD SETTINGS
    SUBSET_SIZE = None         
    NGRAM_ORDER = 4
    PRUNE_ARGS = "0 1 1"       # Prune singleton 3-grams and 4-grams
    OUTPUT_NAME = "final_cased_12B"
    USE_TEMP_DISK = True

# Whether to keep the cleaned corpus .txt file after training (useful for debugging/retrying)
KEEP_CORPUS = True

# PATHS
BIN_LMPLZ = "/home/ebrahim/kenlm/build/bin/lmplz"
BIN_BUILD_BINARY = "/home/ebrahim/kenlm/build/bin/build_binary"
TEMP_DIR = "./kenlm_tmp"
VOCAB_FILE = "/data2/brain2text/lm/words.txt"  # For OOV filtering

# URL for OpenSubtitles (Direct from OPUS)
OPENSUB_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_vocabulary(vocab_path):
    """Load vocabulary set from file for OOV checking."""
    print(f"   -> Loading vocabulary from {vocab_path}...")
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                vocab.add(word)
    print(f"      Loaded {len(vocab):,} words.")
    return vocab

# Characters to strip from words for OOV checking
# Keep apostrophes and hyphens since they're part of words like 'twas, don't, self-driving
PUNCT_TO_STRIP = string.punctuation.replace("'", "").replace("-", "")

def strip_punctuation(word):
    """Strip leading/trailing punctuation from a word."""
    return word.strip(PUNCT_TO_STRIP)

def split_into_sentences(text):
    """
    Split text into sentences using NLTK (handles 'Mr.', 'Dr.', 'U.S.A.' correctly).
    """
    # NLTK handles the splitting smartly
    sentences = nltk.sent_tokenize(text)
    
    cleaned_sentences = []
    for s in sentences:
        # Strip whitespace
        s = s.strip()
        if len(s) > 2:
            cleaned_sentences.append(s)
    
    return cleaned_sentences

def get_oov_ratio(text, vocab):
    """Get the ratio of OOV words in text. Strips punctuation before checking."""
    words = text.lower().split()
    if not words:
        return 0.0
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
    """Check if OOV ratio exceeds the allowed threshold."""
    return get_oov_ratio(text, vocab) > max_ratio

def get_kenlm_cmd(order, prune, input_file, output_arpa, use_temp=False):
    cmd = f"{BIN_LMPLZ} -o {order} --prune {prune}"
    if use_temp:
        cmd += f" -S 80% -T {TEMP_DIR}"
    cmd += f" < {input_file} > {output_arpa}"
    return cmd

def stream_opensubtitles_manual(url, file_handle, vocab, limit=None):
    """
    Streams OpenSubtitles directly from the OPUS server (GZipped),
    bypassing the broken Hugging Face script.
    Splits into sentences and filters out those containing OOV words.
    """
    print(f"   -> Downloading & Streaming OpenSubtitles from source...")
    
    count = 0
    skipped = 0
    # Stream the request so we don't load 4GB into RAM
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Use gzip to decompress the stream on the fly
        with gzip.open(r.raw, 'rt', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if len(line) > 2:
                    # Split line into individual sentences
                    for sentence in split_into_sentences(line):
                        if exceeds_oov_threshold(sentence, vocab):
                            skipped += 1
                        else:
                            file_handle.write(sentence + "\n")
                            count += 1
                    
                    if (count + skipped) % 500_000 == 0:
                        print(f"      Processed {(count + skipped) // 1_000}k sentences, kept {count // 1_000}k...", end='\r')
                
                if limit and count >= limit:
                    break
    print(f"\n      Finished OpenSubtitles: {count} sentences kept, {skipped} skipped (OOV).")

def stream_hf_dataset(dataset_name, config, split, file_handle, text_col, vocab, limit):
    """Safe wrapper for Hugging Face datasets. Splits into sentences and filters out those with OOV words."""
    print(f"   -> Streaming {dataset_name}...")
    try:
        # Try loading without trust_remote_code first (for parquet support)
        try:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        except:
            # Fallback for older datasets
            ds = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
            
    except Exception as e:
        print(f"      [ERROR] Skipped {dataset_name}. Reason: {e}")
        return

    count = 0
    skipped = 0
    for row in ds:
        text = row[text_col]
        if isinstance(text, str):
            text = text.replace('\n', ' ').strip()
            if len(text) > 2:
                # Split text into individual sentences
                for sentence in split_into_sentences(text):
                    if exceeds_oov_threshold(sentence, vocab):
                        skipped += 1
                    else:
                        file_handle.write(sentence + "\n")
                        count += 1
                if (count + skipped) % 500_000 == 0:
                    print(f"      Processed {(count + skipped) // 1_000}k sentences, kept {count // 1_000}k...", end='\r')
        if limit and count >= limit:
            break
    print(f"\n      Finished {dataset_name}: {count} sentences kept, {skipped} skipped (OOV).")

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
    
    # Load vocabulary for OOV filtering
    vocab = load_vocabulary(VOCAB_FILE)
    
    # 1. AGGREGATE DATA
    print(">>> Phase 1: Aggregating Data...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        
        # A. OpenSubtitles (Direct Download)
        stream_opensubtitles_manual(OPENSUB_URL, f, vocab, SUBSET_SIZE)

        # B. WikiText-103 (Facts)
        stream_hf_dataset("wikitext", "wikitext-103-raw-v1", "train", f, "text", vocab, SUBSET_SIZE)

        # C. OpenWebText (Prose/Internet)
        stream_hf_dataset("Skylion007/openwebtext", None, "train", f, "text", vocab, SUBSET_SIZE)

    # 2. TRAIN
    print("\n>>> Phase 2: Training N-gram Model...")
    cmd_train = get_kenlm_cmd(NGRAM_ORDER, PRUNE_ARGS, corpus_file, arpa_file, USE_TEMP_DISK)
    print(f"Running: {cmd_train}")
    
    ret = os.system(cmd_train)
    if ret != 0:
        print("ERROR: lmplz failed.")
        return

    # 3. BINARIZE
    print("\n>>> Phase 3: Binarizing...")
    cmd_bin = f"{BIN_BUILD_BINARY} {arpa_file} {binary_file}"
    os.system(cmd_bin)

    # 4. CLEANUP
    print("\n>>> Cleanup...")
    if KEEP_CORPUS:
        print(f"   Keeping corpus file: {corpus_file}")
    else:
        if os.path.exists(corpus_file): os.remove(corpus_file)
    if os.path.exists(arpa_file): os.remove(arpa_file)
    if USE_TEMP_DISK and os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)

    print("-" * 30)
    print(f"DONE! Model: {binary_file}")
    if KEEP_CORPUS:
        print(f"Corpus saved: {corpus_file}")

if __name__ == "__main__":
    main()