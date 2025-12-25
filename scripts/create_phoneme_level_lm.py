import os
import sys
import gzip
import shutil
import urllib.request
import multiprocessing as mp
import subprocess
import re
from tqdm import tqdm
import nltk
from nltk.corpus import cmudict
from g2p_en import G2p

# --- CONFIGURATION ---
# 1. DEFINE YOUR EXACT VOCABULARY
# Matches your request: standard phonemes + SIL
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]
# Add SIL to valid set
VALID_PHONES = set(PHONE_DEF + ['SIL'])

CORPUS_URL = "http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
DATA_DIR = "./phoneme_lm_data"
RAW_TEXT_FILE = os.path.join(DATA_DIR, "librispeech_raw.txt")
# We rename this to indicate it includes SIL tokens
PHONEME_FILE = os.path.join(DATA_DIR, "librispeech_phonemes_with_sil.txt")
ARPA_FILE = "librispeech_phoneme_sil_6gram.arpa"
BINARY_FILE = "librispeech_phoneme_sil_6gram.binary"

# KenLM Paths
KENLM_LMPLZ = "lmplz" 
KENLM_BUILD_BINARY = "build_binary"

# --- WORKER FUNCTIONS ---
g_dict = None
g_g2p = None

def init_worker():
    global g_dict, g_g2p
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    try:
        g_dict = cmudict.dict()
        g_g2p = G2p()
    except LookupError:
        nltk.download('cmudict', quiet=True)
        g_dict = cmudict.dict()
        g_g2p = G2p()
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)

def clean_phoneme(p):
    """
    Replicates user logic:
    p = re.sub(r'[0-9]', '', p)
    if re.match(r'[A-Z]+', p): ...
    """
    # 1. Remove stress digits
    p_clean = re.sub(r'[0-9]', '', p)
    
    # 2. Check if it is a valid phoneme label
    # We also check against VALID_PHONES to be safe
    if re.match(r'^[A-Z]+$', p_clean) and p_clean in VALID_PHONES:
        return p_clean
    return None

def process_chunk(lines):
    global g_dict, g_g2p
    results = []
    
    for line in lines:
        words = line.strip().lower().split()
        if not words: continue
        
        sentence_phonemes = []
        
        # Iterate through words and insert SIL in between
        for i, word in enumerate(words):
            
            # Get phonemes for this word
            if word in g_dict:
                # CMU Dict returns [['K', 'AE1', 'T']]
                raw_ph = g_dict[word][0]
            else:
                # Neural Fallback
                raw_ph = g_g2p(word)
            
            # Clean and add phonemes for this word
            for p in raw_ph:
                # Handle g2p_en special chars just in case
                if not isinstance(p, str): continue
                
                cleaned = clean_phoneme(p)
                if cleaned:
                    sentence_phonemes.append(cleaned)
            
            # INSERT SILENCE (SPACE)
            # Add SIL after every word, except the very last one?
            # User logic says: "if p == ' ': append SIL". 
            # In a sentence "hello world", there is a space between them.
            # We append SIL after this word, provided it's not the absolute last word.
            if i < len(words) - 1:
                sentence_phonemes.append('SIL')
        
        if sentence_phonemes:
            results.append(" ".join(sentence_phonemes))
            
    return results

# --- MAIN PIPELINE ---
def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download Corpus
    if not os.path.exists(RAW_TEXT_FILE):
        print("Downloading LibriSpeech Corpus...")
        zip_path = RAW_TEXT_FILE + ".gz"
        urllib.request.urlretrieve(CORPUS_URL, zip_path)
        print("Unzipping...")
        with gzip.open(zip_path, 'rb') as f_in:
            with open(RAW_TEXT_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(zip_path)

    # 2. Phonemize
    if not os.path.exists(PHONEME_FILE):
        print("Phonemizing (Inserting 'SIL' at word boundaries)...")
        
        with open(RAW_TEXT_FILE, 'r') as f:
            total_lines = sum(1 for _ in f)

        num_workers = max(1, mp.cpu_count() - 1)
        chunk_size = 5000
        pool = mp.Pool(num_workers, initializer=init_worker)
        
        with open(RAW_TEXT_FILE, 'r') as f_in, open(PHONEME_FILE, 'w') as f_out:
            chunk = []
            pbar = tqdm(total=total_lines, desc="Processing")
            
            def write_callback(res):
                for line in res:
                    f_out.write(line + "\n")
                pbar.update(len(res))

            for line in f_in:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    pool.apply_async(process_chunk, (chunk,), callback=write_callback)
                    chunk = []
            
            if chunk:
                pool.apply_async(process_chunk, (chunk,), callback=write_callback)
            
            pool.close()
            pool.join()
            pbar.close()

    # 3. Train KenLM
    print("\nTraining KenLM with SIL tokens...")
    # -o 6: 6-gram context is vital here to capture cross-word patterns like "T SIL D"
    cmd_arpa = [KENLM_LMPLZ, "-o", "6", "-S", "80%", "--text", PHONEME_FILE, "--arpa", ARPA_FILE, "--discount_fallback"]
    
    try:
        subprocess.check_call(cmd_arpa)
    except Exception as e:
        print(f"Error running lmplz: {e}")
        sys.exit(1)

    # 4. Binary
    print("Converting to Binary...")
    cmd_bin = [KENLM_BUILD_BINARY, ARPA_FILE, BINARY_FILE]
    subprocess.check_call(cmd_bin)
    
    print("\n---------------------------------------------------")
    print(f"SUCCESS. Model saved to: {BINARY_FILE}")
    print("---------------------------------------------------")
    print("Example Sequence the LM expects:")
    print("HH AE L OW SIL W ER L D")

if __name__ == "__main__":
    main()