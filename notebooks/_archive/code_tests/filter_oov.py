import multiprocessing as mp
import string
import sys
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/data2/brain2text/lm_training_data/c4_cleaned.txt"
OUTPUT_FILE = "/data2/brain2text/lm_training_data/c4_filtered.txt"
VOCAB_FILE = "/data2/brain2text/lm/words.txt"
OOV_THRESHOLD = 0.20  # Discard if > 20% of words are unknown

# Global variable for workers
vocab_set = None

def init_worker():
    """
    Load the vocabulary into memory for each worker process.
    Using a set() provides O(1) lookup speed.
    """
    global vocab_set
    vocab_set = set()
    
    # Load punctuation translation table to strip symbols efficiently
    # We treat the vocab file as a list of valid lowercased tokens
    try:
        with open(VOCAB_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip().split()[0] # Take first token if multiple columns exist
                vocab_set.add(word.lower())
    except Exception as e:
        print(f"Error loading vocab: {e}")

def process_line(line):
    """
    Returns the line if OOV rate < Threshold, otherwise None.
    """
    global vocab_set
    
    if not line.strip():
        return None

    # Split into tokens by whitespace
    raw_tokens = line.strip().split()
    
    total_valid_words = 0
    oov_count = 0
    
    # Punctuation stripper
    # specific check to speed up the loop
    table = str.maketrans('', '', string.punctuation)

    for token in raw_tokens:
        # 1. Lowercase and strip punctuation (e.g., "Hello," -> "hello")
        clean_word = token.lower().translate(table)
        
        # 2. Skip empty strings (e.g., if the token was just "!")
        if not clean_word:
            continue
            
        # 3. Check vocabulary
        total_valid_words += 1
        if clean_word not in vocab_set:
            oov_count += 1

    # Avoid division by zero for lines that were only punctuation
    if total_valid_words == 0:
        return None

    # Calculate Rate
    oov_rate = oov_count / total_valid_words
    
    # Return line if it passes the filter
    if oov_rate <= OOV_THRESHOLD:
        return line
    else:
        return None

def main():
    # Use 'spawn' for safety, though 'fork' is faster for simple reads. 
    # 'spawn' is safer with large global sets.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    num_cores = max(1, mp.cpu_count() - 2)
    print(f"Filtering OOV with {num_cores} cores...")

    with mp.Pool(processes=num_cores, initializer=init_worker) as pool:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            # Generator for memory efficiency
            line_generator = (line for line in f_in)
            
            kept_count = 0
            
            # Use imap with chunksize for performance
            for result in tqdm(pool.imap(process_line, line_generator, chunksize=2000)):
                if result:
                    f_out.write(result) # Line already has \n
                    kept_count += 1
                    
    print(f"Done. Filtered sentences saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()