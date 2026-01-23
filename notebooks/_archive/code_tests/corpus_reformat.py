import spacy
import multiprocessing as mp
from tqdm import tqdm
import os

# --- CONFIGURATION ---
INPUT_FILE = "/data2/brain2text/lm_training_data/raw_subtitles.txt" # Double check this path
OUTPUT_FILE = "/data2/brain2text/lm_training_data/subtitles_formatted.txt"
MIN_WORDS = 1
MAX_WORDS = 22
BATCH_SIZE = 2000 # Smaller batch size for smoother progress updates

# Global variable for workers
nlp = None

def init_worker():
    """
    Initialize spaCy in each worker.
    """
    global nlp
    # Ensure we don't use too many threads inside each worker
    os.environ["OMP_NUM_THREADS"] = "1"
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer", "tagger"])
    nlp.add_pipe("sentencizer")

def process_line(line):
    """
    Process a single line (document).
    Returns a string of joined sentences (with newlines) or None.
    """
    global nlp
    if not line.strip():
        return None

    results = []
    try:
        doc = nlp(line)
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text.split()) >= MIN_WORDS and len(text.split()) <= MAX_WORDS:
                results.append(text)
    except Exception:
        return None

    if results:
        return "\n".join(results)
    return None

def main():
    # 1. CRITICAL FIX: Use 'spawn' to prevent deadlocks with spaCy
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Context might already be set

    # Use slightly fewer cores to leave room for system IO
    num_cores = max(1, mp.cpu_count() - 4)
    print(f"Starting processing with {num_cores} cores...")

    # Count total lines for the progress bar (Optional, takes time for 20GB. Can comment out.)
    # print("Counting lines for progress bar... (this might take a minute)")
    # total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore'))
    # print(f"Total lines: {total_lines}")

    with mp.Pool(processes=num_cores, initializer=init_worker) as pool:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            # Create a generator to read lines one by one
            line_generator = (line for line in f_in)
            
            # pool.imap processes items in parallel but yields results in order
            # chunksize helps reduce overhead
            for result in tqdm(pool.imap(process_line, line_generator, chunksize=500)):
                if result:
                    f_out.write(result + "\n")
                    
    print("Done! Formatted file saved.")

if __name__ == "__main__":
    main()