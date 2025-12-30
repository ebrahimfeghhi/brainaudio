import tqdm
import re
from multiprocessing import Pool, cpu_count

# --- SETTINGS ---
INPUT_FILE = "lm_corpus_cleaned_large.txt"
OUTPUT_FILE = "phoneme_lm_train_large.txt"
LEXICON_FILE = "/data2/brain2text/lm/lexicon_phonemes.txt"
CHUNK_SIZE = 50000  # Number of lines per chunk (adjusts memory usage)

# Global variable for workers
LEXICON_DICT = {}

def load_lexicon(path):
    print(f"Loading lexicon from {path}...")
    lex_dict = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')[0].strip().split()
                if len(parts) >= 2:
                    word = parts[0].upper()
                    phonemes = " ".join(parts[1:])
                    lex_dict[word] = phonemes
        print(f"Lexicon loaded: {len(lex_dict)} entries.")
        return lex_dict
    except FileNotFoundError:
        print(f"Error: Lexicon not found at {path}")
        exit(1)

def init_worker(shared_lexicon):
    global LEXICON_DICT
    LEXICON_DICT = shared_lexicon

def process_chunk(lines):
    """
    Worker receives a list of strings (a chunk) and returns a list of result strings.
    """
    results = []
    for line in lines:
        text = line.strip()
        if not text: continue
        
        words = re.findall(r"\b[A-Z]+(?:'[A-Z]+)?\b", text)
        phoneme_sent = []
        
        for w in words:
            # Direct lookup
            if w in LEXICON_DICT:
                phoneme_sent.append(LEXICON_DICT[w])
        
        if phoneme_sent:
            full_string = " SIL ".join(phoneme_sent) + " SIL"
            results.append(full_string)
            
    return results

def chunk_reader(file_path, chunk_size=10000):
    """
    Generator that yields chunks of lines from the file.
    Crucial for low memory usage.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    lexicon_data = load_lexicon(LEXICON_FILE)
    
    # We estimate total chunks just for the progress bar (204M lines / 50k)
    # If estimate is wrong, tqdm just goes over 100%, which is fine.
    ESTIMATED_TOTAL_LINES = 204_224_098
    total_chunks = ESTIMATED_TOTAL_LINES // CHUNK_SIZE
    
    print(f"Streaming {INPUT_FILE} using {cpu_count()} cores...")

    num_processes = cpu_count()
    
    with open(OUTPUT_FILE, "w") as f_out:
        with Pool(processes=num_processes, initializer=init_worker, initargs=(lexicon_data,)) as pool:
            
            # imap consumes the generator lazily
            processor = pool.imap(process_chunk, chunk_reader(INPUT_FILE, CHUNK_SIZE))
            
            for result_chunk in tqdm.tqdm(processor, total=total_chunks):
                # result_chunk is a list of converted strings
                if result_chunk:
                    f_out.write("\n".join(result_chunk) + "\n")

    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()