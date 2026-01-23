import multiprocessing as mp
import re
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/data2/brain2text/lm_training_data/c4_cleaned.txt"
OUTPUT_FILE = "/data2/brain2text/lm_training_data/c4_strict_cleaned.txt"

# The Regex from the paper:
# Allows: a-z, A-Z, space, apostrophe, comma, period, ?, !
# We also allow \n (newline) to preserve line breaks
# Logic: ^ [allowed_chars]+ $ means the WHOLE line must match
ALLOWED_PATTERN = re.compile(r"^[a-zA-Z\s.,?!']+$")

def filter_line(line):
    """
    Returns the line if it ONLY contains allowed characters.
    Returns None if it contains numbers, brackets, $, %, etc.
    """
    if not line.strip():
        return None
    
    # Check if the whole line matches the allowed set
    if ALLOWED_PATTERN.match(line):
        return line
    return None

def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # Use strict filtering
    num_cores = max(1, mp.cpu_count() - 2)
    print(f"Running strict A-Z filter with {num_cores} cores...")

    with mp.Pool(processes=num_cores) as pool:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            line_generator = (line for line in f_in)
            
            kept = 0
            total = 0
            
            for result in tqdm(pool.imap(filter_line, line_generator, chunksize=2000)):
                total += 1
                if result:
                    f_out.write(result)
                    kept += 1
                    
    print(f"Done. Kept {kept} out of {total} lines.")

if __name__ == "__main__":
    main()