import os

# ==========================================
# CONFIGURATION
# ==========================================

# 1. INPUT: Point to your EXISTING massive text file
INPUT_CORPUS = "/home/ebrahim/brainaudio/creating_n_gram_lm/final_cased_12B.txt"

# 2. OUTPUTS
OUTPUT_NAME = "final_cased_12B"
ARPA_FILE = f"{OUTPUT_NAME}.arpa"
BINARY_FILE = f"{OUTPUT_NAME}.binary"

# 3. KENLM PATHS (Absolute paths)
BIN_LMPLZ = "/home/ebrahim/kenlm/build/bin/lmplz"
BIN_BUILD_BINARY = "/home/ebrahim/kenlm/build/bin/build_binary"
TEMP_DIR = "./kenlm_tmp"

# 4. SETTINGS
NGRAM_ORDER = 4
# Prune singleton 3-grams and 4-grams (Safe for 12B words)
PRUNE_ARGS = "0 1 1" 
# Use 80% RAM for sorting
MEMORY_ARGS = "-S 80%" 

def main():
    # Verify input exists
    if not os.path.exists(INPUT_CORPUS):
        print(f"ERROR: Could not find input file: {INPUT_CORPUS}")
        return

    # Create temp dir for KenLM sorting
    os.makedirs(TEMP_DIR, exist_ok=True)

    print(f"=== RESUMING TRAINING ON EXISTING DATA ===")
    print(f"Input: {INPUT_CORPUS}")
    print(f"Output: {BINARY_FILE}")
    print("-" * 40)

    # ---------------------------------------------------------
    # STEP 1: TRAIN (With on-the-fly cleaning)
    # ---------------------------------------------------------
    print(">>> Phase 2: Training N-gram Model...")
    print("    (Piping via 'sed' to remove <s> tags that caused the crash...)")
    
    # We use 'sed' to delete <s> and </s> tags from the stream before KenLM sees them.
    # This prevents the "Special word <s> is not allowed" error.
    cmd_train = (
        f"sed 's/<s>//g; s/<\\/s>//g' {INPUT_CORPUS} | "
        f"{BIN_LMPLZ} -o {NGRAM_ORDER} --prune {PRUNE_ARGS} "
        f"{MEMORY_ARGS} -T {TEMP_DIR} "
        f"> {ARPA_FILE}"
    )
    
    print(f"\nRunning Command:\n{cmd_train}\n")
    
    ret = os.system(cmd_train)
    if ret != 0:
        print("\n[CRITICAL ERROR] lmplz failed. Check the logs above.")
        return

    # ---------------------------------------------------------
    # STEP 2: BINARIZE
    # ---------------------------------------------------------
    print("\n>>> Phase 3: Binarizing to standard format...")
    cmd_bin = f"{BIN_BUILD_BINARY} {ARPA_FILE} {BINARY_FILE}"
    print(f"Running: {cmd_bin}")
    os.system(cmd_bin)

    # ---------------------------------------------------------
    # STEP 3: CLEANUP
    # ---------------------------------------------------------
    print("\n>>> Cleanup...")
    # We do NOT delete the INPUT_CORPUS, only the intermediate ARPA
    if os.path.exists(ARPA_FILE): 
        print(f"Removing intermediate ARPA file: {ARPA_FILE}")
        os.remove(ARPA_FILE)
    
    # Optional: Clean up temp dir if empty
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass

    print("-" * 40)
    print(f"SUCCESS! Your model is ready: {os.path.abspath(BINARY_FILE)}")

if __name__ == "__main__":
    main()