import itertools

# --- SETTINGS ---
TEXT_FILE = "lm_corpus_cleaned.txt"
PHONEME_FILE = "phoneme_lm_train.txt"
NUM_SAMPLES = 20  # How many lines to check

def verify_alignment():
    print(f"Checking alignment between:\n  Text: {TEXT_FILE}\n  Phonemes: {PHONEME_FILE}\n")
    print("-" * 80)
    print(f"{'TEXT':<50} | {'PHONEMES'}")
    print("-" * 80)

    try:
        with open(TEXT_FILE, "r") as f_text, open(PHONEME_FILE, "r") as f_phone:
            # zip allows us to read both files line-by-line simultaneously
            for i, (line_text, line_phone) in enumerate(zip(f_text, f_phone)):
                
                if i >= NUM_SAMPLES:
                    break
                
                # Strip newlines for clean printing
                t = line_text.strip()
                p = line_phone.strip()
                
                # Truncate text if it's too long for the column
                if len(t) > 48:
                    t = t[:45] + "..."
                
                print(f"{t:<50} | {p}")

        print("-" * 80)
        
        # --- CHECK TOTAL LINE COUNTS ---
        print("\nCounting total lines in both files (this might take a second)...")
        with open(TEXT_FILE, "r") as f:
            count_text = sum(1 for _ in f)
        with open(PHONEME_FILE, "r") as f:
            count_phone = sum(1 for _ in f)
            
        print(f"Text File Lines:    {count_text}")
        print(f"Phoneme File Lines: {count_phone}")
        
        if count_text == count_phone:
            print("✅ PERFECT MATCH: Line counts are identical.")
        else:
            diff = abs(count_text - count_phone)
            print(f"⚠️ WARNING: Line counts differ by {diff} lines.")
            print("   (This is normal if some lines were purely punctuation and got skipped during G2P)")

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file. {e}")

if __name__ == "__main__":
    verify_alignment()