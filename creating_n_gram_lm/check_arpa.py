import sys

def check_arpa_file(filename):
    print(f"Checking {filename}...")
    
    # 1. Define the Bigrams we want to benchmark
    # Format: (Bigram String, Expected Category)
    targets = {
        # HIGH PROBABILITY (Common English patterns)
        "DH AH":  "HIGH (Common: 'the', 'that')",
        "AH N":   "HIGH (Common: 'an', 'and')",
        "S T":    "HIGH (Common cluster: 'st')",
        "SIL DH": "HIGH (Sentence starter: 'The...')",
        
        # LOW PROBABILITY (Rare or Boundary-only)
        "NG HH":  "LOW  (Boundary: 'King Henry')",
        "Z SH":   "LOW  (Boundary: 'Cheese shop')",
        
        # PHONOTACTIC VIOLATIONS (Should be very low or non-existent)
        "SIL NG": "VERY LOW (English words can't start with NG)",
        "HH NG":  "VERY LOW (Structure violation)"
    }
    
    found_targets = {}
    vocab = set()
    
    try:
        with open(filename, 'r') as f:
            section = None
            
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Detect sections
                if line == "\\1-grams:":
                    section = "1-grams"
                    print("\n--- VOCABULARY CHECK (1-grams) ---")
                    continue
                elif line == "\\2-grams:":
                    section = "2-grams"
                    continue
                elif line == "\\3-grams:":
                    section = "3-grams"
                    # We can stop reading after 2-grams for this specific check
                    break
                elif line.startswith("\\end\\"):
                    break

                parts = line.split()
                
                # PROCESS 1-GRAMS
                if section == "1-grams":
                    # Format: log_prob token [backoff]
                    if len(parts) >= 2:
                        token = parts[1]
                        log_prob = float(parts[0])
                        vocab.add(token)
                        
                        # Print check for specific tokens
                        if token in ["SIL", "AH", "ZH", "<s>"]:
                            print(f"Token: {token:<4} | LogProb: {log_prob:.6f}")

                # PROCESS 2-GRAMS
                elif section == "2-grams":
                    # Format: log_prob token1 token2 [backoff]
                    if len(parts) >= 3:
                        log_prob = float(parts[0])
                        # Join the next two parts to form the bigram "TOKEN1 TOKEN2"
                        bigram = f"{parts[1]} {parts[2]}"
                        
                        if bigram in targets:
                            found_targets[bigram] = log_prob

        # --- REPORTING ---
        print(f"\nTotal Vocab Size: {len(vocab)}")
        
        print("\n--- BIGRAM PROBABILITY CHECK ---")
        print(f"{'BIGRAM':<10} | {'LOGPROB':<10} | {'EXPECTATION'}")
        print("-" * 55)
        
        for bigram, desc in targets.items():
            prob = found_targets.get(bigram, "N/A")
            prob_str = f"{prob:.6f}" if isinstance(prob, float) else "NOT FOUND"
            print(f"{bigram:<10} | {prob_str:<10} | {desc}")

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")

if __name__ == "__main__":
    # Default to the file you mentioned, or take from command line
    file_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_6gram.arpa"
    check_arpa_file(file_path)