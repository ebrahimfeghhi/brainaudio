import sys

def check_forbidden_starts(filename, forbidden_seq=("SIL", "NG"), implicit_start="NG"):
    print(f"Scanning {filename} for forbidden phoneme sequences...")
    print(f"1. Explicit sequence: {forbidden_seq}")
    print(f"2. Implicit start: Line starts with '{implicit_start}'")
    print("-" * 60)

    forbidden_count = 0
    implicit_count = 0
    
    # Limit example output to avoid flooding the console
    max_examples = 5
    examples_found = 0

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                # Clean whitespace and split into phonemes
                parts = line.strip().split()
                
                if not parts:
                    continue

                # CHECK 1: Implicit Start (Line starts with NG)
                # In KenLM, start-of-line is treated as context <s> (Silence)
                if parts[0] == implicit_start:
                    implicit_count += 1
                    if examples_found < max_examples:
                        print(f"[Line {line_idx}] STARTS with {implicit_start}: {' '.join(parts[:10])}...")
                        examples_found += 1

                # CHECK 2: Explicit Sequence (SIL followed by NG)
                # Iterate through the line to find "SIL" then check if next is "NG"
                # We use a simple loop or zip to find the pair
                for i in range(len(parts) - 1):
                    if parts[i] == forbidden_seq[0] and parts[i+1] == forbidden_seq[1]:
                        forbidden_count += 1
                        if examples_found < max_examples:
                            # Show context around the error
                            start = max(0, i - 2)
                            end = min(len(parts), i + 4)
                            context = " ".join(parts[start:end])
                            print(f"[Line {line_idx}] FOUND {forbidden_seq}: ...{context}...")
                            examples_found += 1
                        # Break after finding one per line to save time, 
                        # or remove break if you want total instance count (slower)
                        break 
        
        print("-" * 60)
        print("SCAN COMPLETE")
        print(f"Lines starting with '{implicit_start}': {implicit_count}")
        print(f"Explicit '{forbidden_seq[0]} {forbidden_seq[1]}' sequences: {forbidden_count}")
        
        if implicit_count > 0 or forbidden_count > 0:
            print("\nVERDICT: Dirty data confirmed. These lines are causing your -2.2 logprob.")
        else:
            print("\nVERDICT: Clean. If -2.2 logprob persists, it is purely a Backoff/Smoothing artifact.")

    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Update this path to your actual training file
    # If the file is huge, this might take a minute or two.
    file_path = "/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_lm_train.txt" 
    
    check_forbidden_starts(file_path)