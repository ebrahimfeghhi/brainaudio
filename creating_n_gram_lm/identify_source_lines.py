import sys
import linecache

def identify_source_lines(phoneme_file, text_file):
    print(f"Mapping 'SIL NG' errors from:\n  {phoneme_file}\n  to source text in:\n  {text_file}\n")
    
    # Configuration
    forbidden_seq = ("SIL", "NG")
    implicit_start = "NG"
    max_examples = 20  # Limit output to first 20 errors
    
    found_count = 0
    
    try:
        # Open phoneme file to scan for errors
        with open(phoneme_file, 'r', encoding='utf-8') as f_phone:
            for line_idx, line in enumerate(f_phone, 1):
                
                parts = line.strip().split()
                if not parts: continue
                
                is_error = False
                
                # Check 1: Starts with NG
                if parts[0] == implicit_start:
                    is_error = True
                    reason = "Starts with NG"
                    
                # Check 2: Contains SIL NG (if not already caught)
                if not is_error and "SIL NG" in line:
                    for i in range(len(parts) - 1):
                        if parts[i] == forbidden_seq[0] and parts[i+1] == forbidden_seq[1]:
                            is_error = True
                            reason = "Contains SIL NG"
                            break
                
                if is_error:
                    found_count += 1
                    
                    # Fetch the corresponding line from the source text file
                    # linecache.getline is 1-based index
                    source_line = linecache.getline(text_file, line_idx).strip()
                    
                    print(f"[Line {line_idx}] {reason}")
                    print(f"  Phonemes: {' '.join(parts[:-1])}  ") # Print first few phonemes
                    print(f"  SOURCE:   \"{source_line}\"")
                    print("-" * 50)
                    
                    if found_count >= max_examples:
                        print(f"...Stopping after {max_examples} examples.")
                        break
                        
        if found_count == 0:
            print("No errors found! Are you pointing to the correct files?")

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # UPDATE THESE PATHS
    path_to_phonemes = "/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_lm_train.txt"
    path_to_source_text = "/home/ebrahim/brainaudio/creating_n_gram_lm/lm_corpus_cleaned.txt"
    
    identify_source_lines(path_to_phonemes, path_to_source_text)