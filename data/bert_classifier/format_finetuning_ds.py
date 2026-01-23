import pandas as pd
import random
import os
from tqdm import tqdm

# --- CONFIGURATION ---
IN_DOMAIN_TRAIN_FILE = "/data2/brain2text/transcripts_train.txt"
IN_DOMAIN_VAL_FILE = "/data2/brain2text/transcripts_val.txt"
C4_FULL_FILE = "/data2/brain2text/lm_training_data/c4_strict_cleaned.txt"

OUTPUT_TRAIN_CSV = "bert_training_data.csv"
OUTPUT_VAL_CSV = "bert_val_data.csv"
OUTPUT_REMAINDER_POOL = "c4_selection_pool.txt"

# Ratio of OOD to In-Domain (e.g., 2.0 means 2 OOD sentences for every 1 In-Domain)
OOD_TRAIN_RATIO = 2.0
OOD_VAL_RATIO = 1.0

def create_split():
    # 1. Load In-Domain Data
    print("Loading In-Domain data...")
    with open(IN_DOMAIN_TRAIN_FILE, 'r') as f:
        id_train = [line.strip() for line in f if line.strip()]
    
    with open(IN_DOMAIN_VAL_FILE, 'r') as f:
        id_val = [line.strip() for line in f if line.strip()]

    print(f"In-Domain Train: {len(id_train)}")
    print(f"In-Domain Val:   {len(id_val)}")

    # 2. Calculate Needed OOD Samples
    needed_train_ood = int(len(id_train) * OOD_TRAIN_RATIO)
    needed_val_ood = int(len(id_val) * OOD_VAL_RATIO)
    total_ood_needed = needed_train_ood + needed_val_ood
    
    print(f"Need {needed_train_ood} OOD for Train and {needed_val_ood} OOD for Val.")
    print(f"Total OOD to extract: {total_ood_needed}")

    # 3. Stream C4: Extract needed lines + Write remainder
    print("Streaming C4 to extract samples and save remainder...")
    
    extracted_ood = []
    lines_processed = 0
    
    with open(C4_FULL_FILE, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(OUTPUT_REMAINDER_POOL, 'w', encoding='utf-8') as f_pool:
        
        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
            
            # Phase 1: Fill the reservoir
            if len(extracted_ood) < total_ood_needed:
                extracted_ood.append(line)
            
            # Phase 2: Randomly replace items
            else:
                # current_n is the count of items seen so far (1-based index)
                # current_n = len(extracted_ood) + (count of items sent to pool)
                # simpler: lines_processed + 1
                current_n = lines_processed + 1
                
                # Probability to keep this new item is K / N
                prob_keep = total_ood_needed / current_n
                
                if random.random() < prob_keep:
                    # We KEEP this new line. 
                    # We must EVICT a random old line to the pool.
                    victim_idx = random.randint(0, total_ood_needed - 1)
                    victim_line = extracted_ood[victim_idx]
                    
                    # Write the victim (evicted line) to the pool file
                    f_pool.write(victim_line + "\n")
                    
                    # Replace it with the new line
                    extracted_ood[victim_idx] = line
                else:
                    # We DISCARD this new line directly to the pool
                    f_pool.write(line + "\n")

            lines_processed += 1

    if len(extracted_ood) < total_ood_needed:
        print(f"WARNING: C4 file was too small! Needed {total_ood_needed} but only got {len(extracted_ood)}")

    # 4. Shuffle and Split OOD Data
    # We shuffle the extracted chunk to ensure Training and Val OOD are random relative to each other
    print("Shuffling extracted OOD samples...")
    random.shuffle(extracted_ood)

    # Slice the list
    ood_train = extracted_ood[:needed_train_ood]
    ood_val = extracted_ood[needed_train_ood:]

    print(f"OOD Train created: {len(ood_train)}")
    print(f"OOD Val created:   {len(ood_val)}")

    # 5. Create Final DataFrames
    print("Creating CSVs...")
    
    # --- Prepare Train ---
    df_train_pos = pd.DataFrame({'text': id_train, 'label': 1})
    df_train_neg = pd.DataFrame({'text': ood_train, 'label': 0})
    df_train = pd.concat([df_train_pos, df_train_neg]).sample(frac=1).reset_index(drop=True)
    df_train.to_csv(OUTPUT_TRAIN_CSV, index=False)
    
    # --- Prepare Val ---
    df_val_pos = pd.DataFrame({'text': id_val, 'label': 1})
    df_val_neg = pd.DataFrame({'text': ood_val, 'label': 0})
    df_val = pd.concat([df_val_pos, df_val_neg]).sample(frac=1).reset_index(drop=True)
    df_val.to_csv(OUTPUT_VAL_CSV, index=False)

    print("Done!")
    print(f"Train set saved to: {OUTPUT_TRAIN_CSV}")
    print(f"Val set saved to:   {OUTPUT_VAL_CSV}")
    print(f"Remainder saved to: {OUTPUT_REMAINDER_POOL}")

if __name__ == "__main__":
    create_split()