import os
import sys
import time
import truecase
from truecase import Trainer
from tqdm import tqdm
import nltk

# Ensure nltk punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
# The file you generated with your download script
INPUT_CORPUS = "truecase_training_corpus_2m.txt"  # Use small for testing, then switch to full

# The name of the file where the model will be saved
OUTPUT_MODEL = "brainaudio_truecase.dist"

def train_model():
    # 1. Validation
    if not os.path.exists(INPUT_CORPUS):
        print(f"ERROR: Could not find input file: {INPUT_CORPUS}")
        print("Please run your download script first.")
        sys.exit(1)

    print(f"--- Starting Truecase Training ---")
    print(f"Input Corpus:  {INPUT_CORPUS}")
    print(f"Output Model:  {OUTPUT_MODEL}")
    print(f"Target Size:   ~2 GB of text")
    print("-" * 40)

    # 2. Initialize Trainer
    # The trainer builds N-gram statistics (Unigram, Backward Bigram, Forward Bigram)
    trainer = Trainer()

    # 3. Train
    # Read the corpus file and TOKENIZE each sentence (the Trainer expects list of word lists)
    start_time = time.time()
    print("Loading and tokenizing corpus...")
    corpus = []
    with open(INPUT_CORPUS, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if line:
                # Tokenize into words - Trainer expects List[List[str]]
                tokens = nltk.word_tokenize(line)
                corpus.append(tokens)
    
    print(f"Loaded {len(corpus):,} tokenized sentences")
    
    # Print first 5 sentences as a sanity check
    print("\nFirst 5 tokenized sentences:")
    for i, sent in enumerate(corpus[:5]):
        print(f"  {i+1}. {sent[:15]}{'...' if len(sent) > 15 else ''}")
    print()
    
    print("Training in progress... (This typically takes 5-15 minutes)")
    
    try:
        trainer.train(tqdm(corpus, desc="Training"))
    except Exception as e:
        print(f"\nCRITICAL ERROR during training: {e}")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} minutes.")

    # 4. Save
    print(f"Saving model to disk...")
    trainer.save_to_file(OUTPUT_MODEL)
    
    # Check resulting file size
    size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"Model saved successfully! Size: {size_mb:.2f} MB")

    # 5. Sanity Check
    print("-" * 40)
    print("Running Sanity Check...")
    
    # Load the truecaser with the new model to test it
    truecaser = truecase.TrueCaser(OUTPUT_MODEL)
    
    test_sentences = [
        "he is a member of the royal irish academy.",
        "i use pytorch and tensorflow for machine learning.",
        "i live in the us but i am moving to the uk."
    ]
    
    for sent in test_sentences:
        result = truecaser.get_true_case(sent)
        print(f"\nIn:  {sent}")
        print(f"Out: {result}")

    print("-" * 40)
    print("DONE. You can now use this .dist file in your pipeline.")

if __name__ == "__main__":
    train_model()