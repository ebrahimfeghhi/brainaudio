"""
Train a MosesTruecaser on the 2M sentence corpus.
Uses sacremoses library (faster than the truecase library).
"""

from sacremoses import MosesTruecaser, MosesTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CORPUS = "truecase_training_corpus_2m.txt"
OUTPUT_MODEL = "brainaudio_moses.truecasemodel"

def main():
    print(f"Training MosesTruecaser on: {INPUT_CORPUS}")
    
    # Initialize tokenizer and truecaser
    mtok = MosesTokenizer(lang='en')
    mtr = MosesTruecaser()
    
    # Read and tokenize all lines with progress bar
    print("Loading and tokenizing corpus...")
    with open(INPUT_CORPUS, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Tokenizing {len(lines):,} sentences...")
    tokenized_docs = [mtok.tokenize(line.strip()) for line in tqdm(lines)]
    
    # Train the truecaser
    print("Training truecaser...")
    mtr.train(tokenized_docs, save_to=OUTPUT_MODEL)
    
    print(f"Model saved to: {OUTPUT_MODEL}")
    
    # Quick test
    print("\n--- Quick Test ---")
    test_sentences = [
        "hello my name is john",
        "i live in new york city",
        "the quick brown fox jumps over the lazy dog",
    ]
    for sent in test_sentences:
        # truecase() expects a string, returns a string
        truecased = mtr.truecase(sent)
        print(f"  {sent!r} -> {truecased!r}")

if __name__ == "__main__":
    main()
