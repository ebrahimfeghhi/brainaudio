import math
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from brainaudio.inference.decoder.ngram_lm_fusion import KenLMFusion

def test_kenlm_scoring(model_path):
    """
    Tests the KenLM fusion module by scoring common vs rare continuations.
    """
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        return

    print(f"Loading KenLM model from: {model_path} ...")
    try:
        # Initialize with weight=1.0 to see raw log probabilities
        lm = KenLMFusion(model_path=model_path, weight=1.0)
    except Exception as e:
        print(f"[Error] Failed to load KenLM: {e}")
        return

    # --- Test 1: Standard Sentence Completion ---
    context1 = "thank you very"
    candidates1 = ["much", "mach", "match", "mud"]
    
    print(f"\n--- Test 1: Context = '{context1}' ---")
    
    # Passing list of contexts and list of candidate lists
    scores = lm.score_continuations(
        contexts=[context1], 
        candidate_words_list=[candidates1]
    )
    
    # Retrieve the first beam's scores
    beam_scores = scores[0]
    
    # Sort results by score to show what the model prefers
    results = sorted(zip(candidates1, beam_scores), key=lambda x: x[1], reverse=True)

    for word, score in results:
        # Calculate approximate raw probability for visualization (p = e^score)
        prob = math.exp(score)
        print(f"Word: {word:<10} | Log Score: {score:8.4f} | Prob: {prob:.6f}")

    # --- Test 2: Start of Sentence (Empty Context) ---
    # N-gram models handle <s> (start of sentence) implicitly
    context2 = "" 
    candidates2 = ["the", "because", "UNK"] # "Zjxkv" should be very low/OOV
    
    print(f"\n--- Test 2: Start of Sentence (Context = '') ---")
    scores_start = lm.score_continuations([context2], [candidates2])
    
    for word, score in zip(candidates2, scores_start[0]):
        print(f"Word: {word:<10} | Log Score: {score:8.4f}")

    print("\n✅ Test Complete.")

if __name__ == "__main__":
    # Replace with the path to your actual ARPA or Binary file
    # Example: python scripts/test_kenlm_fusion.py /path/to/4gram.binary
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    else:
        # Hardcode your path here for quick testing
        MODEL_PATH = "/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm" 
    
    test_kenlm_scoring(MODEL_PATH)