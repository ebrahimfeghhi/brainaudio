import kenlm
from typing import List

class KenLMFusion:
    def __init__(self, model_path: str, weight: float = 1.0):
        print(f"Loading KenLM from {model_path}...")
        try:
            self.model = kenlm.Model(model_path)
        except Exception as e:
            print(f"Error loading KenLM: {e}")
            raise
        
        self.weight = weight

    def score_continuations(self, contexts: List[str], candidate_words_list: List[List[str]]) -> List[List[float]]:
        results = []
        
        for context, candidates in zip(contexts, candidate_words_list):
            state = kenlm.State()
            self.model.BeginSentenceWrite(state)
            
            # 1. Process Context (Lowercased)
            if context:
                # "The cat" -> "the" "cat"
                for word in context.lower().split():
                    next_state = kenlm.State()
                    self.model.BaseScore(state, word, next_state)
                    state = next_state
            
            # 2. Score Candidates (Lowercased)
            beam_scores = []
            for word in candidates:
                next_state = kenlm.State()
                
                # "Apple" -> "apple"
                word_lower = word.lower()
                
                # Get log10 prob
                score_log10 = self.model.BaseScore(state, word_lower, next_state)
                
                # Convert log10 -> ln (natural log) for PyTorch
                score_ln = score_log10 * 2.30258509
                
                beam_scores.append(self.weight * score_ln)
            
            results.append(beam_scores)
            
        return results