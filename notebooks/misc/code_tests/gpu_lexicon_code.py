import torch
import collections
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class CTCHypothesis:
    """Mimics torchaudio.models.decoder.CTCHypothesis"""
    tokens: torch.Tensor
    words: List[str]
    score: float
    timesteps: torch.Tensor

class VectorizedLexicon:
    def __init__(self, lexicon_path: str, tokens: List[str]):
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        
        # 1. Build the Trie on CPU
        self.trie = {'children': {}, 'is_word': False, 'id': 0}
        self.node_count = 1
        
        with open(lexicon_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # Handling format: "CAT c a t"
                # If your file is just "cat", adjust parsing here.
                word = parts[0]
                spelling = parts[1:] if len(parts) > 1 else list(word)
                
                node = self.trie
                for char in spelling:
                    if char not in self.token_to_id:
                        continue # Skip unknown tokens
                    token_id = self.token_to_id[char]
                    if token_id not in node['children']:
                        node['children'][token_id] = {'children': {}, 'is_word': False, 'id': self.node_count}
                        self.node_count += 1
                    node = node['children'][token_id]
                node['is_word'] = True

        # 2. Flatten Trie to GPU Tensor: [Num_Nodes, Vocab_Size]
        self.vocab_size = len(tokens)
        self.transitions = torch.full((self.node_count, self.vocab_size), -1, dtype=torch.long)
        self.is_word_node = torch.zeros(self.node_count, dtype=torch.bool)
        
        # Queue for BFS traversal
        queue = [self.trie]
        while queue:
            node = queue.pop(0)
            u = node['id']
            if node['is_word']:
                self.is_word_node[u] = True
            
            for token_id, child_node in node['children'].items():
                v = child_node['id']
                self.transitions[u, token_id] = v
                queue.append(child_node)

    def to(self, device):
        self.transitions = self.transitions.to(device)
        self.is_word_node = self.is_word_node.to(device)
        return self


class GPUCTCDecoder:
    def __init__(self, 
                 lexicon_path: str, 
                 tokens: List[str], 
                 beam_size: int = 10, 
                 blank_token: str = "-",
                 space_token: str = "|"): # Explicit space token required
        
        self.beam_size = beam_size
        self.tokens = tokens
        self.blank_id = tokens.index(blank_token) if blank_token in tokens else 0
        self.space_id = tokens.index(space_token) if space_token in tokens else -1
        
        if self.space_id == -1:
            print("Warning: Space token not found in vocab. Multi-word decoding will fail.")

        self.lexicon = VectorizedLexicon(lexicon_path, tokens)
        
    def __call__(self, emissions: torch.Tensor) -> List[List[CTCHypothesis]]:
        device = emissions.device
        if self.lexicon.transitions.device != device:
            self.lexicon.to(device)

        B, T, V = emissions.shape
        
        # --- Initialize Beams ---
        beam_scores = torch.full((B, self.beam_size), -float('inf'), device=device)
        beam_scores[:, 0] = 0.0
        
        # Start at Trie Root (0)
        beam_trie_nodes = torch.zeros((B, self.beam_size), dtype=torch.long, device=device)
        
        # Start with Blank history
        beam_last_tokens = torch.full((B, self.beam_size), self.blank_id, dtype=torch.long, device=device)
        
        # Path History (CPU List)
        beam_paths = [[[] for _ in range(self.beam_size)] for _ in range(B)]

        # --- DECODING LOOP ---
        for t in range(T):
            # 1. Expand
            log_probs = torch.nn.functional.log_softmax(emissions[:, t, :], dim=-1)
            next_scores = beam_scores.unsqueeze(-1) + log_probs.unsqueeze(1)
            next_scores = next_scores.view(B, -1) 
            
            # 2. Transition Logic
            # [B, Beam * V]
            current_nodes_expanded = beam_trie_nodes.unsqueeze(-1).expand(-1, -1, V).reshape(B, -1)
            last_tokens_expanded = beam_last_tokens.unsqueeze(-1).expand(-1, -1, V).reshape(B, -1)
            candidate_tokens = torch.arange(V, device=device).reshape(1, 1, -1).expand(B, self.beam_size, -1).reshape(B, -1)
            
            # A. Standard Trie Lookup
            next_trie_nodes_lookup = self.lexicon.transitions[current_nodes_expanded, candidate_tokens]

            # B. Space / Word Boundary Logic
            # If token is Space AND current node is a valid word -> Reset to Root (0)
            # Else -> Invalid (-1)
            is_space = (candidate_tokens == self.space_id)
            current_is_word = self.lexicon.is_word_node[current_nodes_expanded]
            
            space_target = torch.where(current_is_word, 
                                       torch.tensor(0, device=device), 
                                       torch.tensor(-1, device=device))
            
            # Apply Space Override
            next_trie_nodes_lookup = torch.where(is_space, space_target, next_trie_nodes_lookup)

            # C. CTC Logic (Blank & Repeat)
            is_blank = (candidate_tokens == self.blank_id)
            is_repeat = (candidate_tokens == last_tokens_expanded)
            
            final_next_nodes = torch.where(
                is_blank | is_repeat,
                current_nodes_expanded, # Stay
                next_trie_nodes_lookup  # Move
            )
            
            # 3. Masking
            valid_mask = (final_next_nodes != -1)
            next_scores = torch.where(valid_mask, next_scores, torch.tensor(-float('inf'), device=device))

            # [[ INSERT LLM SCORING HERE ]]

            # 4. Prune
            top_scores, top_indices = torch.topk(next_scores, self.beam_size, dim=1)
            beam_scores = top_scores
            
            prev_beam_indices = top_indices // V
            new_token_indices = top_indices % V
            
            beam_trie_nodes = torch.gather(final_next_nodes, 1, top_indices)
            beam_last_tokens = new_token_indices

            # 5. Update Paths
            new_paths = []
            for b in range(B):
                batch_paths = []
                for k in range(self.beam_size):
                    prev_k = prev_beam_indices[b, k].item()
                    token = new_token_indices[b, k].item()
                    current_path = list(beam_paths[b][prev_k])
                    current_path.append(token)
                    batch_paths.append(current_path)
                new_paths.append(batch_paths)
            beam_paths = new_paths
            
        # --- Finalize Results ---
        results = []
        
        # Check validity of where the beams ended
        final_node_validity = self.lexicon.is_word_node[beam_trie_nodes]
        
        for b in range(B):
            hyps = []
            for k in range(self.beam_size):
                # OPTIONAL: Filter partial words
                # If you want to force valid ending: uncomment below
                # if not final_node_validity[b, k] and beam_trie_nodes[b,k] != 0:
                #     continue

                raw_tokens = beam_paths[b][k]
                
                # CTC Collapse
                collapsed_indices = []
                prev = -1
                for t_id in raw_tokens:
                    if t_id != self.blank_id and t_id != prev:
                        collapsed_indices.append(t_id)
                    prev = t_id
                
                # Convert to String
                # Note: We treat Space token as " " for display, others as looked up
                words = []
                for t_id in collapsed_indices:
                    if t_id == self.space_id:
                        words.append(" ")
                    else:
                        words.append(self.tokens[t_id])
                
                words_str = "".join(words)
                
                hyps.append(CTCHypothesis(
                    tokens=torch.tensor(collapsed_indices),
                    words=[words_str],
                    score=beam_scores[b, k].item(),
                    timesteps=torch.tensor([])
                ))
            results.append(hyps)
            
        return results

# ==========================================
# TEST CASE: Decoding "cat dog"
# ==========================================

# 1. Define Lexicon & Vocab
with open("lexicon.txt", "w") as f:
    f.write("CAT c a t\n")
    f.write("DOG d o g\n")

# Vocab includes Blank (-) and Space (|)
vocab = ["-", "c", "a", "t", "d", "o", "g", "|"] 
decoder = GPUCTCDecoder("lexicon.txt", vocab, beam_size=3, blank_token="-", space_token="|")

# 2. Construct "Forced" Emissions
# We want to force the path: c-a-t-|-d-o-g
# Total steps: 10
# Sequence: c(0), a(1), t(2), |(3), d(4), o(5), g(6), -(7..9)
T = 12
emissions = torch.full((1, T, len(vocab)), -10.0).cuda()

def set_prob(time, char):
    idx = vocab.index(char)
    emissions[0, time, idx] = 10.0

# "cat"
set_prob(0, "c")
set_prob(1, "a")
set_prob(2, "t")

# " " (Reset to root)
set_prob(3, "|") 

# "dog"
set_prob(4, "d")
set_prob(5, "o")
set_prob(6, "g")

# Rest blank
for t in range(7, T):
    set_prob(t, "-")

# 3. Run
print("Running Decoder...")
results = decoder(emissions)
# 4. Print
best_hyp = results[0][0]
print(f"Decoded: '{best_hyp.words[0]}'")
print(f"Score:   {best_hyp.score}")
# 4.