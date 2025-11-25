# Neural Language Model Fusion - Quick Start Guide

## Overview

Neural LM fusion has been integrated into the CTC beam search decoder. This allows you to use modern language models (GPT-2, LLaMA, etc.) to improve decoding quality by rescoring hypotheses at word boundaries.

## Basic Usage

### 1. Without LM Fusion (Baseline)

```python
from brainaudio.inference.decoder import BatchedBeamCTCComputer, LexiconConstraint

# Load lexicon
lexicon = LexiconConstraint.from_file_paths(
    tokens_file="path/to/tokens.txt",
    lexicon_file="path/to/lexicon.txt",
    device=torch.device('cuda:0'),
)

# Create decoder (no LM fusion)
decoder = BatchedBeamCTCComputer(
    blank_index=0,
    beam_size=10,
    lexicon=lexicon,
    allow_cuda_graphs=False,  # Disable for debugging
)

# Run beam search
result = decoder(logits, logits_lengths)
```

### 2. With Dummy LM Fusion (Testing)

```python
from brainaudio.inference.decoder import BatchedBeamCTCComputer, LexiconConstraint, DummyLMFusion

# Create dummy LM (returns uniform scores)
lm_fusion = DummyLMFusion(weight=0.3)

# Create decoder with LM fusion
decoder = BatchedBeamCTCComputer(
    blank_index=0,
    beam_size=10,
    lexicon=lexicon,
    lm_fusion=lm_fusion,  # Add LM fusion
    allow_cuda_graphs=False,
)

# Run beam search (same API)
result = decoder(logits, logits_lengths)
```

### 3. With GPT-2 LM Fusion (Real Usage)

```python
from brainaudio.inference.decoder import BatchedBeamCTCComputer, LexiconConstraint, HuggingFaceLMFusion
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create LM fusion module
lm_fusion = HuggingFaceLMFusion(
    model=model,
    tokenizer=tokenizer,
    weight=0.3,  # LM weight (tune this!)
    homophone_aggregation='max',  # or 'logsumexp'
    device=torch.device('cuda:0'),
)

# Create decoder with GPT-2 fusion
decoder = BatchedBeamCTCComputer(
    blank_index=0,
    beam_size=10,
    lexicon=lexicon,
    lm_fusion=lm_fusion,
    allow_cuda_graphs=False,
)

# Run beam search
result = decoder(logits, logits_lengths)
```

## Parameters

### LM Fusion Parameters

- **`weight`** (float, default=0.3): Controls LM influence
  - Range: 0.1-0.5 typical
  - Higher = more LM influence
  - Lower = more acoustic model influence
  - Start with 0.3 and tune based on validation WER

- **`homophone_aggregation`** (str, default='max'): How to combine homophone scores
  - `'max'`: Take best homophone (recommended for 1-best)
  - `'logsumexp'`: Bayesian average (better for N-best lists)

- **`device`** (torch.device): GPU/CPU for LM inference
  - Should match logits device
  - Default: auto-detect

### Decoder Parameters

- **`lm_fusion`** (NeuralLanguageModelFusion, optional): LM fusion module
  - None = no LM fusion (baseline)
  - DummyLMFusion = testing
  - HuggingFaceLMFusion = real GPT-2/LLaMA/etc

- **`allow_cuda_graphs`** (bool, default=True): Use CUDA graphs for speed
  - Set to False for debugging (allows breakpoints)
  - Set to True for production (2-3x faster)

## How It Works

1. **Sparse Fusion**: LM only applied at word boundaries (not every frame)
2. **Word Boundary Detection**: Uses lexicon trie to detect completed words
3. **Homophone Handling**: Scores all word interpretations (e.g., "aunt" vs "ant")
4. **Score Aggregation**: Combines homophone scores via max or logsumexp
5. **Before Pruning**: LM scores added before topk selection to guide search

## Performance Tips

### Tuning the LM Weight

```python
# Too low: LM has no effect
lm_fusion = HuggingFaceLMFusion(..., weight=0.05)  # ❌ Too weak

# Good range: Noticeable improvement
lm_fusion = HuggingFaceLMFusion(..., weight=0.3)   # ✅ Start here

# Too high: LM dominates, acoustic model ignored
lm_fusion = HuggingFaceLMFusion(..., weight=1.0)   # ❌ Too strong
```

**Tuning strategy:**
1. Start with weight=0.3
2. Run on validation set
3. Try [0.1, 0.2, 0.3, 0.4, 0.5]
4. Pick weight with lowest WER

### Speed Optimization

1. **Enable CUDA graphs** (production):
   ```python
   decoder = BatchedBeamCTCComputer(..., allow_cuda_graphs=True)
   ```

2. **Use smaller LM** for faster inference:
   ```python
   # Faster: distilgpt2 (82M params)
   model = AutoModelForCausalLM.from_pretrained("distilgpt2")
   
   # Slower but better: gpt2-medium (345M params)
   model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
   ```

3. **Reduce beam size** if too slow:
   ```python
   decoder = BatchedBeamCTCComputer(..., beam_size=5)  # Instead of 10
   ```

## Custom LM Integration

You can create your own LM fusion class:

```python
from brainaudio.inference.decoder import NeuralLanguageModelFusion

class MyCustomLMFusion(NeuralLanguageModelFusion):
    def __init__(self, my_model, weight=0.3):
        super().__init__(weight=weight)
        self.my_model = my_model
    
    def score_continuations(self, contexts, candidate_words):
        """
        Score candidate words given contexts.
        
        Args:
            contexts: List[str] - ["I saw my", "picnic with"]
            candidate_words: List[List[str]] - [["aunt", "ant"], ["aunt", "ant"]]
        
        Returns:
            List[List[float]] - log-probabilities for each candidate
        """
        all_scores = []
        for context, words in zip(contexts, candidate_words):
            word_scores = []
            for word in words:
                # Your custom scoring logic here
                full_text = f"{context} {word}"
                score = self.my_model.score(full_text)  # Your API
                word_scores.append(score)
            all_scores.append(word_scores)
        return all_scores

# Use it
lm_fusion = MyCustomLMFusion(my_model, weight=0.3)
decoder = BatchedBeamCTCComputer(..., lm_fusion=lm_fusion)
```

## Testing

Run the test script to verify installation and measure impact:

```bash
cd scripts
uv run test_neural_lm_fusion.py
```

This will:
1. Run baseline beam search (no LM)
2. Run with dummy LM (should match baseline)
3. Run with GPT-2 LM (if transformers installed)
4. Compare WERs and show improvement

## Troubleshooting

### "lm_fusion is enabled but lexicon is None"
- **Cause**: LM fusion requires lexicon for word boundary detection
- **Fix**: Always provide `lexicon` parameter when using `lm_fusion`

### LM has no effect on results
- **Cause**: Weight too low or LM scores are zero
- **Fix**: Increase weight, check LM is returning non-zero scores

### Out of memory
- **Cause**: Large LM + large beam size
- **Fix**: Use smaller LM (distilgpt2) or reduce beam size

### Very slow
- **Cause**: CUDA graphs disabled or large LM
- **Fix**: Enable CUDA graphs, use smaller/faster LM model

## Example Results

Expected WER improvements (approximate):

- **No LM**: 25% WER (baseline)
- **GPT-2**: 22% WER (~12% relative improvement)
- **GPT-2 Medium**: 21% WER (~16% relative improvement)
- **Domain-specific LM**: 18-20% WER (best)

Your mileage may vary depending on acoustic model quality and data domain.
