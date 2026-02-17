#!/bin/bash
# Script to generate corrector LLM finetuning dataset for b2t_24 and b2t_25

# Configuration
BEAM_SIZE=300
TOP_N=100
DEVICE="cuda:0"

# Paths (adjust these to your actual paths)
TOKENS="/home/ebrahim/data2/brain2text/lm/units_pytorch.txt"
LEXICON="/home/ebrahim/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt"
WORD_LM="/home/ebrahim/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm"

# b2t_24 dataset
echo "=== Generating dataset for b2t_24 ==="
python finetune_llm/generate_data_lift.py \
  --logits-path /data2/brain2text/b2t_24/logits_train.npz \
  --transcripts-path /data2/brain2text/b2t_24/transcripts_train.pkl \
  --output-path /home/ebrahim/brainaudio/finetune_llm/data/b2t_24_corrector_train.jsonl \
  --tokens "$TOKENS" \
  --lexicon "$LEXICON" \
  --word-lm-path "$WORD_LM" \
  --beam-size $BEAM_SIZE \
  --top-n $TOP_N \
  --device $DEVICE

# b2t_25 dataset
echo ""
echo "=== Generating dataset for b2t_25 ==="
python finetune_llm/generate_data_lift.py \
  --logits-path /home/ebrahim/data2/brain2text/b2t_25/logits_train.npz \
  --transcripts-path /home/ebrahim/data2/brain2text/b2t_25/transcripts_train.pkl \
  --output-path /home/ebrahim/brainaudio/finetune_llm/data/b2t_25_corrector_train.jsonl \
  --tokens "$TOKENS" \
  --lexicon "$LEXICON" \
  --word-lm-path "$WORD_LM" \
  --beam-size $BEAM_SIZE \
  --top-n $TOP_N \
  --device $DEVICE

echo ""
echo "=== Combining datasets ==="
cat /home/ebrahim/brainaudio/finetune_llm/data/b2t_24_corrector_train.jsonl \
    /home/ebrahim/brainaudio/finetune_llm/data/b2t_25_corrector_train.jsonl \
    > /home/ebrahim/brainaudio/finetune_llm/data/b2t_24_25_corrector_train.jsonl

echo "Dataset generation complete!"
echo "Combined dataset: /home/ebrahim/brainaudio/finetune_llm/data/b2t_24_25_corrector_train.jsonl"
