import kenlm
model = kenlm.Model("/home/ebrahim/brainaudio/creating_n_gram_lm/huge_pruned_10gram.arpa")

# Check the bigram probability of "I I"
# KenLM expects a space-separated string of tokens
score = model.score("AY AY", bos=True, eos=False)
print(f"Log probability of 'I I': {score}")

