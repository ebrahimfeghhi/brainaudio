## Batched Beam Search Torch
1. Loop through each frame 
2. Generate active mask to mark active batches.
3. Mark all beams which have emitted a repeated or CTC blank token.
4. Expand the log probs at this frame to be of shape beam size. 
5. Add the beam scores to every token for that beam. Since we are operating in log space, addition is equivalent to multiplication of the probabilities.
6. Add a bonus for beams that extend the sequence (non-blank or non-repeated).
7. Mask out tokens that cannot form valid word sequences with a -INF score. 
8. Get the top K beams.
9. Obtain the indices of the beams being extended and the new token indices. 
10. Prune beams that fall below a certain threshold from the top beam. 
11. Generate an inactive beam mask, use it to update next labels and next indices.
12. Obtain the previous last labels and next labels for beams. 
13. Update the lexicon state.
14. Update beams with the add_results function.
15. Apply word level N-gram score.
16. Recombine equivalent beams.
17. Apply LLM rescoring. 
18. After frame-level loop has finished, apply EOS scoring for both word level LM and N-Gram LM. 


## Add results
1. Updates transcript_wb and transcript_wb_prev_ptr using next_labels and next_indices, respectively.
2. Updates context texts. 
3. Increase current lengths nb if the bneam was extended with a non blank, non-repeating label.
4. Increment current lengths wb. 
5. Update transcript hash for beams that were extended with a label.
6. Update last label with next labels if is_extended is true. This means last label is the actual last label emitted by the model, even if it's a blank or repeated token.

## Word Level N-gram LM

### WordHistory
1. Creates a words and parents list. ALso create a deduplication cache.
2. Get_text method: Walks backwards similar to how we reconstruct phoneme sequences. Basically the word list keeps track of the current word. We output the most recent word, then use the parent list to find the index for the previous word. 
3. Add method: Appends new word and parent index. Avoids duplication by checking to see if this path already exists (in other words, was the same word already added with the same parent history).


### FastNGramLM
1. Takes as input KenLM model path, alpha, beta, unk_score_offset, and score_boundary in init. Keeps track of LM states and transitions. 
2. Init start state method initializes the root state, which the n-gram LM start token.
3. The is_unk method creates a running store of known and unknown words.
4. The get_eos_score method returns the score of the EOS token.

### Apply Word Ngram LM Scoring
1. Compute indices for all beams that have emitted a first boundary token (i.e. are ready to be scored). Loop through each beam individually. 
2. Use parent state to get the possible word indices formed by that phoneme sequence. Need to use the parent state because state resets back to root node when a word is emitted.
3. Retrieve words from lexicon.
4. Obtain context text tuples for that beam. Context text is a tuple which stores the LM score, the N-gram LM state, and the Word History State. 
5. Loop through each candidate word. 
6. Define a cache, as LM state and currently selected word.
7. Check lm_cache to see if word_score and child LM state have already been computed.
8. If not, obtain the parent state using the ID. Then, score the candidate word and generate a new state with the updated candidate word. 
9. Compute word score by multiplying by alpha_ngram and adding word insertion bonus.
10. Add the new state into lm_states, save the index in lm_states this state is in and the word score in the lm_cache. lm_states and lm_cache are maintained by the FastNGramLm object, which are initialized at the beginning of the trial (double check).
11. Update word history and total score, then append into all candidates list. 
12. Sort all candidates by total score, and select the top K, where K is the number of homophone beams allowed.
13. Prune beams that fall below homophone prune threshold from best beam. 
14. Update beam score by subtracting the old best LM score, and adding the new best LM score. 
15. Update context texts with the best new context text tuples. 


### Apply word ngram EOS scoring
1. Adds n-gram LM end token score to each beam. 


## Recombine hyps
1. Determine beam equivalence using the transcript history, the last predicted label, and current lengths nb. Last predicted label is necessary because two beams with equivalent transcript histories after CTC merging rules can't be merged if one ends in a blank and the other doesn't. Current lengths nb seems to be added in cases hashes collide (double check).
2. For all equivalent beams, remove lower scoring one or add the probabilities of equivalent beams together. 




## Apply Neural LM Rescoring


### LLMRescoring 
1. Keeps track of model, tokenizer, device, alpha, beta, scoring_chunk_size, llm_call_count.

### Score_texts_batch
1. Receives as input model, tokenizer, and texts.
2. Sorts text by length to minimize padding.
3. Loops through each chunk of text (default = 256).
4. Tokenizes the text.
5. Obtains logits for text.
6. Compute cross entropy. 


### Apply LLM Rescoring Full
1. Takes as input LLM rescorer instance, word history, and beam hyps object.
2. Creates a dictionary where keys are unique text attributes, and values are the batch idx, beam idx, homophone beam idx, and number of words. 
3. Pass unique texts to score texts batch to get log probs for each text. 
4. Fill context texts back in using the dictionary values.


### Apply LLM Rescoring End
1. Computes the probability of the text sequence given a period, ?, or ! for every single context text. This is probably a bit overkill. 
2. 