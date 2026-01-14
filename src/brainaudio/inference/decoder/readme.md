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



