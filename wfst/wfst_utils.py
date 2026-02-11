import os
import numpy as np
import lm_decoder
from functools import lru_cache

def build_lm_decoder(model_path,
                     max_active=7000,
                     min_active=200,
                     beam=17.,
                     lattice_beam=8.,
                     acoustic_scale=1.5,
                     ctc_blank_skip_threshold=1.0,
                     length_penalty=0.0,
                     nbest=1):
    
    
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest
    )

    TLG_path = os.path.join(model_path, 'TLG.fst')
    words_path = os.path.join(model_path, 'words.txt')
    G_path = os.path.join(model_path, 'G.fst')
    rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')
    if not os.path.exists(rescore_G_path):
        rescore_G_path = ""
        G_path = ""
    if not os.path.exists(TLG_path):
        raise ValueError('TLG file not found at {}'.format(TLG_path))
    if not os.path.exists(words_path):
        raise ValueError('words file not found at {}'.format(words_path))

    decode_resource = lm_decoder.DecodeResource(
        TLG_path,
        G_path,
        rescore_G_path,
        words_path,
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

    return decoder

def lm_decode(decoder, logits, returnNBest=False, rescore=False,
              blankPenalty=0.0,
              logPriors=None):
    
    assert len(logits.shape) == 2

    if logPriors is None:
        
        logPriors = np.zeros([1, logits.shape[1]])
        #logPriors = np.zeros_like(logits)
        
    lm_decoder.DecodeNumpy(decoder, logits, logPriors, blankPenalty)
    
    decoder.FinishDecoding()
    if rescore:
        decoder.Rescore()
        
    if not returnNBest:
        if len(decoder.result()) == 0:
            decoded = ''
        else:
            decoded = decoder.result()[0].sentence
    else:
        decoded = []
        for r in decoder.result():
            decoded.append((r.sentence, r.ac_score, r.lm_score))
    
    decoder.Reset()

    return decoded

def arrange_logits(logits):
    
    """
    Rearranges and reshapes logits for lm decoding
    
    Args:
        logits (ndarray): num_timesteps x num_logits array
    """
    
    # move the blank token to the last logit dimension 
    logits = np.concatenate([logits[:, 1:], logits[:, 0:1]], axis=-1)
    
    # add extra dimension 
    logits = logits[None, :, :]
    
    # order logits so that blank -> silence -> phonemes
    logits = np.concatenate([logits[:, :, -1:], logits[:, :, -2:-1], logits[:, :, :-2]], axis=-1)
    
    return logits


    
def compute_wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def _cer_and_wer(decodedSentences, trueSentences, outputType='handwriting',
                 returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = compute_wer([c for c in trueSent], [c for c in decSent])
        if outputType == 'handwriting':
            trueWords = trueSent.replace(">", " > ").split(" ")
            decWords = decSent.replace(">", " > ").split(" ")
        elif outputType == 'speech' or outputType == 'speech_sil':
            trueWords = trueSent.split(" ")
            decWords = decSent.split(" ")
        nWordErr = compute_wer(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)

    if not returnCI:
        return cer, wer
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledCER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])
        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledCER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
        cerCI = np.percentile(resampledCER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        return (cer, cerCI[0], cerCI[1]), (wer, werCI[0], werCI[1])


# function to augment the nbest list by swapping words around, artificially increasing the number of candidates
def augment_nbest(nbest, top_candidates_to_augment=20, acoustic_scale=0.3, score_penalty_percent=0.01):

    sentences = []
    ac_scores = []
    lm_scores = []
    total_scores = []

    for i in range(len(nbest)):
        sentences.append(nbest[i][0].strip())
        ac_scores.append(nbest[i][1])
        lm_scores.append(nbest[i][2])
        total_scores.append(acoustic_scale*nbest[i][1] + nbest[i][2])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # new sentences and scores
    new_sentences = []
    new_ac_scores = []
    new_lm_scores = []
    new_total_scores = []

    # swap words around
    for i1 in range(np.min([len(sentences)-1, top_candidates_to_augment])):
        words1 = sentences[i1].split()

        for i2 in range(i1+1, np.min([len(sentences), top_candidates_to_augment])):
            words2 = sentences[i2].split()

            if len(words1) != len(words2):
                continue
            
            _, path1, _ = get_string_differences(sentences[i1], sentences[i2])
            _, path2, _ = get_string_differences(sentences[i2], sentences[i1])

            replace_indices1 = [i for i, p in enumerate(path2) if p == 'R']
            replace_indices2 = [i for i, p in enumerate(path1) if p == 'R']

            for r1, r2 in zip(replace_indices1, replace_indices2):
                
                new_words1 = words1.copy()
                new_words2 = words2.copy()

                new_words1[r1] = words2[r2]
                new_words2[r2] = words1[r1]

                new_sentence1 = ' '.join(new_words1)
                new_sentence2 = ' '.join(new_words2)

                if new_sentence1 not in sentences and new_sentence1 not in new_sentences:
                    new_sentences.append(new_sentence1)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

                if new_sentence2 not in sentences and new_sentence2 not in new_sentences:
                    new_sentences.append(new_sentence2)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

    # combine new sentences and scores with old
    for i in range(len(new_sentences)):
        sentences.append(new_sentences[i])
        ac_scores.append(new_ac_scores[i])
        lm_scores.append(new_lm_scores[i])
        total_scores.append(new_total_scores[i])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # return nbest
    nbest_out = []
    for i in range(len(sentences)):
        nbest_out.append([sentences[i], ac_scores[i], lm_scores[i]])

    return nbest_out



# function to get string differences between two sentences
def get_string_differences(cue, decoder_output):
        decoder_output_words = decoder_output.split()
        cue_words = cue.split()

        @lru_cache(None)
        def reverse_w_backtrace(i, j):
            if i == 0:
                return j, ['I'] * j
            elif j == 0:
                return i, ['D'] * i
            elif i > 0 and j > 0 and decoder_output_words[i-1] == cue_words[j-1]:
                cost, path = reverse_w_backtrace(i-1, j-1)
                return cost, path + [i - 1]
            else:
                insertion_cost, insertion_path = reverse_w_backtrace(i, j-1)
                deletion_cost, deletion_path = reverse_w_backtrace(i-1, j)
                substitution_cost, substitution_path = reverse_w_backtrace(i-1, j-1)
                if insertion_cost <= deletion_cost and insertion_cost <= substitution_cost:
                    return insertion_cost + 1, insertion_path + ['I']
                elif deletion_cost <= insertion_cost and deletion_cost <= substitution_cost:
                    return deletion_cost + 1, deletion_path + ['D']
                else:
                    return substitution_cost + 1, substitution_path + ['R']

        cost, path = reverse_w_backtrace(len(decoder_output_words), len(cue_words))

        # remove insertions from path
        path = [p for p in path if p != 'I']

        # Get the indices in decoder_output of the words that are different from cue
        indices_to_highlight = []
        current_index = 0
        for label, word in zip(path, decoder_output_words):
            if label in ['R','D']:
                indices_to_highlight.append((current_index, current_index+len(word)))
            current_index += len(word) + 1

        return cost, path, indices_to_highlight
