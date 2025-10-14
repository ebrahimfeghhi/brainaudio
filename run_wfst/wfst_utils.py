import os
import numpy as np
import lm_decoder


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