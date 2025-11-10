import scipy
import numpy as np
import re
import numpy as np
from g2p_en import G2p
import numpy as np
import os
import pickle

# Adopted from https://github.com/cffan/neural_seq_decoder/notebooks/formatCompetitionData


# ----------------------- USER-SPECIFIC INFORMATION -----------------------
dataDir = "/data2/brain2text/b2t_24/competitionData/" # directory where data is stored at 
output_type = "phonemes"
dataSave = "/data2/brain2text/b2t_24/brain2text24.pkl" # directory where processed data is saved
logBoth = False # if True, log both spike band power and tx crossings before normalization
# -------------------------------------------------------------------------

import nltk
nltk.download('averaged_perceptron_tagger_eng')

g2p = G2p()
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']

def phoneToId(p):
    return PHONE_DEF_SIL.index(p)

CHAR_VOCAB = [
    "<sp>",          # space token
    "!", ",", ".", "?", "'",   # punctuation (incl. apostrophe)
] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 'a'..'z'

# Build mappings
_CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
_ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}

# Convenience indices
SPACE_ID = _CHAR_TO_ID["<sp>"]

def charToId(c: str) -> int:
    """Map raw input char to ID, normalizing space and lowercase."""
    if c == " ":
        c = "<sp>"
    c = c.lower()
    return _CHAR_TO_ID[c]

def idToChar(i: int) -> str:
    return _ID_TO_CHAR[i]

def loadFeaturesAndNormalize(sessionPath, logBoth):
    
    '''
    
    This function 
    
    args: 
        list sessionPath: 
        bool logBoth: 
        
    returns: 
        data for that session, normalized within each block. 
    '''
    
    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = dat['sentenceText'].shape[0]

    #collect area 6v tx1 and spikePow features
    for i in range(n_trials):    
        #get time series of TX and spike power for this trial
        #first 128 columns = area 6v only
        if logBoth:
            tx_crossings = np.log1p(dat['tx1'][0, i][:, 0:128])
            log_pow = np.log(dat['spikePow'][0,i][:,0:128])
            features = np.concatenate([tx_crossings, log_pow], axis=1)
        else:
            features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    #block-wise feature normalization
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    
    for b in range(len(blockList)):
        
        sentIdx = np.argwhere(blockNums==blockList[b])
        sentIdx = sentIdx[:,0].astype(np.int32)
        blocks.append(sentIdx)

    
    for b in range(len(blocks)):
        
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    #convert to tfRecord file
    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens, 
        'blockNums': blockNums
    }

    return session_data


def getDataset(fileName, logBoth, output_type):
    
    session_data = loadFeaturesAndNormalize(fileName, logBoth)

    allDat = []
    trueSentences = []
    seq = []

    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x])
        trueSentences.append(session_data['transcriptions'][x])

        # clean sentence
        thisTranscription = str(session_data['transcriptions'][x]).strip()
        thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
        thisTranscription = thisTranscription.replace('--', '').lower()
        addInterWordSymbol = True
        maxSeqLen = 500

        # ----- build phoneme ids (used for 'phoneme' or 'both') -----
        if output_type == "phonemes":
            phonemes = []
            for p in g2p(thisTranscription):
                if addInterWordSymbol and p == ' ':
                    phonemes.append('SIL')
                p = re.sub(r'[0-9]', '', p)           # remove stress
                if re.match(r'[A-Z]+', p):            # keep only phoneme labels
                    phonemes.append(p)
            if addInterWordSymbol:
                phonemes.append('SIL')

            seqLenP = len(phonemes)
            seqIDsP = np.zeros([maxSeqLen], dtype=np.int32)
            seqIDsP[:seqLenP] = [phoneToId(p) + 1 for p in phonemes]

            seq.append(seqIDsP)         
            
        elif output_type == "char":
            
            txt = thisTranscription.replace('-', ' ')
            characters = []
            for c in txt:
                if addInterWordSymbol and c == ' ':
                    characters.append('<sp>')
                else:
                    characters.append(c)
            if addInterWordSymbol:
                characters.append('<sp>')

            seqLenC = len(characters)
            seqIDsC = np.zeros([maxSeqLen], dtype=np.int32)
            seqIDsC[:seqLenC] = [charToId(c) + 1 for c in characters]
            seq.append(seqIDsC)         
            
    # ----- package dataset -----
    newDataset = {
        'sentenceDat': allDat,
        'transcriptions': trueSentences,
        'text': seq,                     # primary (chars for 'both')
        'blockNums': session_data['blockNums']
    }

    # lengths for primary text
    timeSeriesLens = [arr.shape[0] for arr in newDataset['sentenceDat']]
    textLens = []
    for arr in newDataset['text']:
        zeroIdx = np.argwhere(arr == 0)
        textLens.append(zeroIdx[0, 0] if zeroIdx.size else arr.shape[0])

    newDataset['timeSeriesLens'] = np.array(timeSeriesLens)
    newDataset['textLens'] = np.array(textLens)
    newDataset['textPerTime'] = newDataset['textLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)

    return newDataset


trainDatasets = []
testDatasets = []
competitionDatasets = []

sessionNames = ['t12.2022.04.28',  't12.2022.05.26',  't12.2022.06.21',  't12.2022.07.21',  't12.2022.08.13',
't12.2022.05.05',  't12.2022.06.02',  't12.2022.06.23',  't12.2022.07.27',  't12.2022.08.18',
't12.2022.05.17',  't12.2022.06.07',  't12.2022.06.28',  't12.2022.07.29',  't12.2022.08.23',
't12.2022.05.19',  't12.2022.06.14',  't12.2022.07.05',  't12.2022.08.02',  't12.2022.08.25',
't12.2022.05.24',  't12.2022.06.16',  't12.2022.07.14',  't12.2022.08.11']
sessionNames.sort()

for dayIdx in range(len(sessionNames)):
    trainDataset = getDataset(dataDir + '/train/' + sessionNames[dayIdx] + '.mat', logBoth, output_type)
    testDataset = getDataset(dataDir + '/test/' + sessionNames[dayIdx] + '.mat', logBoth, output_type)

    trainDatasets.append(trainDataset)
    testDatasets.append(testDataset)

    if os.path.exists(dataDir + '/competitionHoldOut/' + sessionNames[dayIdx] + '.mat'):
        dataset = getDataset(dataDir + '/competitionHoldOut/' + sessionNames[dayIdx] + '.mat', logBoth, output_type)
        competitionDatasets.append(dataset)
        
allDatasets = {}
allDatasets['train'] = trainDatasets
allDatasets['val'] = testDatasets
allDatasets['test'] = competitionDatasets


with open(dataSave, 'wb') as handle:
    pickle.dump(allDatasets, handle)