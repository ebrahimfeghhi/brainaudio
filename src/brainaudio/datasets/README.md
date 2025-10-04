This folder contains scripts for converting datasets to a general format.

We currently have scripts for the Brain to Text '24 and Brain to Text '25 datasets.
These scripts are adopted from their original repositories, which are referenced in the respective scripts. 

***General Format***

- Data is stored as a hierarchical dictionary.
- Top-level keys: 'train', 'val', 'test'
- Each top-level key is a list of dictionaries containing day-specific data. 
    - Day-specific dictionaries should be ordered in time (i.e. Day 0 should be first element).
    - Each day-specific dictionary contains the following lower-level keys.
        - *sentenceDat:* list, where each element is a 2D numpy array of shape T (time points) $\times$ N (features).
        - *transcriptions:* list, where each element is a string containing the ground truth sentence. For the test set, this is some filler string.
        - *text:* list, each element is a M dimensional array, which has phoneme integers and then zero padded.
        - *timeSeriesLen:* list, length of neural data trial
        - *textLens:* list, length of phoneme sequence per trial

- For days that lack data, there will be a $None$ Type as placeholder
