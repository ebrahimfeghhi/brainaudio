import os
import pickle
import time
from edit_distance import SequenceMatcher
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from itertools import zip_longest

# brainaudio internal package imports
from brainaudio.training.utils.augmentations import gauss_smooth
from brainaudio.training.utils.loss import forward_ctc, evaluate
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.learning_scheduler import create_learning_rate_scheduler

def trainE2EModel(args, model):

    
    trainLoaders, valLoaders, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"], 
        return_alignments=True
    )
    
    # Watch the model
        
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_decay'], eps=args['eps'], 
    #                            betas=(args['beta1'], args['beta2']), fused=True)
    # scheduler = create_learning_rate_scheduler(args, optimizer)
        
    max_dataset_train_length = max(len(loader) for loader in trainLoaders)
    
    for epoch in range(args['n_epochs']):
        
        #model.train()

        train_loop = tqdm(
            zip_longest(*trainLoaders), 
            total=max_dataset_train_length, 
            desc="Training Epoch"
        )
        
        # batches is a list containing batched data for each participant
        for batch_idx, batches in enumerate(train_loop):
            
            for participant_id, batch in enumerate(batches):
                
                '''
                not all participants have the same number of batches
                zip_longest will return None once data runs out for a participant
                '''
                if batch is None:
                    continue
                        
                #optimizer.zero_grad()    
                                    
                # Base case: always unpack the first 5
                X, y, X_len, y_len, dayIdx, forced_alignments = batch[:6]

                # Send to device
                X      = X.to(args["device"])
                y      = y.to(args["device"])
                X_len  = X_len.to(args["device"])
                y_len  = y_len.to(args["device"])
                dayIdx = dayIdx.to(args["device"])
                
                with torch.autocast(device_type = args["device"], enabled = args['use_amp'], dtype = torch.bfloat16):

                    if args["whiteNoiseSD"] > 0:
                        X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

                    if args["constantOffsetSD"] > 0:
                        X += (
                            torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                            * args["constantOffsetSD"]
                        )
                        
                    # added in random cut from B2T '25
                    if args["random_cut"] > 0: 
                        cut = np.random.randint(0, args['random_cut'])
                        X = X[:, cut:, :]
                        X_len -= cut 
                        
                    X = gauss_smooth(inputs=X, device=args['device'], smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
                    
                    adjustedLens = model.encoder.compute_length(X_len)
                    
                    llm_outs, logits = model(X, X_len, None, forced_alignments, adjustedLens, participant_idx=0)


                    breakpoint()
                    # loss calculation (ctc + llm)
                    ctc_loss = forward_ctc(pred, adjustedLens, y, y_len)
                    ce_loss = llm_outs.loss

                    loss = ctc_loss + ce_loss
                    
                    optimizer.step()            
                    scheduler.step()
                    
        
                    