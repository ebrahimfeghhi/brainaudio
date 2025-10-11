from tqdm import tqdm
from itertools import zip_longest
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.augmentations import gauss_smooth
import torch
import numpy as np
import torch.nn as nn
from brainaudio.models.e2e import set_up_neural_encoder

def trainE2EModel(args, model):

    
    trainLoaders, valLoaders, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"], 
        return_alignments=True
    )
    
    # Watch the model
        
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_decay'], eps=args['eps'], 
    #                            betas=(args['beta1'], args['beta2']), fused=True)

        
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
                breakpoint()

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
                    
                    
                    
                    #adjustedLens = model.compute_length(X_len)
                    
                    encoder = model
                    logits, final_layer = encoder.forward(X, X_len, forced_alignments, participant_id, dayIdx)
                    linear_layer = nn.Linear(len(final_layer), 640)
                    projected_outs = final_layer
                    
                    
args = {'device': 'cuda:2', 'smooth_kernel_size': 100, 'gaussianSmoothWidth': 2, 
        'random_cut': 0, 'constantOffsetSD': 0.05, 'whiteNoiseSD': 0.2, 'n_epochs': 1, 'beta1': 0.9, 
        'beta2': 0.999, 'l2_decay': 1e-5, 'learning_rate': 0.005, 'eps': 1e-8, 'modelName': 'None', 
        'outputDir': None, 'batchSize': 64, 'datasetPath': ["/data2/brain2text/b2t_25/brain2text25_with_fa"]}

test_model , _, _ = set_up_neural_encoder()
trainE2EModel(args=args, model=test_model)