import os
import pickle
import time
from edit_distance import SequenceMatcher
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from itertools import zip_longest

# brainaudio internal package imports
from .utils.augmentations import gauss_smooth
from .utils.loss import forward_ctc, evaluate
from ..datasets.loading_data import getDatasetLoaders
from .utils.learning_scheduler import create_learning_rate_scheduler

def trainModel(args, model):

    wandb.init(project="nejm-brain-to-text", 
                entity="lionelhu926-ucla", config=dict(args), name=args['modelName'])
        
    outputDir = f'{args["outputDir"]}{args["modelName"]}'
    
    os.makedirs(outputDir, exist_ok=False)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(outputDir + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoaders, valLoaders, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"]
    )
    
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms
    

    if args['optimizer'] == 'AdamW':
        
         optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_decay'], eps=args['eps'], 
                                       betas=(args['beta1'], args['beta2']), fused=True)
         
    if args['optimizer'] == 'Adam':
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["learning_rate"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        
    scheduler = create_learning_rate_scheduler(args, optimizer)
            
    if len(args['load_pretrained_model']) > 0:
        
        optimizer_path = os.path.join(args['load_pretrained_model'], 'optimizer')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=args['device']))
        
        scheduler_path = os.path.join(args['load_pretrained_model'], 'scheduler')
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=args['device']))
        print(f"Loaded optimizer and scheduler state from {args['load_pretrained_model']}")
        
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    train_loss = []
        
    max_dataset_train_length = max(len(loader) for loader in trainLoaders)
    
    for epoch in range(args['n_epochs']):
        
        train_loss = []
        grad_norm_store = []
        model.train()

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
                        
                optimizer.zero_grad()    
                                    
                # Base case: always unpack the first 5
                X, y, X_len, y_len, dayIdx = batch[:5]

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
                    
                    adjustedLens = model.compute_length(X_len)
                
                    logits, final_layer = model.forward(X, X_len, participant_id, dayIdx)
                    
                    # final_layer is of shape B x T x D, where D is the model hidden dim
                    
            