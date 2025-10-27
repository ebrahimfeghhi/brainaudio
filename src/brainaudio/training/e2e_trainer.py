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
from brainaudio.training.utils.loss import forward_ctc, evaluate_e2e
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.learning_scheduler import create_learning_rate_scheduler
from brainaudio.training.utils.loss import forward_ctc, evaluate_e2e
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.learning_scheduler import create_learning_rate_scheduler

def trainE2EModel(args, model):

    wandb.init(project=args["wandb"]["project"], 
                entity=args["wandb"]["entity"], config=dict(args), name=args['modelName'])
        
    outputDir = f'{args["outputDir"]}{args["modelName"]}'
    
    os.makedirs(outputDir, exist_ok=False)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(outputDir + "/args", "wb") as file:
        pickle.dump(args, file)
    
    trainLoaders, valLoaders, testLoaders, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"], 
        return_transcript=True,
        return_alignments=True
    )
    
    # Watch the model
    ctc_weight = args["ctc_weight"]
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_decay'], eps=args['eps'], 
                               betas=(args['beta1'], args['beta2']), fused=True)
    scheduler = create_learning_rate_scheduler(args, optimizer)
        
    max_dataset_train_length = max(len(loader) for loader in trainLoaders)

    test_loss = []
    test_wer = []
    startTime = time.time()
    
    
    for epoch in range(args['n_epochs']):
        
        # Metric Logger
        trainLoss = []
        ctcLoss = []
        ceLoss = []
        grad_norm_store=[]

        model.train()

        train_loop = tqdm(
            zip_longest(*trainLoaders), 
            total=max_dataset_train_length, 
            desc=f"Training Epoch {epoch+1}/{args['n_epochs']}"
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
                                    
                # Base case: always unpack the first 6
                X, y, X_len, y_len, dayIdx, _, forced_alignments = batch[:6]

                # Send to device
                X      = X.to(args["device"])
                y      = y.to(args["device"])
                X_len  = X_len.to(args["device"])
                y_len  = y_len.to(args["device"])
                dayIdx = dayIdx.to(args["device"])
                
                with torch.autocast(device_type = args["device"], enabled = args['use_amp'], dtype = torch.bfloat16):
                    
                    # ----- Data Preprocessing -----
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
                    
                    # ----- Model Forward Pass -----

                    adjustedLens = model.encoder.compute_length(X_len)

                    #! Replace with Dynamic chunking logic here!
                    chunk_size=args.get('chunk_size', 0)
                    llm_context_chunks=args.get('llm_context_chunks', 1)


                    llm_outs, logits = model(
                        neuralInput=X,
                        Input_len=X_len,
                        adjusted_lens=adjustedLens,
                        forced_alignments=forced_alignments,
                        chunk_size=chunk_size,
                        llm_context_chunks=llm_context_chunks,
                        participant_idx=participant_id
                    )
                    # --- Hybrid Loss Calculation ---
                    ctc_loss = forward_ctc(logits, adjustedLens, y, y_len)
                    # The CE loss is conveniently calculated inside the LLM
                    ce_loss = llm_outs.loss
                    # Combine the two losses with the weighting factor
                    loss = (1 - ctc_weight) * ce_loss + ctc_weight * ctc_loss
                    ctcLoss.append(ctc_loss.cpu().detach().numpy())
                    ceLoss.append(ce_loss.cpu().detach().numpy())
                    trainLoss.append(loss.cpu().detach().numpy())  
                
                loss.backward()
                if args.get('grad_norm_clip_value', -1) > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['grad_norm_clip_value'])
                    grad_norm_store.append(total_norm.item())
                
                optimizer.step()            
                scheduler.step()
        

        avgDayLoss_array = []
        avgDayctcLoss_array = []
        avgDayceLoss_array = []
        wers = []

        # Preparing metrics
        current_lr = optimizer.param_groups[0]['lr']
        avgTrainLoss = np.mean(trainLoss)
        avgCTC = np.mean(ctcLoss)
        avgCE = np.mean(ceLoss)
        endTime = time.time()

        for participant_id, valLoader in enumerate(valLoaders):
            test_losses, wer = evaluate_e2e(valLoader, model, participant_id, forward_ctc, args)
            test_loss, test_ctc, test_ce = test_losses
            avgDayLoss_array.append(test_loss)
            avgDayctcLoss_array.append(test_ctc)
            avgDayceLoss_array.append(test_ce)
            wers.append(wer)

        # Log the metrics to wandb
        log_dict = {
            "train_loss": avgTrainLoss,
            "train_ctc_loss": avgCTC,
            "train_ce_loss": avgCE,
            "learning_rate": current_lr, 
            "grad_norm": np.mean(grad_norm_store), 
            "time_per_epoch": (endTime - startTime) / 100
        }

        if len(avgDayLoss_array) > 0:
            for pid, (test_loss, test_ctc_loss, test_ce_loss, wer) in enumerate(zip(avgDayLoss_array, avgDayctcLoss_array, avgDayceLoss_array, wers)):
                log_dict[f"test_loss_{pid}"] = test_loss
                log_dict[f"test_ctc_loss{pid}"] = test_ctc_loss
                log_dict[f"test_ce_loss_{pid}"] = test_ce_loss
                log_dict[f"WER_{pid}"] = wer

        wandb.log(log_dict)

        if len(test_wer) > 0 and np.mean(wers) < np.min(test_wer):
            torch.save(model.state_dict(), outputDir + "/modelWeights")
            torch.save(optimizer.state_dict(), outputDir + "/optimizer")
            torch.save(scheduler.state_dict(), outputDir + '/scheduler')
            save_path = os.path.join(outputDir, "best_model_adapters")
            model.save_pretrained(save_path)
            
        if len(test_loss) > 0 and np.mean(avgDayLoss_array) < np.min(test_loss):
            torch.save(model.state_dict(), outputDir + "/modelWeights_ctc")


        test_loss.append(np.mean(avgDayLoss_array))
        test_wer.append(np.mean(wers))

        test_stats = {}
        test_stats["test_loss"] = np.array()
        test_stats["test_wer"] = np.array()

        with open(outputDir + "/trainingStats", "wb") as file:
          pickle.dump(test_stats, file)

        
    wandb.finish()
    return