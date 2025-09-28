import os
import pickle
import time
from edit_distance import SequenceMatcher
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb

# brainaudio internal package imports
from utils.loss import forward_ctc 
from utils.loading_data import getDatasetLoaders

def trainModel(args, model):

    wandb.init(project="End to End", 
                entity="ebrahimfeghhi", config=dict(args), name=args['modelName'])
        
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"]
    )
    
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    if args['optimizer'] == 'AdamW':
        
         optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_decay'], betas=(args['beta1'], args['beta2']))
         
    if args['optimizer'] == 'Adam':
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["learning_rate"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        
    if args['learning_scheduler'] == 'multistep': 

        print("Multistep scheduler")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])
        
    elif args['learning_scheduler'] == 'cosine':
        
        print("Cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['n_epochs'],     # Total epochs to decay over
            eta_min=args['lrEnd']    # Final learning rate
        )
            
    elif args['learning_scheduler'] == "None":
    
        scheduler = None
    
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
    
    
    for epoch in range(args['n_epochs']):
        
        train_loss = []
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(trainLoader, desc="Training")):
           
            # Base case: always unpack the first 5
            X, y, X_len, y_len, dayIdx = batch[:5]

            # Send to device
            X      = X.to(args["device"])
            y      = y.to(args["device"])
            X_len  = X_len.to(args["device"])
            y_len  = y_len.to(args["device"])
            dayIdx = dayIdx.to(args["device"])


            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                    * args["constantOffsetSD"]
                )
                
            adjustedLens = model.compute_length(X_len)
          
            pred = model.forward(X, X_len, dayIdx)
            
            loss = forward_ctc(pred, adjustedLens, y, y_len)
                
            train_loss.append(loss.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
    
        with torch.no_grad():
    
            avgTrainLoss = np.mean(train_loss)
            
            model.eval()
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
  
            for batch in testLoader:
                
                X, y, X_len, y_len, testDayIdx = batch               

                # move to device (and y2 if present)
                X, y, X_len, y_len, testDayIdx = (
                    X.to(args["device"]),
                    y.to(args["device"]),
                    X_len.to(args["device"]),
                    y_len.to(args["device"]),
                    testDayIdx.to(args["device"]),
                )
                
                adjustedLens = model.compute_length(X_len)
            
                pred = model.forward(X, X_len, testDayIdx)            
                loss = forward_ctc(pred, adjustedLens, y, y_len)
                allLoss.append(loss.item())                               

                for iterIdx in range(pred.shape[0]):
                    decodedSeq = torch.argmax(pred[iterIdx, 0:adjustedLens[iterIdx], :], dim=-1) 
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])

                    trueSeq = np.array(y[iterIdx][0:y_len[iterIdx]].cpu().detach())

                    matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(trueSeq)

            avgDayLoss = np.mean(allLoss) if allLoss else 0.0
            cer = total_edit_distance / total_seq_length if total_seq_length > 0 else float('nan')

            endTime = time.time()
            print(
                f"Epoch {epoch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}"
                + f", time/batch: {(endTime - startTime)/100:>7.3f}"
            )
                
            # Log the metrics to wandb
            log_dict = {
                "train_ctc_Loss": avgTrainLoss,
                "ctc_loss": avgDayLoss,
                "cer": cer,
                "time_per_epoch": (endTime - startTime) / 100,
            }
            
            wandb.log(log_dict)

        if len(testCER) > 0 and cer < np.min(testCER):
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer")
            torch.save(scheduler.state_dict(), args['outputDir'] + '/scheduler')
            
        if len(testLoss) > 0 and avgDayLoss < np.min(testLoss):
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights_ctc")
            
                
        testLoss.append(avgDayLoss)
        testCER.append(cer)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)
            
        if scheduler is not None:
            scheduler.step()
                    
    wandb.finish()
    return 