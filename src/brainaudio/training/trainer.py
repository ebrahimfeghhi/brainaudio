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
from torchaudio.models.decoder import ctc_decoder

# brainaudio internal package imports
from brainaudio.training.utils.augmentations import gauss_smooth
from brainaudio.training.utils.loss import forward_ctc, evaluate
from brainaudio.datasets.lazy_data_loading import getDatasetLoaders
from brainaudio.training.utils.learning_scheduler import create_learning_rate_scheduler

def trainModel(args, model):

    wandb.init(project=args["wandb"]["project"], 
                entity=args["wandb"]["entity"], config=dict(args), name=args['modelName'])
        
    outputDir = f'{args["outputDir"]}{args["modelName"]}'
    
    os.makedirs(outputDir, exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(outputDir + "/args", "wb") as file:
        pickle.dump(args, file)
    
    trainLoaders, valLoaders, _ = getDatasetLoaders(
        args["manifest_paths"],
        args["batchSize"],
        return_transcript=True
    )
    
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    def get_participant_suffix(pid: int) -> str:
        """Return a suffix for saving/logging per-participant artifacts."""
        suffix_cfg = args.get("participant_suffixes")

        if isinstance(suffix_cfg, dict):
            # YAML may convert int keys to strings, so check both
            if str(pid) in suffix_cfg:
                return suffix_cfg[str(pid)]
            if pid in suffix_cfg:
                return suffix_cfg[pid]
        elif isinstance(suffix_cfg, (list, tuple)):
            if 0 <= pid < len(suffix_cfg):
                return suffix_cfg[pid]
        elif suffix_cfg is not None:
            raise ValueError("participant_suffixes must be a list, tuple, or dict")

        raise ValueError(
            "participant_suffixes must provide an entry for each participant. "
            f"Missing suffix for participant id {pid}."
        )
    

    if args['optimizer'] == 'AdamW':
        
         optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], 
                                       weight_decay=args['l2_decay'], eps=args['eps'], 
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
    valLoss = []
    valWER = []
    valPER = []
    
    # Track best metrics by participant
    best_wer_by_participant = {}
    best_per_by_participant = {}
    
    startTime = time.time()
    train_loss = []

    
    max_dataset_train_length = max(len(loader) for loader in trainLoaders)
    
    # LM definition and initialization
    language_model_path = "/data2/brain2text/lm/"
    units_txt_file_pytorch = f"{language_model_path}units_pytorch.txt"
    imagineville_vocab_phoneme = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
    decoder = ctc_decoder(tokens=units_txt_file_pytorch, lexicon=imagineville_vocab_phoneme, 
                    beam_size=args["beam_size"], nbest=1, lm="/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm", 
                    lm_weight=args["lm_weight"], word_score=args["word_score"]) if args["evaluate_wer"] else None
    
    no_improvement_count = 0
    
    for epoch in range(args['n_epochs']):
        
        if no_improvement_count >= args["early_stopping_no_improvement"]: 
            wandb.log({"early_stopping_triggered": True, "early_stopping_reason": "no_improvement", "early_stopping_epoch": epoch})
            break
        
    
        train_loss = []
        grad_norm_store = []
        model.train()

        train_loop = tqdm(
            zip_longest(*trainLoaders), 
            total=max_dataset_train_length, 
            desc=f"Training Epoch {epoch+1} / {args['n_epochs']}"
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
                
                with torch.autocast(device_type = args["device"].split(":")[0], enabled = args['use_amp'], dtype = torch.bfloat16):
                    
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
                    if args["modelType"] == "gru":
                        pred = model.forward(X, X_len, dayIdx)
                    elif args["modelType"] == "transformer":
                        pred = model.forward(X, X_len, participant_id, dayIdx)
                    loss = forward_ctc(pred, adjustedLens, y, y_len)  
                    train_loss.append(loss.cpu().detach().numpy())    
                    
                loss.backward()
                
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                grad_norm_store.append(total_norm.item())
                
                if args['grad_norm_clip_value'] > 0: 
                    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                max_norm = args['grad_norm_clip_value'],
                                                error_if_nonfinite = True,
                                                foreach = True
                                                )
                optimizer.step()
                scheduler.step()  
                          
        avgTrainLoss = np.mean(train_loss)
        
        loss_array = []
        wer_array = [] 
        per_array = []
        
        current_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()  
        
        
        if epoch % args["evaluate_every_n_epochs"] == 0:
            
            for participant_id, valLoader in enumerate(valLoaders):
                
                loss, wer, per = evaluate(valLoader, model, participant_id, forward_ctc, args, decoder)
                loss_array.append(loss)
                if wer is not None:
                    wer_array.append(wer)
                per_array.append(per)
            
            endTime = time.time()
            
            # Log the metrics to wandb
            if args["evaluate_wer"]:
                log_dict = {
                    "train_ctc_Loss": avgTrainLoss,
                    "ctc_loss": loss_array[0],
                    "wer": wer_array[0],
                    "per": per_array[0],
                    "learning_rate": current_lr, 
                    "grad_norm": np.mean(grad_norm_store), 
                    "time_per_epoch": (endTime - startTime) / 100
                }
            else:
                log_dict = {
                    "train_ctc_Loss": avgTrainLoss,
                    "ctc_loss": loss_array[0],
                    "per": per_array[0],
                    "learning_rate": current_lr, 
                    "grad_norm": np.mean(grad_norm_store), 
                    "time_per_epoch": (endTime - startTime) / 100
                }

            
            if len(loss_array) > 0:
                if args["evaluate_wer"]:
                    for pid, (avgDayLoss, wer, per) in enumerate(zip(loss_array[1:], wer_array[1:], per_array[1:])):
                        log_dict[f"ctc_loss_{pid}"] = avgDayLoss
                        log_dict[f"wer_{pid}"] = wer
                        log_dict[f"per_{pid}"] = per
                else:
                    for pid, (avgDayLoss, per) in enumerate(zip(loss_array[1:], per_array[1:])):
                        log_dict[f"ctc_loss_{pid}"] = avgDayLoss
                        log_dict[f"per_{pid}"] = per

                    

            wandb.log(log_dict)
            
            # Save best models for mean across participants
            if args["evaluate_wer"] and len(valWER) > 0 and np.mean(wer_array) < np.min(valWER):
                torch.save(model.state_dict(), outputDir + "/modelWeights_WER")
                
            if len(valPER) > 0 and np.mean(per_array) < np.min(valPER):
                torch.save(model.state_dict(), outputDir + "/modelWeights_PER")
            
            # Save best models per participant
            for pid in range(len(loss_array)):
                suffix = get_participant_suffix(pid)
                if args["evaluate_wer"]:
                    if pid not in best_wer_by_participant or wer_array[pid] < best_wer_by_participant[pid]:
                        best_wer_by_participant[pid] = wer_array[pid]
                        torch.save(model.state_dict(), outputDir + f"/modelWeights_WER{suffix}")
                
                if pid not in best_per_by_participant or per_array[pid] < best_per_by_participant[pid]:
                    best_per_by_participant[pid] = per_array[pid]
                    torch.save(model.state_dict(), outputDir + f"/modelWeights_PER{suffix}")
                
            valLoss.append(np.mean(loss_array))
            if args["evaluate_wer"]:
                valWER.append(np.mean(wer_array)) 
            valPER.append(np.mean(per_array))
            
        
            if args["evaluate_wer"]: 
                if np.mean(wer_array) > np.min(valWER):
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            else:
                if np.mean(per_array) > np.min(valPER):
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    
                    
            if args["early_stopping_enabled"] and epoch == args["early_stopping_checkpoint"]:
                    # ! Stopping based upon b2t25' <HARD CODED>
                    if args["evaluate_wer"] and wer_array[0] > args["early_stopping_wer_threshold"]:
                        wandb.log({"early_stopping_triggered": True, "early_stopping_reason": "wer_threshold", "early_stopping_epoch": epoch})
                        break
                    elif per_array[0] > args["early_stopping_per_threshold"]:
                        wandb.log({"early_stopping_triggered": True, "early_stopping_reason": "per_threshold", "early_stopping_epoch": epoch})
                        break
                
    wandb.finish()
    
    # Log and print results
    best_mean_wer = np.min(valWER)
    best_mean_per = np.min(valPER)
    
    for pid in best_wer_by_participant.keys():
        suffix = get_participant_suffix(pid)
        print(f"Participant {pid}{suffix} - Best WER: {best_wer_by_participant[pid]:.4f}, Best PER: {best_per_by_participant[pid]:.4f}")
    
    print(f"Mean - Best WER: {best_mean_wer:.4f}, Best PER: {best_mean_per:.4f}")
    
    return best_mean_wer, best_mean_per, best_wer_by_participant, best_per_by_participant