import torch
import tqdm 
from .augmentations import gauss_smooth
from edit_distance import SequenceMatcher
import numpy as np
import pandas as pd
from brainaudio.inference.inference_utils import _cer_and_wer

def forward_ctc(
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
    
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = encoder_out.log_softmax(2) # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2).cpu(),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="mean",
        )
        return ctc_loss
    
    
  
def evaluate(val_loader, model, participant_id, forward_ctc, args, epoch):
    """
    Runs the validation loop for the model.

    Args:
        val_loader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        participant_id (int): The id of the participant.
        forward_ctc (function): The CTC loss function.
        args (Namespace or dict): A configuration object with necessary parameters 
                                  like smooth_kernel_size and gaussianSmoothWidth.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average validation loss.
            - cer (float): The Character Error Rate.
    """
    # Set the model to evaluation mode
    model.eval()
    all_losses = []
    total_edit_distance = 0
    total_seq_length = 0

    device = args["device"]
    # Disable gradient calculations for validation
    with torch.no_grad():
      
        # Wrap loader in tqdm for a progress bar
        
        for batch in tqdm.tqdm(val_loader, desc=f"Evaluating Participant {participant_id}"):
          
            X, y, X_len, y_len, testDayIdx = batch

            # Move data to the specified device
            X = X.to(device)
            y = y.to(device)
            X_len = X_len.to(device)
            y_len = y_len.to(device)
            testDayIdx = testDayIdx.to(device)

            X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
            adjusted_lens = model.compute_length(X_len)

            pred = model.forward(X, X_len, participant_id, testDayIdx)
            loss = forward_ctc(pred, adjusted_lens, y, y_len)
            
            # Use .item() to get the scalar value of the loss
            all_losses.append(loss.item())

            # --- Decode and Calculate CER ---
            for i in range(pred.shape[0]):
                # Get the single predicted and true sequences
                pred_seq = pred[i, :adjusted_lens[i], :]
                true_seq_full = y[i, :y_len[i]]

                # Decode the prediction
                decoded_seq_indices = torch.argmax(pred_seq, dim=-1)
                decoded_seq_uniqued = torch.unique_consecutive(decoded_seq_indices)
                decoded_seq_list = [i for i in decoded_seq_uniqued.cpu().numpy() if i != 0]

                # Convert true sequence to list
                true_seq_list = true_seq_full.cpu().numpy().tolist()

                # Calculate edit distance
                matcher = SequenceMatcher(a=true_seq_list, b=decoded_seq_list)
                total_edit_distance += matcher.distance()
                total_seq_length += len(true_seq_list)

    # --- Calculate Final Metrics ---
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    
    # Calculate CER, handling division by zero
    cer = total_edit_distance / total_seq_length if total_seq_length > 0 else float('nan')

    return avg_loss, cer


 
def evaluate_wer(val_loader, model, participant_id, forward_ctc, args, beam_search_decoder):
    """
    Runs the validation loop for the model.

    Args:
        val_loader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        participant_id (int): The id of the participant.
        forward_ctc (function): The CTC loss function.
        args (Namespace or dict): A configuration object with necessary parameters 
                                  like smooth_kernel_size and gaussianSmoothWidth.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average validation loss.
            - cer (float): The Character Error Rate.
    """
    # Set the model to evaluation mode
    model.eval()
    all_losses = []
    pred_arr = []
    
    if participant_id == 1:
        val_transcripts = pd.read_pickle("/data2/brain2text/b2t_24/transcripts_val.pkl")
    else:
        val_transcripts = pd.read_pickle("/data2/brain2text/b2t_25/transcripts_val.pkl")

    device = args["device"]
    acoustic_scale = 0.8
    # Disable gradient calculations for validation
    with torch.no_grad():
      
        print("EVAL WER")
      
        # Wrap loader in tqdm for a progress bar
        
        for batch in tqdm.tqdm(val_loader, desc=f"Evaluating Participant {participant_id}"):
          
            X, y, X_len, y_len, testDayIdx = batch

            # Move data to the specified device
            X = X.to(device)
            y = y.to(device)
            X_len = X_len.to(device)
            y_len = y_len.to(device)
            testDayIdx = testDayIdx.to(device)

            X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
            adjusted_lens = model.compute_length(X_len)

            pred = model.forward(X, X_len, participant_id, testDayIdx)
            loss = forward_ctc(pred, adjusted_lens, y, y_len)
            
            # Use .item() to get the scalar value of the loss
            all_losses.append(loss.item())
            
            beam_out = beam_search_decoder(pred.to("cpu")*acoustic_scale)
            beam_search_transcript = " ".join(beam_out[0][0].words).strip()
            pred_arr.append(beam_search_transcript)

    # --- Calculate Final Metrics ---
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    cer, wer, wer_sent = _cer_and_wer(pred_arr, val_transcripts)
    
    print("WER: ", wer)

    return avg_loss, wer

def evaluate_e2e(val_loader, model, participant_id, forward_ctc, args, chunk_size, llm_context_chunks):
  """
    Runs the validation loop for E2E model.

    Args:
        val_loader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        participant_id (int): The id of the participant.
        forward_ctc (function): The CTC loss function.
        args (Namespace or dict): A configuration object with necessary parameters 
                                  like smooth_kernel_size and gaussianSmoothWidth.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average validation loss.
            - cer (float): The Character Error Rate.
    """

    # Set the model to evaluation mode
  model.eval()
  all_losses = []
  all_ctc_losses = []
  all_ce_losses = []
  
  # For WER calculation
  all_predictions = []
  all_ground_truths = []

  device = args["device"]
  ctc_weight = args["ctc_weight"]

  with torch.no_grad():
    for batch in tqdm(val_loader, desc=f"Evaluating Participant {participant_id}"):
      X, y, X_len, y_len, testDayIdx, transcript, forced_alignments = batch 

      # Move data to the specified device
      X = X.to(device)
      y = y.to(device)
      X_len = X_len.to(device)
      y_len = y_len.to(device)
      testDayIdx = testDayIdx.to(device)
      forced_alignments = forced_alignments.to(device)

      X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])

      with torch.autocast(device_type=args["device"], enabled=args['use_amp'], dtype=torch.bfloat16):
        adjustedLens = model.encoder.compute_length(X_len)
        llm_outs, ctc_logits = model(
            neuralInput=X,
            adjusted_lens=adjustedLens,
            forced_alignments=forced_alignments,
            chunk_size=chunk_size,
            llm_context_chunks=llm_context_chunks,
            participant_idx=participant_id
        )
        ctc_loss = forward_ctc(ctc_logits, adjustedLens, y, y_len)
        ce_loss = llm_outs.loss
        loss = (1 - ctc_weight) * ce_loss + ctc_weight * ctc_loss

        all_losses.append(loss.item())
        all_ctc_losses.append(ctc_loss.item())
        all_ce_losses.append(ce_loss.item())

        predictions = model.generate(
            neuralInput=X,
            adjusted_lens=adjustedLens,
            chunk_size=args.get('chunk_size', 0),
            llm_context_chunks=args.get('llm_context_chunks', 1),
            participant_idx=participant_id
        )
        all_predictions.extend(predictions)
        all_ground_truths.extend(transcript)

  # Calculate final metrics
  avg_loss = np.mean(all_losses)
  avg_ctc = np.mean(all_ctc_losses)
  avg_ce = np.mean(all_ce_losses)

  
  # Simple WER calculation using edit distance
  total_edit_distance = 0
  total_words = 0
  for pred, truth in zip(all_predictions, all_ground_truths):
      pred_words = pred.split()
      truth_words = truth.split()
      matcher = SequenceMatcher(a=pred_words, b=truth_words)
      total_edit_distance += matcher.distance()
      total_words += len(truth_words)
  
  wer = (total_edit_distance / total_words) * 100 if total_words > 0 else 0

  return (avg_loss, avg_ctc, avg_ce), wer