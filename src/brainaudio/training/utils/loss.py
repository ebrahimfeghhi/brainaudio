import torch
import tqdm 
from .augmentations import gauss_smooth
from edit_distance import SequenceMatcher
import numpy as np
import pandas as pd
from brainaudio.inference.inference_utils import _cer_and_wer, normalize_shorthand, clean_string

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
    


def evaluate(val_loader, model, participant_id, forward_ctc, args, beam_search_decoder):
    
    """
    Runs the validation loop for the model.

    Args:
        val_loader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        participant_id (int): The id of the participant.
        forward_ctc (function): The CTC loss function.
        args (Namespace or dict): A configuration object with necessary parameters 
                                  like smooth_kernel_size and gaussianSmoothWidth.
        beam_search_decoder: Pytorch CTC beam search decoder

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average validation loss.
            - wer (float): The Word Error Rate.
            - per (float): The Phoneme Error Rate.
    """
    
    # Set the model to evaluation mode
    model.eval()
    all_losses = []
    pred_arr = []
    val_transcripts = []

    device = args["device"]

    # Disable gradient calculations for validation
    with torch.no_grad():
          
        # Wrap loader in tqdm for a progress bar
        
        for batch in tqdm.tqdm(val_loader, desc=f"Evaluating Participant {participant_id}"):
          
            X, y, X_len, y_len, testDayIdx, val_transcript = batch
            
            val_transcripts.append(clean_string(val_transcript[0]))

            # Move data to the specified device
            X = X.to(device)
            y = y.to(device)
            X_len = X_len.to(device)
            y_len = y_len.to(device)
            testDayIdx = testDayIdx.to(device)

            X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
            adjusted_lens = model.compute_length(X_len)

            pred = model.forward(X, X_len, participant_id, testDayIdx)
            
            # --- Decode and Calculate PER ---
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
                
            loss = forward_ctc(pred, adjusted_lens, y, y_len)
            
            # Use .item() to get the scalar value of the loss
            all_losses.append(loss.item())
            
            beam_out = beam_search_decoder(pred.to("cpu")*args["acoustic_scale"])
            beam_search_transcript = normalize_shorthand(" ".join(beam_out[0][0].words).strip())
            pred_arr.append(clean_string(beam_search_transcript))
           

    # --- Calculate Final Metrics ---
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    _, wer, _= _cer_and_wer(pred_arr, val_transcripts)
    per = total_edit_distance / total_seq_length if total_seq_length > 0 else float('nan')
    
    print("WER: ", wer, "PER: ", per)

    return avg_loss, wer, per
