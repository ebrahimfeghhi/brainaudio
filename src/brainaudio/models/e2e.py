import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import numpy as np
import os
import yaml
from peft import get_peft_model, LoraConfig



from brainaudio.models.gru_b2t_24 import GRU_24
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.transformer import TransformerModel
# ------------------------------------- Encoder Projector Setup Functions ------------------------------------- #
class EncoderProjectorConcat(nn.Module):
    def __init__(self, config:dict, output_dim:int):
        super().__init__()
        self.k = config['encoder_projector_ds_rate']
        self.input_dim = config['encoder_dim']
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.output_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config:dict, input_dim:int, output_dim:int):
        super().__init__()
        self.k = config['encoder_projector_ds_rate']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1d = nn.Conv1d(in_channels=self.input_dim, out_channels=self.input_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.input_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.output_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config:dict, input_dim:int, output_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.input_dim
        configuration.num_hidden_layers = config['qformer_layers']

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj

def setup_encoder_projector(model_config, input_dim, output_dim):
    """
    Sets up an encoder projector model to project encoder last layer to LLM token space.

    Args:
        model_config (dict): A dictionary containing the key "encoder_model_path".
        input_dim (int): The dimension of ctc encoder last hidden layer
        output_dim (int): The dimension of LLM token space that the network outputs to
        
    Returns:
        model (nn.Module) : The loaded model that outputs neural embedding.
    """
    if model_config["projector_type"] == "linear":  #! assume we have this field in the model config
        encoder_projector = EncoderProjectorConcat(model_config, input_dim, output_dim)
    elif model_config["projector_type"] == "cov1d-linear":
        encoder_projector = EncoderProjectorCov1d(model_config, input_dim, output_dim)
    elif model_config["projector_type"] == "q-former":
        encoder_projector = EncoderProjectorQFormer(model_config, input_dim, output_dim)
    else:
        return None
    return encoder_projector 


# ------------------------------------- E2E model class definition ------------------------------------- #
class E2EModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int, 
        llm: nn.Module,
        tokenizer,
        device,
        peft_config: dict = None, # Pass your PEFT args here
    ):  
        super().__init__()
        
        self.device = device
        self.tokenizer = tokenizer

        # If a peft_config is provided, apply it to the base LLM
        if peft_config:
            lora_config = LoraConfig(**peft_config) # Unpack dict to create LoraConfig
            self.llm = get_peft_model(llm, lora_config)
            print("PEFT LoRA enabled for LLM.")
            self.llm.print_trainable_parameters()
        else:
            self.llm = llm # Use the original LLM if no PEFT config

        self.embedding_layer = llm.get_input_embeddings()
        llm_dim = self.embedding_layer.embedding_dim
        # neural ctc encoder 
        self.encoder = encoder 
        # projector
        self.encoder_projecter = nn.Linear(encoder_dim, llm_dim)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.pad_token_id = self.tokenizer.pad_token_id
        self.EOS_TOKEN = self.tokenizer.eos_token
        

        self.EOS_TOKEN = self.tokenizer.eos_token
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.eoc_id = self.tokenizer(self.EOS_TOKEN, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        # self.eoc_embed =  self.embedding_layer(self.eoc_id)
        self.pad_id = self.tokenizer(self.PAD_TOKEN, return_tensors="pt").input_ids.to(device)
        self.pad_embed = self.embedding_layer(self.pad_id)
        


    def forward(self, neuralInput, Input_len, forced_alignments, adjusted_lens, chunk_size, llm_context_chunks, participant_idx=None, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            Input_len: Tensor of shape (B, )
            attention_mask: 
            forced_alignments: Tensor of shape (B , ) in the format of [{}]
            adjusted_lens: Tensor of shape (B , ) the adjusted length for encoder model
            chunk_size: int
            llm_context: int, how many chunks the llm may look back
            participant_idx: integer ID, all data for a given batch must come from the same participant 
            dayIdx: Not used for Transformer 

        Returns:
            Tensor: (B, num_patches, dim)
        """
        # I don't think this version is working but kept in case the entire logic of the chunk-counting outerloop and batch-counting inner loop is wrong
        # B = neuralInput.shape[0]
        # logits, encoder_outs = self.encoder(neuralInput, Input_len, participant_idx, day_idx) # B x T x encoder_dim
        # projected_outs = self.encoder_projecter(encoder_outs) # B x T x llm_dim

        # # Account for the case when we don't chunk and this is non-streaming
        # effective_chunk_size = chunk_size if chunk_size > 0 else projected_outs[1]
        # num_chunks = (projected_outs[1] + effective_chunk_size - 1) // effective_chunk_size

        # chunked_input_embeds, chunked_labels = [], []

        # past_key_values = None
        # for batch_idx in range(B):
        #     effective_chunk_size = chunk_size if chunk_size > 0 else adjusted_lens[batch_idx]
        #     num_chunks = (adjusted_lens[batch_idx] + effective_chunk_size - 1) // effective_chunk_size

        #     batch_neural_embed = projected_outs[batch_idx]
        #     batch_fa = forced_alignments[batch_idx]

        #     last_chunk_padding_needed = None
        #     for chunk_idx in range(num_chunks):
        #         embeds_for_chunk, labels_for_chunk = [], [] # Shape: (1, num_chunks, *effective_chunk_size + text_token_size)
        #         start_timestep = chunk_idx * effective_chunk_size
        #         end_timestep = start_timestep + effective_chunk_size
                
        #         if end_timestep > len(batch_neural_embed[0]):
        #             # For the unfinished neural data after chunking
        #             last_chunk_padding_needed = len(batch_neural_embed[0]) - 1 - end_timestep
        #             end_timestep = len(batch_neural_embed[0])-1

        #         # Prepare inputs of the chunk
        #         current_chunk_neural = batch_neural_embed[start_timestep:end_timestep]
        #         span_fa = {t: w for t, w in batch_fa.items() if start_timestep <= t <= end_timestep}
        #         span_boundary_timesteps = list(span_fa.keys())

        #         embeds_for_chunk.append(current_chunk_neural.unsqueeze(0))
        #         labels_for_chunk.append(torch.tensor([-100], device = self.device).repeat(end_timestep-start_timestep))

        #         if last_chunk_padding_needed:
        #             pad_embeds = self.pad_embed.repeat(last_chunk_padding_needed, 1)
        #             # Create padding labels: (padding_needed,) filled with -100
        #             pad_labels = torch.full((last_chunk_padding_needed,), -100, device=self.device)
        #             embeds_for_chunk  = torch.cat([seq_embeds, pad_embeds], dim=0)
        #             labels_for_chunk = torch.cat([seq_labels, pad_labels], dim=0)


        #         word_text = ""
        #         for word_end_time in span_boundary_timesteps:
        #             word_text += span_fa[word_end_time]+' ' if word_end_time != span_boundary_timesteps[-1] else span_fa[word_end_time]
                    
        #         # Tokenize the word and get its IDs and embeddings
        #         token_ids = self.tokenizer(word_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        #         word_embeds = self.embedding_layer(token_ids) # Shape: (1, num_tokens, llm_dim)
        #         embeds_for_chunk.append(word_embeds.squeeze(0))

        #         labels_for_chunk.append(token_ids.squeeze(0))
        #         labels_for_chunk.append(self.eoc_id.squeeze(0))


        #     chunked_input_embeds.append(embeds_for_chunk)
        #     chunked_labels.append(labels_for_chunk)

        # max_seq_len = max(len(seq) for seq in chunked_input_embeds)
        # padded_embeds = []
        # padded_labels = []  
        # for i in range(B):
        #     seq_embeds = chunked_input_embeds[i]  # Shape: (T, llm_dim)
        #     seq_labels = chunked_labels[i]        # Shape: (T,)
            
        #     current_len = seq_embeds.shape[0]
        #     padding_needed = max_seq_len - current_len

        #     if padding_needed > 0:
        #         # Create padding embeddings: (padding_needed, llm_dim)
        #         pad_embeds = self.pad_embed.repeat(padding_needed, 1)
                
        #         # Create padding labels: (padding_needed,) filled with -100
        #         pad_labels = torch.full((padding_needed,), -100, device=self.device)
                
        #         # Concatenate original + padding
        #         padded_seq_embeds = torch.cat([seq_embeds, pad_embeds], dim=0)
        #         padded_seq_labels = torch.cat([seq_labels, pad_labels], dim=0)
        #     else:
        #         padded_seq_embeds = seq_embeds
        #         padded_seq_labels = seq_labels
        #     padded_embeds.append(padded_seq_embeds)
        #     padded_labels.append(padded_seq_labels)

        # padded_embeds = torch.stack(padded_embeds)                          # Shape: (B, T_max, llm_dim)
        # padded_labels = torch.stack(padded_labels)                          # Shape: (B, T_max)
        # padded_attention_mask = (padded_embeds.sum(dim=-1) != 0).long()     # Shape: (B, T_max)

        
        # # caching the 
        # model_outputs = self.llm(
        #         inputs_embeds=padded_embeds,
        #         attention_mask=padded_attention_mask,
        #         labels=padded_labels,
        #         past_key_values=past_key_values,
        #         use_cache=True
        #     )

        # if chunk_size > 0:
        #         max_cache_len = (llm_context * chunk_size) + padded_embeds.shape[1]
        #         past_key_values = self.trim_kv_cache(past_key_values, max_len=max_cache_len)
            
        # return model_outputs, logits

        # ======================================================================================== #
        B, T, _ = neuralInput.shape
        _, encoder_outs = self.encoder(neuralInput, Input_len, participant_idx, day_idx)
        projected_outs = self.encoder_projecter(encoder_outs)

        # If non streaming, essentially build one big chunk
        effective_chunk_size = chunk_size if chunk_size > 0 else T
        num_chunks = (T + effective_chunk_size - 1) // effective_chunk_size

        past_key_values = None
        all_logits = []

        for chunk_idx in range(num_chunks):
            batch_embeds_for_chunk = []
            batch_labels_for_chunk = []
            
            start_timestep = chunk_idx * effective_chunk_size
            end_timestep = start_timestep + effective_chunk_size

            # INNER LOOP: Prepare this chunk for all items in the batch
            for item_idx in range(B):
                true_len = adjusted_lens[item_idx]
                embeds, labels = self.prepare_interleaved_inputs(
                    projected_outs[item_idx], forced_alignments[item_idx], start_timestep, end_timestep, true_len
                )
                if embeds is not None:
                    batch_embeds_for_chunk.append(embeds)
                    batch_labels_for_chunk.append(labels)

            if not batch_embeds_for_chunk:
                continue
            
            # Padding
            # Find the length of the longest sequence in this chunk's batch
            max_len = max(embeds.shape[0] for embeds in batch_embeds_for_chunk)
            # 1. Get the true length of each sequence in this chunk's batch
            true_lengths = torch.tensor([len(labels) for labels in batch_labels_for_chunk], device=self.device)
            # 2. Create the correct attention mask based on these true lengths
            max_len = true_lengths.max()
            padded_attention_mask = (torch.arange(max_len, device=self.device)[None, :] < true_lengths[:, None]).long() #?

            padded_embeds_list = []
            padded_labels_list = []

            # Now create the attention mask based on the labels
            padded_attention_mask = (padded_labels != -100)


            # Manually pad each sequence in the batch
            for i in range(len(batch_embeds_for_chunk)):
                embeds = batch_embeds_for_chunk[i]
                labels = batch_labels_for_chunk[i]
                
                current_len = embeds.shape[0]
                padding_needed = max_len - current_len
                
                if padding_needed > 0:
                    # Create padding embeddings using the dedicated pad token
                    pad_embeds = self.embedding_layer(
                        torch.tensor([self.pad_token_id] * padding_needed, device=self.device)
                    )
                    
                    # Create padding labels filled with -100
                    pad_labels = torch.full((padding_needed,), -100, device=self.device, dtype=torch.long)
                    
                    # Concatenate original + padding
                    final_embeds = torch.cat([embeds, pad_embeds], dim=0)
                    final_labels = torch.cat([labels, pad_labels], dim=0)
                else:
                    final_embeds = embeds
                    final_labels = labels
                    
                padded_embeds_list.append(final_embeds)
                padded_labels_list.append(final_labels)

            # Stack the padded sequences into a final batch tensor
            padded_embeds = torch.stack(padded_embeds_list)
            padded_labels = torch.stack(padded_labels_list)


            # LLM call is INSIDE the chunk loop for streaming
            model_outputs = self.llm(
                inputs_embeds=padded_embeds,
                attention_mask=padded_attention_mask,
                labels=padded_labels,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            all_logits.append(model_outputs.logits)
            past_key_values = model_outputs.past_key_values

            # Only trim cache if in streaming mode
            if chunk_size > 0:
                max_cache_len = (llm_context_chunks * chunk_size) + padded_embeds.shape[1]
                past_key_values = self._trim_kv_cache(past_key_values, max_len=max_cache_len)
        
        return model_outputs, all_logits



        # ======================================================================================== #
        # forced_alignments [{13: 'Nuclear', 35: 'is', 50: 'the', 106: 'future'}]
        # This is an even older implementation

        # Final sequences for the batch
        batch_input_embeds = []
        batch_labels = []

        for idx in range(B):
            fa = forced_alignments[idx]
            boundary_timesteps = list(fa.keys())
            
            # Build one sequence at a time
            single_sequence_embeds = []
            single_sequence_labels = []

            for i in range(adjusted_lens[idx]):
                # Append the projected neural vector
                # Shape: (1, llm_dim)

                single_sequence_embeds.append(projected_outs[idx, i].unsqueeze(0))
                
                # For neural parts, we ignore the label during loss calculation
                single_sequence_labels.append(torch.tensor([-100], device=self.device))

                if i in boundary_timesteps:
                    word_text = fa[i]
                    
                    if i == boundary_timesteps[0]:
                        word = self.BOS_TOKEN + word_text
                    elif i == boundary_timesteps[-1]:
                        word = word_text + self.EOS_TOKEN
                    else:
                        word = word_text  + ' '

                    # Tokenize the word and get its IDs and embeddings
                    token_ids = self.tokenizer(word_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                    word_embeds = self.embedding_layer(token_ids) # Shape: (1, num_tokens, llm_dim)
                    
                    # Squeeze the batch dim and append the embeddings
                    single_sequence_embeds.append(word_embeds.squeeze(0))
                    
                    # Append the corresponding token IDs as labels
                    single_sequence_labels.append(token_ids.squeeze(0))
            
            # Concatenate all parts for this single example 
            final_embeds = torch.cat(single_sequence_embeds, dim=0) # Shape: (B, *num_tokens, llm_dim)
            final_labels = torch.cat(single_sequence_labels, dim=0) # Shape: (B, *num_tokens, llm_dim)


            batch_input_embeds.append(final_embeds)
            batch_labels.append(final_labels)

        max_seq_len = max(len(seq) for seq in batch_input_embeds)
        padded_embeds = []
        padded_labels = []  
        for i in range(B):
            seq_embeds = batch_input_embeds[i]  # Shape: (seq_len, llm_dim)
            seq_labels = batch_labels[i]        # Shape: (seq_len,)
            
            current_len = seq_embeds.shape[0]
            padding_needed = max_seq_len - current_len

            if padding_needed > 0:
                # Create padding embeddings: (padding_needed, llm_dim)
                pad_embeds = self.pad_embed.repeat(padding_needed, 1)
                
                # Create padding labels: (padding_needed,) filled with -100
                pad_labels = torch.full((padding_needed,), -100, device=self.device)
                
                # Concatenate original + padding
                padded_seq_embeds = torch.cat([seq_embeds, pad_embeds], dim=0)
                padded_seq_labels = torch.cat([seq_labels, pad_labels], dim=0)
            else:
                padded_seq_embeds = seq_embeds
                padded_seq_labels = seq_labels
            padded_embeds.append(padded_seq_embeds)
            padded_labels.append(padded_seq_labels)

        padded_embeds = torch.stack(padded_embeds)  # Shape: (B, max_seq_len, llm_dim)
        padded_labels = torch.stack(padded_labels)  # Shape: (B, max_seq_len)

        # Create the attention mask for the padded batch
        # 1 for real tokens/embeddings, 0 for padding
        # padded_attention_mask = (padded_embeds.sum(dim=-1) != 0).long()

        # Pass everything to the LLM
        # Use `inputs_embeds` when providing embeddings directly
        model_outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention_mask,
            labels=padded_labels
        )
        
        return model_outputs, logits

    
    @torch.no_grad()
    def generate(self, neuralInput, adjusted_lens, chunk_size, llm_context_chunks, max_new_tokens_per_chunk=50, top_k=10, participant_idx=None):
        """
        Generates text autoregressively from neural input.

        Args:
            neuralInput (Tensor): The input neural data (B, T, F).
            adjusted_lens (Tensor): The true length of each sample in the batch (B,).
            chunk_size (int): The size of each neural chunk for streaming. 0 for non-streaming.
            llm_context_chunks (int): The number of past chunks to use as context.
            max_new_tokens_per_chunk (int): The maximum number of text tokens to generate for each chunk.
            participant_idx (Tensor, optional): Participant indices.

        Returns:
            list[str]: A list of generated transcriptions for each sample in the batch.
        """
        self.eval() # Set the model to evaluation mode
        B, T, _ = neuralInput.shape

        # 1. Get Projected Neural Embeddings
        _, encoder_outs = self.encoder(neuralInput, adjusted_lens, participant_idx)
        projected_outs = self.encoder_projecter(encoder_outs)

        # 2. Unified Logic for Chunking
        effective_chunk_size = chunk_size if chunk_size > 0 else T
        num_chunks = (T + effective_chunk_size - 1) // effective_chunk_size

        # 3. Initialize Generation Variables
        past_key_values = None
        # This will store the final generated token IDs for each sample in the batch
        generated_ids = [[] for _ in range(B)]

        # Generations
        # 4. Outer Loop: Process Chunk by Chunk
        for chunk_idx in range(num_chunks):
            start_timestep = chunk_idx * effective_chunk_size
            # Prepare the initial input for this chunk (the neural data) for all batch items
            # Find the max length of real data in this chunk across the batch
            max_len_in_chunk = 0
            initial_chunk_embeds = []
            for i in range(B):
                true_len = adjusted_lens[i]
                end_timestep = min(start_timestep + effective_chunk_size, true_len)
                chunk_data = projected_outs[i, start_timestep:end_timestep]
                initial_chunk_embeds.append(chunk_data)

                if len(chunk_data) > max_len_in_chunk:
                    max_len_in_chunk = len(chunk_data)
                else:
                    # This sample has run out of data, add an empty tensor
                    initial_chunk_embeds.append(torch.empty(0, projected_outs.shape[-1], device=self.device))

            # If the entire batch has no data for this chunk, stop.
            if max_len_in_chunk == 0:
                break
            
            # Keep track of which sequences in the batch have finished generating their chunk
            unfinished_sequences = torch.ones(B, dtype=torch.long, device=self.device)
            padded_embeds_list = []
            for i in range(len(initial_chunk_embeds)):
                embeds = initial_chunk_embeds[i]
                current_len = embeds.shape[0]
                padding_needed = max_len_in_chunk - current_len
                
                if padding_needed > 0:
                    # Create padding embeddings using the dedicated pad token
                    pad_embeds = self.embedding_layer(
                        torch.tensor([self.pad_token_id] * padding_needed, device=self.device)
                    )
                    # Concatenate original + padding
                    final_embeds = torch.cat([embeds, pad_embeds], dim=0)
                else:
                    final_embeds = embeds
                    
                padded_embeds_list.append(final_embeds)
            # Stack the padded sequences into a final batch tensor
            current_input_embeds = torch.stack(padded_embeds_list)

            # 5. Inner Loop: Autoregressive Token-by-Token Generation
            for _ in range(max_new_tokens_per_chunk):
                attention_mask = (current_input_embeds.sum(dim=-1) != 0).long()
                # Get model outputs
                outputs = self.llm(
                    inputs_embeds=current_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                # Get the logits for the very last token prediction
                next_token_logits = outputs.logits[:, -1, :]
                # Perform top-k + Beam Search decoding to get the next token
                if top_k > 0:
                    # 1. Get the top k logits and their original indices
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)

                    # 2. Convert the filtered logits to probabilities
                    top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

                    # 3. Sample from the Top-K distribution
                    # multinomial returns the index within the top_k_probs tensor
                    sampled_indices_in_top_k = torch.multinomial(top_k_probs, num_samples=1)

                    # 4. Get the actual token IDs from the original vocabulary
                    next_token_ids = torch.gather(top_k_indices, dim=-1, index=sampled_indices_in_top_k).squeeze(-1)
                else:
                    # Fallback to greedy decoding if top_k is 0
                    next_token_ids = torch.argmax(next_token_logits, dim=-1)
                    # Mark sequences that generated an EOS token as "finished" for this chunk
                    unfinished_sequences = unfinished_sequences & (next_token_ids != self.tokenizer.eos_token_id)

                # Append the generated token to the results for unfinished sequences
                for i in range(B):
                    if unfinished_sequences[i]:
                        generated_ids[i].append(next_token_ids[i].item())

                # If all sequences in the batch have generated EOS, we can stop early
                if unfinished_sequences.sum() == 0:
                    break

                # Prepare for the next step
                past_key_values = outputs.past_key_values
                next_token_embeds = self.embedding_layer(next_token_ids.unsqueeze(-1))
                current_input_embeds = next_token_embeds

            # 6. Manage Context for the Next Chunk
            if chunk_size > 0:
                # Estimate a safe length to trim the cache to
                max_cache_len = (llm_context_chunks * chunk_size) * 2 # A heuristic, can be tuned
                past_key_values = self.trim_kv_cache(past_key_values, max_len=max_cache_len)

        # 7. Decode the final token IDs into text
        final_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return final_outputs


    def prepare_interleaved_inputs(self, projected_outs_single, fa_single, start_timestep, end_timestep, true_len):
        """Prepares interleaved inputs for a span, respecting the true sequence length."""
        # This check is now primary: if the whole chunk is beyond the true length, it's empty.
        if start_timestep >= true_len:
            return None, None

        span_fa = {t: w for t, w in fa_single.items() if start_timestep <= t < end_timestep}
        span_boundary_timesteps = list(span_fa.keys())

        embeds, labels = [], []

        # This loop now correctly stops at the true end of the data
        for i in range(start_timestep, min(end_timestep, true_len)):
            embeds.append(projected_outs_single[i])
            labels.append(torch.tensor(-100, device=self.device, dtype=torch.long))

            if i in span_boundary_timesteps:
                word_text = ' ' + span_fa[i]
                token_ids = self.tokenizer(word_text, add_special_tokens=False).input_ids
                word_embeds = self.embedding_layer(torch.tensor(token_ids, device=self.device))
                
                embeds.extend(list(torch.unbind(word_embeds, dim=0)))
                labels.extend(torch.tensor(token_ids, device=self.device, dtype=torch.long))

        if span_fa:
            eoc_id = self.tokenizer(self.EOS_TOKEN, add_special_tokens=False).input_ids
            eoc_embed = self.embedding_layer(torch.tensor(eoc_id, device=self.device))
            
            embeds.extend(list(torch.unbind(eoc_embed, dim=0)))
            labels.extend(torch.tensor(eoc_id, device=self.device, dtype=torch.long))
        
        if not embeds: return None, None
        return torch.stack(embeds), torch.stack(labels)

        

    def trim_kv_cache(self, past_key_values, max_len):
        """Trims the KV cache to a maximum sequence length."""
        if past_key_values is None: return None
        trimmed_cache = []
        for key, value in past_key_values:
            if key.shape[2] > max_len:
                trimmed_key = key[:, :, -max_len:, :]
                trimmed_value = value[:, :, -max_len:, :]
                trimmed_cache.append((trimmed_key, trimmed_value))
            else:
                trimmed_cache.append((key, value))
        return tuple(trimmed_cache)

