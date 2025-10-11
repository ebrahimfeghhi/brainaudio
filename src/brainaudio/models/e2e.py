import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import numpy as np
import os
import yaml


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
        device
    ):  
        super().__init__()
        
        self.device = device
        
        
        self.tokenizer = tokenizer
        
        # llm
        self.llm = llm
    
        self.embedding_layer = llm.get_input_embeddings()
        llm_dim = self.embedding_layer.embedding_dim

        # neural ctc encoder 
        self.encoder = encoder 
        
        self.encoder_projecter = nn.Linear(encoder_dim, llm_dim)
        # projector
        #self.encoder_projector = setup_encoder_projector(model_config, input_dim=encoder_dim, output_dim=llm_dim)
        
        
        self.BOS_TOKEN = self.tokenizer.bos_token
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer(self.PAD_TOKEN, return_tensors="pt").input_ids.to(device)
        


    def forward(self, neuralInput, X_len, attention_mask, forced_alignments, adjusted_lens, participant_idx=None, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape ()
            attention_mask: 
            forced_alignments: (B , ) of shape [{}]
            adjusted_lens: (B , ) 
            participant_idx: integer ID, all data for a given batch must come from the same participant 
            dayIdx: Not used for Transformer 
        Returns:
            Tensor: (B, num_patches, dim)
        """
        B = neuralInput.shape[0]
        logits, encoder_outs = self.encoder(neuralInput, X_len, participant_idx, day_idx)
        projected_outs = self.encoder_projecter(encoder_outs) # B x T x llm_dim
        
        # labels [{13: 'Nuclear', 35: 'is', 50: 'the', 106: 'future'}]
        batched_llm_inputs = []
        for idx in range(B):
            fa = forced_alignments[idx]
            llm_inputs = []
            for i in range(adjusted_lens[idx]):
            
                llm_inputs.append(projected_outs[idx][i]) # 
                
                if i in fa.keys():
                    boundary_timesteps = list(fa.keys())
                    if i == boundary_timesteps[0]:
                        word = f'{self.BOS_TOKEN}{fa[i]}'
                    elif i == boundary_timesteps[-1]:
                        word = f'{fa[i]}{self.EOS_TOKEN}'
                    else:
                        word = f'{fa[i]} '
                    # tokenize word + get embedding vectors
                    token_id  = self.tokenizer(word, return_tensors="pt").input_ids.to(self.device)
                    embeddings = self.embedding_layer(token_id) # add tokens one by one 
                    llm_inputs.append(embeddings)
                        
            breakpoint()
                        
            batched_llm_inputs.append(torch.stack(llm_inputs))
            
        batched_llm_inputs = torch.stack(batched_llm_inputs)
        breakpoint()
        
        #model_outputs = self.llm(llm_inputs, attention_mask=None, )
        #return model_outputs, logits


    # if training, return 




