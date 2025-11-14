# Save this file as hpo_trainer.py

import yaml
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.transformer_chunking import TransformerModel
from brainaudio.training.trainer import trainModel

def run_single_trial(config):
    """
    Runs a single training trial with the given config.
    Returns the final validation metric (e.g., WER or loss).
    """
    
    model_type = config['modelType']
    
    # We only care about the Transformer, but this makes the code robust
    if model_type == 'transformer':
        model_args = config['model'][model_type]
        model = TransformerModel(
            features_list=model_args['features_list'], 
            samples_per_patch=model_args['samples_per_patch'], 
            dim=model_args['d_model'], 
            depth=model_args['depth'], 
            heads=model_args['n_heads'], 
            mlp_dim_ratio=model_args['mlp_dim_ratio'],  
            dim_head=model_args['dim_head'], 
            dropout=config['dropout'], 
            input_dropout=config['input_dropout'], 
            nClasses=config['nClasses'], 
            max_mask_pct=config['max_mask_pct'], 
            num_masks=config['num_masks'], 
            num_participants=len(model_args['features_list']), 
            return_final_layer=False, 
            chunked_attention=model_args["chunked_attention"]
        )

    elif model_type == 'gru':
        model_args = config['model'][model_type]
        model = GRU_25(
            neural_dim=model_args['nInputFeatures'], 
            n_classes=config['nClasses'], 
            hidden_dim=model_args['nUnits'], 
            layer_dim=model_args['nLayers'], 
            nDays=model_args['nDays'], 
            dropout=config['dropout'], 
            input_dropout=config['input_dropout'],
            strideLen=model_args['strideLen'], 
            kernelLen=model_args['kernelLen'], 
            bidirectional=model_args['bidirectional'], 
            max_mask_pct=config['max_mask_pct'], 
            num_masks=config['num_masks']
        )
    
    else:
        raise ValueError(f"Unknown modelType: {model_type}")
        
    model.to(config['device'])

    # This function MUST return the validation metric you want to optimize
    wer, per = trainModel(config, model)
    
    return wer, per