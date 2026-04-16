import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import yaml
#from brainaudio.models._archive.gru_b2t_25 import GRU_25
from brainaudio.models.gru_b2t_24 import GRU_24
#from brainaudio.models.transformer_interctc import TransformerModel 
from brainaudio.models.transformer_chunking import TransformerModel
#from brainaudio.models.transformer_demichunking import TransformerModel
#from brainaudio.models._archive.transformer import TransformerModel
from brainaudio.training.trainer import trainModel


config_path = "gru_b2t_24_baseline.yaml"
config_file = f"../src/brainaudio/training/utils/custom_configs/{config_path}"
device = "cuda:0"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

def _resolve_path(p: str) -> str:
    """Try the path as-is; if missing, try adding/removing /home/ebrahim prefix."""
    if os.path.exists(p):
        return p
    if p.startswith("/home/ebrahim"):
        alt = p.replace("/home/ebrahim", "", 1)
    else:
        alt = "/home/ebrahim" + p
    return alt if os.path.exists(alt) else p

config["outputDir"] = _resolve_path(config["outputDir"])
config["manifest_paths"] = [_resolve_path(p) for p in config["manifest_paths"]]

config["device"] = device
    
model_type = config['modelType']

model_args = config['model'][model_type]

if model_type == "gru":
    year = model_args["year"]

model_name = config["modelName"]

for seed in config['seeds']:

    print(f"Training with seed {seed}")
    
    config['seed'] = seed

    config["modelName"] = f"{model_name}_seed_{seed}"

    if model_type == 'transformer':
        
                
        model_args["d_model"] = model_args["n_heads"]*model_args['dim_head']
        config['learning_rate_min'] = config['learning_rate']*config['lr_scaling_factor']


        model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'],
                                 depth=model_args['depth'], heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'],
                                dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'],
                                max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], num_participants=len(model_args['features_list']), return_final_layer=False,
                                 chunked_attention=model_args["chunked_attention"], nDays=model_args.get("nDays"), day_softsign=model_args.get("day_softsign", False))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters are {total_params}")

    elif model_type == "gru" and year == "2024":
        model = GRU_24(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'],
                    layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                    strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'],
                    num_masks=config['num_masks'])
    elif model_type == "gru" and year == "2025":
        model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'],
                    layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                    strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'],
                    num_masks=config['num_masks'])
        

        
    model.to(config['device'])

    _ = trainModel(config, model)
