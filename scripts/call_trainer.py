import yaml
from brainaudio.models._archive.gru_b2t_25 import GRU_25
#from brainaudio.models.transformer_interctc import TransformerModel 
from brainaudio.models.transformer_chunking import TransformerModel
#from brainaudio.training.trainer_interctc import trainModel
from brainaudio.training.trainer import trainModel


config_path = "neurips_b2t_25_chunked_transformer.yaml"
config_file = f"../src/brainaudio/training/utils/custom_configs/{config_path}"
device = "cuda:6"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
config["device"] = device
    
model_type = config['modelType']

model_args = config['model'][model_type]

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
                                 chunked_attention=model_args["chunked_attention"])

    if model_type == 'gru':
        
        model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'], 
                    layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                    strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
                    num_masks=config['num_masks'])
        
    model.to(config['device'])

    _ = trainModel(config, model)


