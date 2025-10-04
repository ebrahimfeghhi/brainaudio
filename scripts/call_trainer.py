# Pseudocode for loading the config
import yaml
from brainaudio.models.gru_b2t_24 import GRU_24
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.transformer import TransformerModel
from brainaudio.training.trainer import trainModel

###### user specified parameters ######
config_file = 'tm_transformer_b2t_24+25_large.yaml'
#######################################

config_file = f"/home3/ebrahim2/brainaudio/src/brainaudio/training/utils/custom_configs/{config_file}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
model_type = config['modelType']


model_args = config['model'][model_type]


if model_type == 'transformer':
    
    model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], 
                     heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'], dim_head=model_args['dim_head'], 
                     dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                     max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
                     kernel_size=config['smooth_kernel_size'], num_participants=len(model_args['features_list']))
    
if model_type == 'gru':
    
    model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'], 
                layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
                kernel_size=config['smooth_kernel_size'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
                num_masks=config['num_masks'])
        
        
    
model.to(config['device'])
    
trainModel(config, model)
