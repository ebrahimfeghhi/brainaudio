# Pseudocode for loading the config
import yaml
from models.gru import GRU
from models.transformer import TransformerModel
from trainer import trainModel

###### user specified parameters ######
config_file = 'time_masked_gru_b2t_25.yaml'
#######################################

config_file = f"utils/custom_configs/{config_file}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
model_type = config['modelType']

model_args = config['model'][model_type]


if model_type == 'transformer':
    
    model = TransformerModel(model_args['patch_size'], model_args['d_model'], model_args['depth'], 
                        model_args['n_heads'], model_args['mlp_dim_ratio'], model_args['dim_head'], 
                        config['dropout'], config['input_dropout'], 
                        config['nClasses'], config['max_mask_pct'], 
                        config['num_masks'], config['gaussianSmoothWidth'], config['smooth_kernel_size'])
    
    
if model_type == 'gru':
    
    model = GRU(neural_dim=model_args['nInputFeatures'], n_classes=model_args['nClasses'], hidden_dim=model_args['nUnits'], 
                layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
                kernel_size=config['smooth_kernel_size'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
                num_masks=config['num_masks'])
        
    
model.to(config['device'])
    
trainModel(config, model)
