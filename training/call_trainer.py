# Pseudocode for loading the config
import yaml
from models.gru import GRU
from models.transformer import TransformerModel
from trainer import trainModel


config_file = 'time_masked_gm_b2t_25.yaml'
model_type = 'gru'

config_file = f"utils/custom_configs/{config_file}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

model_args = config['model'][model_type]


if model_type == 'transformer':
    
    model = TransformerModel(model_args['patch_size'], model_args['d_model'], model_args['depth'], 
                        model_args['n_heads'], model_args['mlp_dim_ratio'], model_args['dim_head'], 
                        config['dropout'], config['input_dropout'], 
                        config['nClasses'], config['max_mask_pct'], 
                        config['num_masks'], config['gaussianSmoothWidth'], config['smooth_kernel_size'])
    
    
if model_type == 'gru':
    
    model = GRU(model_args['nInputFeatures'], model_args['nClasses'], model_args['nUnits'], model_args['nLayers'], 
                model_args['nDays'], config['dropout'], config['input_dropout'], 
                model_args['strideLen'], model_args['kernelLen'], config['gaussianSmoothWidth'], config['smooth_kernel_size'], 
                model_args['bidirectional'], config['max_mask_pct'], config['num_masks'])
    
model.to(config['device'])
    
breakpoint()
trainModel(config, model)
