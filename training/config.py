# Pseudocode for loading the config
import yaml
from models.gru import GRU
from models.transformer import Transformer

with open('default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Select which model configuration to use based on an argument
model_type = 'transformer' # This could come from argparse
model_args = config['model'][model_type]

print(config)

if model_type == 'transformer':
    
    breakpoint()
    