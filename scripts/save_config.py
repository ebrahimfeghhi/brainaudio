'''
This script is used to save a config setting.
To use this script, first modify the custom_config.yaml file.
Running the script will save the contents of the custom_config file into the 
custom_configs folder under the name provided by the user. 
'''
import yaml

with open('custom_config.yaml', 'r') as f:
    custom_config = yaml.safe_load(f)
    
config_saved_name = custom_config['modelName']

full_path = f"../src/brainaudio/training/utils/custom_configs/{config_saved_name}.yaml"

with open(full_path, 'w') as f:
    yaml.dump(custom_config, f, sort_keys=False)