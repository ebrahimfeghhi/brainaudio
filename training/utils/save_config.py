'''
This script is used to save a config setting.
To use this script, first modify the custom_config.yaml file.
Running the script will save the contents of the custom_config file into the 
custom_configs folder under the name provided by the user. 
'''
import yaml

# specify file name, config setup is saved in custom_configs folder
config_saved_name = "time_masked_transformer" 

with open('custom_config.yaml', 'r') as f:
    custom_config = yaml.safe_load(f)

full_path = f"custom_configs/{config_saved_name}.yaml"

with open(full_path, 'w') as f:
    yaml.dump(custom_config, f, sort_keys=False)