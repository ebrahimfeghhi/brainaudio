import yaml
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


from brainaudio.models.gru_b2t_24 import GRU_24
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.transformer import TransformerModel
from brainaudio.models.e2e import E2EModel
from brainaudio.training.trainer import trainModel
from brainaudio.training.e2e_trainer import trainE2EModel

argparser = argparse.ArgumentParser()
#argparser.add_argument('--mode', type=str, choices=['train_e2e', 'train_ctc'], default='train_ctc')
#argparser.add_argument('--config_path', type=str)
#args = argparser.parse_args()

#mode = args['mode']
#config_path = args['config_path']
#mode = 'train_e2e'

mode = 'train_ctc'

config_path = "tm_transformer_combined_lw_char.yaml"
config_file = f"../src/brainaudio/training/utils/custom_configs/{config_path}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
model_type = config['modelType']

model_args = config['model'][model_type]

if mode == 'train_e2e':
    return_final_layer = True
else:
    return_final_layer = False

if model_type == 'transformer':
    
    model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], 
                     heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'], 
                     dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                     max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], num_participants=len(model_args['features_list']), return_final_layer=return_final_layer)
    
if model_type == 'gru':
    
    model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'], 
                layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
                num_masks=config['num_masks'])
    
if mode == 'train_e2e':
    llm = AutoModelForCausalLM.from_pretrained(config['llm_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    # Load model and tokenizer
    if config["use_peft"]:
        peft_config = config["peft"]
        e2e_model = E2EModel(model, model_args['d_model'], llm, tokenizer, config["device"], peft_config)
    else:
        e2e_model = E2EModel(model, model_args['d_model'], llm, tokenizer, config["device"])
            
    e2e_model.to(config['device'])
    trainE2EModel(config, e2e_model)


elif mode == 'train_ctc':
    label = "phoneme" if config["nClasses"] == 40 else "char"
    model.to(config['device'])
    trainModel(config, model, label)

