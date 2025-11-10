import yaml
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.transformer_interctc import TransformerModel 
# from brainaudio.models.transformer_chunking import TransformerModel
from brainaudio.training.trainer_interctc import trainModel
# from brainaudio.training.trainer import trainModel


config_path = "tm_transformer_b2t25_interctc_test_run.yaml"
config_file = f"../src/brainaudio/training/utils/custom_configs/{config_path}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    
model_type = config['modelType']

model_args = config['model'][model_type]

return_final_layer = False

if model_type == 'transformer':

    # model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'], 
    #                  dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
    #                  max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], num_participants=len(model_args['features_list']), return_final_layer=return_final_layer, 
    #                  chunked_attention=model_args["chunked_attention"])

    model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'], 
                     dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                     max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], num_participants=len(model_args['features_list']), return_final_layer=return_final_layer, 
                     bidirectional=model_args['bidirectional'], inter_ctc_per_layers=config['interctc']['inter_ctc_per_layers'])
    
if model_type == 'gru':
    
    model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'], 
                layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
                strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
                num_masks=config['num_masks'])
    
model.to(config['device'])
trainModel(config, model)


