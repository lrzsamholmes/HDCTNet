import numpy as np
import ml_collections

# Dataset length specification
# Healthy, UUO, Adenine, Alport, IRI, NTN
testDatasetsSizes = np.array([160, 60, 60, 60, 60, 60])
diseaseModels = ['HEALTHY', 'UUO', 'ADENINE', 'ALPORT', 'IRI', 'NTN']

useAug = False
useSmallImg = False
useSmallEmb = False
if useSmallImg:
    img_size = 320
else:
    img_size = 640
n_channels = 3
n_labels = 8
datasets = 'Data_Original'
image_dir_base = '<change path>'+datasets

LR_Reduce_No_Train_Improvement = 30
LR_Reduce_No_Val_Improvement = 30
EARLY_STOP_LR_TOO_LOW = 1e-4
divideLRfactor=3.0

saveFinalTestResults = True

# CTrans configs
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    if useSmallEmb:
        config.patch_sizes = [32,16,8,4]
    else:
        config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    return config

# DCTrans configs
def get_DCTrans_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 1472  # KV_size = Q1 + Q2 + Q3 + Q4 + Q5
    config.transformer.num_heads  = 6
    config.transformer.num_layers = 6
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.5
    config.transformer.attention_dropout_rate = 0.5
    config.transformer.dropout_rate = 0.5
    config.offset_conv_k = 5
    config.offset_range_factor = 5
    if useSmallEmb:
        config.patch_sizes = [32,16,8,4,2]
    else:
        config.patch_sizes = [16,8,4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.activation = 'GELU' # activation of U-Net
    return config