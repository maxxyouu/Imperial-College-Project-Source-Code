import torch


USE_GPU = True
# training device
DTYPE = torch.float32
DEVICE = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')

# WORK_ENV = 'COLAB'
WORK_ENV = 'LOCAL'
DATA_PARENT_PATH = '../'
STORAGE_PATH = './'
SAVED_MODEL_PATH = './trained_models'
if WORK_ENV == 'COLAB':
    DATA_PARENT_PATH = '/content/drive/MyDrive/CLEdata/'
    STORAGE_PATH = '/content/drive/MyDrive/'
    SAVED_MODEL_PATH = '/content/drive/MyDrive/trained_models'

DATA_MEAN = 0.1496
DATA_STD = 0.1960

SIMCLR_MODEL_PATH = 'SupCon_models/path_models/SimCLR_path_skresnext50_32x4d_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine'
