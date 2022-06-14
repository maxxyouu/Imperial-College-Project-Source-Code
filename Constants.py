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

