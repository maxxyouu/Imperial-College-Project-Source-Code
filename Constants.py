from modulefinder import STORE_GLOBAL
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
if WORK_ENV == 'COLAB':
    DATA_PARENT_PATH = '/content/drive/MyDrive/CLEdata/'
    STORAGE_PATH = '/content/drive/MyDrive/'

DATA_MEAN = 0.1496
DATA_STD = 0.1960

