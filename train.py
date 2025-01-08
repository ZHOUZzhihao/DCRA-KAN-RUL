


import time
from model_DCRAKAN import *
import torch.nn as nn
from data_load import *
import numpy as np
from torch.utils.data import DataLoader
from utils.logger import init_logger
from torch.utils.tensorboard import SummaryWriter
import warnings
from tslearn.metrics import SoftDTWLossPyTorch

soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)  # gamma是Soft-DTW的平滑参数
warnings.filterwarnings("ignore")


def Training(opt):
## Will be offered unsolicited after article acceptance
    return
