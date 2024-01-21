"""
Example code for RNN, LSTM and GRU on MNIST dataset
"""

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

# Hyperparameter settings
