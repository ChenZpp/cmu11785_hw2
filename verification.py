import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from dataloader import *
from train import *
from models import *

#train_dataset = verification_train_dataset()
#train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

vali_dataset = verification_validation_dataset()
vali_loader = DataLoader(vali_dataset, batch_size=10, num_workers=1, shuffle=False)

test_dataset = verification_test_dataset()
test_loader = DataLoader(vali_dataset, batch_size = 10, num_workers=1, shuffle=False)

