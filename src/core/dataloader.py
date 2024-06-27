import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import (
    DataLoader, TensorDataset
                              )
from torch.utils.data import (
    DataLoader, Dataset
                              )
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dloader(x, y, bs=64):

    inputs = torch.from_numpy(x.values).to(device).float()
    outputs = torch.from_numpy(y.values).to(device).float()
    tensor = TensorDataset(inputs, outputs)

    loader = DataLoader(tensor, bs, shuffle=True, drop_last=True)

    return loader

class LogDataset(Dataset):
    def __init__(self, x:DataFrame, y:DataFrame):
        self.x = torch.from_numpy(x.values)
        self.y = torch.from_numpy(y.values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].to(device).float()
        y = self.y[idx].to(device).float()

        return x, y

def dataloader(x, y, bs:int=32, train:bool=True, val_size:float=0.18) -> DataLoader:

    tensor = LogDataset(x, y)

    # obtain training indices that will be used for validation
    np.random.seed(2024)
    num_train = len(tensor)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_train))
    train_idx, val_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    if train:
        train_loader = DataLoader(tensor, bs,
                                  sampler=SubsetRandomSampler(train_idx),
                                  drop_last=True)
        return train_loader
    else:
        val_loader = DataLoader(tensor, bs,
                                sampler=SubsetRandomSampler(val_idx),
                                drop_last=True)
        return val_loader
