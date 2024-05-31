import torch
from pandas import DataFrame
from torch.utils.data import (
    DataLoader, TensorDataset
                              )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dloader(x, y, bs=64):

    inputs = torch.from_numpy(x.values).to(device).float()
    outputs = torch.from_numpy(y.values).to(device).float()
    tensor = TensorDataset(inputs, outputs)

    loader = DataLoader(tensor, bs, shuffle=True, drop_last=True)

    return loader
