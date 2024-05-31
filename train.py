import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from numpy import Inf
from pathlib import Path
from torch.optim import Adam, SGD, Adagrad
from src import criterion, Log1DNetv3

def main(model:Log1DNetv3, trainloader:DataLoader, testloader:DataLoader, epochs):
    
    train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score = list(), list(), list(), list()
    val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score = list(), list(), list(), list()

    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_loss = Inf

    for epoch in tqdm(range(epochs)):
        tn_dtc_avg_loss, tn_dtc_avg_score, tn_dts_avg_loss, tn_dts_avg_score = criterion(model, trainloader, train=True, optimizer=optimizer)
        tt_dtc_avg_loss, tt_dtc_avg_score, tt_dts_avg_loss, tt_dts_avg_score = criterion(model, testloader, train=False, optimizer=optimizer)
        #train
        train_dtc_loss.append(tn_dtc_avg_loss); train_dtc_score.append(tn_dtc_avg_score)
        train_dts_loss.append(tn_dts_avg_loss); train_dts_score.append(tn_dts_avg_score)
        #val
        val_dtc_loss.append(tt_dtc_avg_loss); val_dtc_score.append(tt_dtc_avg_score)
        val_dts_loss.append(tt_dts_avg_loss); val_dts_score.append(tt_dts_avg_score)

        if (tt_dtc_avg_loss+tt_dts_avg_loss) < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print('Model Saved')

            best_val_loss = tt_dtc_avg_loss+tt_dts_avg_loss

        print(f'Epoch {epoch+1} Train: DTC Loss={tn_dtc_avg_loss:.4f} | DTC Score={tn_dtc_avg_score:.4f} | DTS Loss={tn_dts_avg_loss:.4f} | DTS Score={tn_dts_avg_score:.4f}')
        print(f'Epoch {epoch+1} Val: DTC Loss={tt_dtc_avg_loss:.4f} | DTC Score={tt_dtc_avg_score:.4f} | DTS Loss={tt_dts_avg_loss:.4f} | DTS Score={tt_dts_avg_score:.4f}')
        print()


    return train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score, val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score


