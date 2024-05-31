import torch
from src.core.model import Log1DNetv3
from ignite.contrib.metrics.regression.r2_score import R2Score
from torch.nn import MSELoss
from torch.utils.data import DataLoader 
from torch.optim import Adam

def criterion(model:Log1DNetv3, dataloader:DataLoader, train:bool=False, optimizer:Adam=None):
    dtc_loss, dts_loss = MSELoss(), MSELoss()
    dtc_score, dts_score = R2Score(), R2Score()#.to('cuda:0')

    dtc_avg_loss = 0; dtc_avg_score = 0
    dts_avg_loss = 0; dts_avg_score = 0

    count = 0

    for input, output in iter(dataloader):
        predictions = model(input)
        #dtc
        dtc_loss_ = torch.sqrt(dtc_loss(predictions[:, 0], output[:, 0]))
        dtc_score.update([predictions[:, 0], output[:, 0]])
        dtc_score_ = dtc_score.compute()

        #dts
        dts_loss_ = torch.sqrt(dts_loss(predictions[:, 1], output[:, 1]))
        dts_score.update([predictions[:, 1], output[:, 1]])
        dts_score_ = dts_score.compute()

        if train:
            optimizer.zero_grad()
            dtc_loss_.backward(retain_graph=True)
            dts_loss_.backward()
            optimizer.step()

        dtc_avg_loss += dtc_loss_.item()
        dtc_avg_score += dtc_score_

        dts_avg_loss += dts_loss_.item()
        dts_avg_score += dts_score_

        count += 1

    return dtc_avg_loss/count, dtc_avg_score/count, dts_avg_loss/count, dts_avg_score/count
