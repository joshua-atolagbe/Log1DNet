from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt

def metrics(pred, gt, name):
    print(f'The R2 score for Well {name}: {r2_score(pred, gt)}')
    print(f'The RMSE for Well {name}: {sqrt(mean_squared_error(pred, gt))}')
    print('-'*50)
