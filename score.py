import numpy as np
from sklearn.metrics import mean_squared_error as mse

alpha = np.array(4 * [1.5] + 7 * [2] + 7 * [3] + 6 * [4])


def get_score(true, pred):
    true_mean = np.mean(true, axis=0)
    pred_mean = np.mean(pred, axis=0)
    true_std = np.std(true, axis=0)
    pred_std = np.std(pred, axis=0)
    sigma = true_std * pred_std * true.shape[0]
    cor = np.sum((true - true_mean) * (pred - pred_mean), axis=0) / (sigma + 1e-8)
    rmse = np.sqrt(mse(true, pred))
    accskill = 0
    for i in range(24):
        accskill += alpha[i] * np.log(i + 1) * cor[i]
    score = (2 / 3) * accskill - rmse
    return score

def get_score1(true, pred):
    pred_mean = np.mean(pred, 0)
    true_mean = np.mean(true, 0)
    N = pred.shape[0]
    pred_m = []
    true_m = []
    for i in range(N):
        pred_m.append(pred_mean)
        true_m.append(true_mean)
    pred_mean = np.array(pred_m)
    true_mean = np.array(true_mean)
    sigma = np.sqrt(np.sum((pred - pred_mean) ** 2, 0) * np.sum((true - true_mean) ** 2, 0))
    print(sigma)
    cor = np.sum((pred - pred_mean) * (true - true_mean), 0) / (sigma + 1e-8)
    print(cor)
    rmse = np.sqrt(np.sum((true - pred) ** 2.0) / N)
    RMSE = np.sum(rmse)
    print(RMSE)
    accskill = 0
    for i in range(24):
        accskill += alpha[i] * np.log(i + 1) * cor[i]
    score = 2/3 * accskill - RMSE
    print(accskill)
    return score
