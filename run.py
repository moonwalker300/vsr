import argparse
from data import CF_Dataset
from loader import load_config
from vsr import calcVSR
from predictor import TrainPredictor
import numpy as np
import torch

def RMSE(y1, y2):
    return np.sqrt(np.mean((y1 - y2) * (y1 - y2)))
def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str)
    return parser.parse_args()

if (__name__ == '__main__'):
    args = init_arg()
    configs = load_config(args.config_file)
    data = CF_Dataset(configs['train_file'], configs['test_file'], configs['if_train_noise'], configs['train_noise_std'])
    data.switch('Train')
    n, n_t, n_x = data.getBasicInfo()
    x, t, y = data.getData()

    w = calcVSR(x, t, configs)
    print(1 / np.sum(np.square(w / n)))
    data.switch('Test')
    test_n, n_t, n_x = data.getBasicInfo()
    test_x, test_t, test_y = data.getData()
    rmses = []
    for iter in range(configs['repetitions']):
        pre = TrainPredictor(x, t, y, w, configs)
        pre_test_y = np.zeros(test_n)
        for i in range(test_n):
            pre_test_x = torch.FloatTensor(test_x[i:i + 1])
            pre_test_t = torch.FloatTensor(test_t[i:i + 1])
            pre_test_y[i] = pre(pre_test_x, pre_test_t).item()
        rmse = RMSE(pre_test_y, test_y)
        rmses.append(rmse)
        print('Repetitions %d RMSE: %f' % (iter, rmse))

    rmses = np.array(rmses)
    print('Test RMSE Mean:', rmses.mean())
    print('Test RMSE STD:', rmses.std())