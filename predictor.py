import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch import nn
import numpy as np
from network import y_predictor, y_predictor2
def TrainPredictor(x, t, y, w, configs):
    predictor = y_predictor(x.shape[1], t.shape[1], configs['pre_dim_hidden_t'], configs['pre_dim_hidden'], configs['pre_n_hidden'])
    optimizer = optim.Adam(predictor.parameters(), lr = configs['pre_lr'], weight_decay = configs['l2_norm'])
    epoch_num = configs['pre_epochs']
    ss = configs['pre_lr_decay_step']
    gm = configs['pre_lr_decay_gamma']
    scheduler = StepLR(optimizer, step_size = ss, gamma = gm)
    batch_size = configs['pre_batchsize']
    MSEloss = torch.nn.MSELoss(reduction='none')
    n = x.shape[0]
    print_iter = 10
    for epoch in range(epoch_num):
        idx = np.random.permutation(n)
        y_loss = []
        for i in range(0, n, batch_size):
            start, end = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[idx[start:end]])
            t_batch = torch.FloatTensor(t[idx[start:end]])
            y_batch = torch.FloatTensor(y[idx[start:end]])

            w_batch = torch.FloatTensor(w[idx[start:end]])
            pre_y = predictor(x_batch, t_batch)
            target = y_batch.view(-1, 1)
            w_batch = w_batch.view(-1, 1)
            loss = (MSEloss(pre_y, target) * w_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_loss.append(loss.item())
        if ((epoch + 1) % print_iter == 0):
            print('Epoch %d' % epoch)
            print('Y Loss %f' % (sum(y_loss) / len(y_loss)))
        scheduler.step()

    return predictor