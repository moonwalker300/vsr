import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch import nn
from network import p_t_z, q_z_t, DomainClassifer

def getDomainClassifer(q_z_t_dist, x, t, configs):
    classiferDimHidden = configs['classifer_dimhidden']
    classiferNHidden = configs['classifer_n_hidden']
    dc = DomainClassifer(x.shape[1], q_z_t_dist.dim_latent, classiferNHidden, classiferDimHidden)
    optimizer = optim.Adam(dc.parameters(), lr = 0.01)
    batch_size = configs['classifer_batchsize']
    epoch_num = configs['classifer_epochs']
    print_iter = 10
    n = x.shape[0]
    bceloss = nn.BCELoss(reduction='mean')
    for ep in range(epoch_num):
        idx = np.random.permutation(n)
        idx2 = np.random.permutation(n)
        cl_losses = []
        for i in range(0, n, batch_size):
            start, end = i, min(i + batch_size, n)
            x_batch = torch.FloatTensor(x[idx[start:end]])
            t_batch = torch.FloatTensor(t[idx[start:end]])
            z_batch_loc, z_batch_log_std = q_z_t_dist(t_batch)
            z_batch = z_batch_loc + torch.exp(z_batch_log_std) * torch.randn(size=z_batch_loc.size())
            z_batch = z_batch.detach()
            z_batch_neg = torch.randn(size=z_batch.size())

            x_batch = torch.cat((x_batch, x_batch), dim = 0)
            z_batch = torch.cat((z_batch, z_batch_neg), dim = 0)
            label_batch = torch.cat((torch.zeros(end - start, 1), torch.ones(end - start, 1)), dim = 0)

            pre_d = dc(x_batch, z_batch)
            loss = bceloss(pre_d, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cl_losses.append(loss.item())
        if ((ep + 1) % print_iter == 0):
            print('Epoch %d' % ep)
            print('Loss %f' % (sum(cl_losses) / len(cl_losses)))
    return dc

def trainVAE(t, n_t, configs):
    n = t.shape[0]
    dim_latent = configs['vae_dim_latent']
    n_hidden = configs['vae_n_hidden']
    dim_hidden = configs['vae_dim_hidden']
    p_t_z_dist = p_t_z(dim_latent, n_hidden, dim_hidden, n_t)
    q_z_t_dist = q_z_t(dim_latent, n_hidden, dim_hidden, n_t)
    print_iter = 10
    epoch_num = configs['vae_epochs']
    batch_size = configs['vae_batchsize']

    params = list(p_t_z_dist.parameters()) + list(q_z_t_dist.parameters())
    optimizer = optim.Adam(params, lr = 0.01)
    scheduler = StepLR(optimizer, step_size = epoch_num // 40, gamma=0.95)
    bceloss = nn.BCELoss(reduction='none')
    for epoch in range(epoch_num):
        idx = np.random.permutation(n)
        print_loss = []
        con_loss = []
        lat_loss = []
        for i in range(0, n, batch_size):
            start, end = i, min(i + batch_size, n)
            t_batch = torch.FloatTensor(t[idx[start:end]])

            z_infer_loc, z_infer_log_std = q_z_t_dist(t_batch)
            std_z = torch.randn(size=z_infer_loc.size())
            z_infer_sample = z_infer_loc + torch.exp(z_infer_log_std) * std_z

            t_infer_loc = p_t_z_dist(z_infer_sample)
            construct_loss = bceloss(t_infer_loc, t_batch).sum(1)
            latent_loss = (-z_infer_log_std + 1 / 2 * (
                        torch.exp(z_infer_log_std * 2) + z_infer_loc * z_infer_loc - 1)).sum(1)
            loss = torch.mean(construct_loss + latent_loss)

            print_loss.append(loss.detach().numpy())
            con_loss.append(torch.mean(construct_loss).detach().numpy())
            lat_loss.append(torch.mean(latent_loss).detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if ((epoch + 1) % print_iter == 0):
            print('Epoch %d' % epoch)
            print('Loss %f' % (sum(print_loss) / len(print_loss)))
            print('Construct Loss %f' % (sum(con_loss) / len(con_loss)))
            print('Latent Loss %f' % (sum(lat_loss) / len(lat_loss)))
    return q_z_t_dist, p_t_z_dist

def calcSampleWeights(x, t, q_z_t_dist, lat_clf):
    batch_size = 100
    n = x.shape[0]
    weight = np.zeros(n)
    for i in range(0, n, batch_size):
        start, end = i, min(i + batch_size, n)
        xi = torch.FloatTensor(x[start:end])
        ti = torch.FloatTensor(t[start:end])
        nums = 50
        z_batch_loc, z_batch_log_std = q_z_t_dist(ti)
        for j in range(nums):
            z_batch = z_batch_loc + torch.exp(z_batch_log_std) * torch.randn(size=z_batch_loc.size())
            pre_d = lat_clf(xi, z_batch)
            pre_d = pre_d.detach().numpy().squeeze()
            weight[start:end] += ((1 - pre_d) / pre_d) / nums
        weight[start:end] = 1 / weight[start:end]
    weight /= weight.mean()
    return weight

def calcVSR(x, t, configs):
    vaeName = configs['vae_name']
    if (configs['use_pre_vae']):
        p_t_z_dist = torch.load(vaeName + '_p_t_z.mdl')
        q_z_t_dist = torch.load(vaeName + '_q_z_t.mdl')
        lc = torch.load(vaeName + '_dc.mdl')
    else:
        q_z_t_dist, p_t_z_dist = trainVAE(t, t.shape[1], configs)
        lc = getDomainClassifer(q_z_t_dist, x, t, configs)
        torch.save(p_t_z_dist, vaeName + '_p_t_z.mdl')
        torch.save(q_z_t_dist, vaeName + '_q_z_t.mdl')
        torch.save(vaeName + '_dc.mdl', lc)
    w = calcSampleWeights(x, t, q_z_t_dist, lc)
    return w