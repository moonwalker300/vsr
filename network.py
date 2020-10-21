import torch
from torch import nn
import torch.nn.functional as F

class p_t_z(nn.Module):
    def __init__(self, dim_latent, n_hidden, dim_hidden, n_t):
        super(p_t_z, self).__init__()
        self.n_hidden = n_hidden
        self.input_net = nn.Linear(dim_latent, dim_hidden)
        self.hidden_net = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1)])
        self.treatment_net = nn.Linear(dim_hidden, n_t)
    def forward(self, z):
        z = F.elu(self.input_net(z))
        for i in range(self.n_hidden - 1):
            z = F.elu(self.hidden_net[i](z))

        t = torch.sigmoid(self.treatment_net(z))
        #t = F.log_softmax(self.treatment_net(z), dim=1)
        return t

class q_z_t(nn.Module):
    def __init__(self, dim_latent, n_hidden, dim_hidden, n_t):
        super(q_z_t, self).__init__()
        self.n_hidden = n_hidden
        self.dim_latent = dim_latent
        self.input_net_t = nn.Linear(n_t, dim_hidden)
        self.hidden_net_t = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1)])

        self.zt_net_loc = nn.Linear(dim_hidden, dim_latent)
        self.zt_net_log_std = nn.Linear(dim_hidden, dim_latent)

    def forward(self, t):
        zt = F.elu(self.input_net_t(t))
        for i in range(self.n_hidden - 1):
            zt = F.elu(self.hidden_net_t[i](zt))
        zt_loc = self.zt_net_loc(zt)
        zt_log_std = self.zt_net_log_std(zt)
        return zt_loc, zt_log_std

class y_predictor2(nn.Module):
    def __init__(self, dim_x, dim_t, dim_hidden_t, dim_hidden_x, dim_hidden, n_hidden):
        super(y_predictor2, self).__init__()
        self.n_hidden = n_hidden
        self.input_net_t = nn.Linear(dim_t, dim_hidden_t)
        self.input_net_x = nn.Linear(dim_x, dim_hidden_x)
        hidden_units = [dim_hidden_x + dim_hidden_t] + [dim_hidden] * n_hidden
        self.hidden_net = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(n_hidden)])
        self.output_net = nn.Linear(dim_hidden, 1)

    def forward(self, x, t):
        zt = F.elu(self.input_net_t(t))
        zx = F.elu(self.input_net_x(x))
        zxt = torch.cat((zx, zt), dim = 1)
        for i in range(self.n_hidden):
            zxt = F.elu(self.hidden_net[i](zxt))

        y = self.output_net(zxt)
        return y

class y_predictor(nn.Module):
    def __init__(self, dim_x, dim_t, dim_latent_t, dim_hidden, n_hidden):
        super(y_predictor, self).__init__()
        self.n_hidden = n_hidden
        self.input_net_t = nn.Linear(dim_t, dim_latent_t)
        self.input_net_x = nn.Linear(dim_x, dim_x)
        self.input_net_xt = nn.Linear(dim_x + dim_latent_t, dim_hidden)
        self.hidden_net = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1)])
        self.output_net = nn.Linear(dim_hidden, 1)

    def forward(self, x, t):
        zt = F.elu(self.input_net_t(t))
        zx = F.elu(self.input_net_x(x))
        zxt = torch.cat((zx, zt), dim = 1)
        zy = F.elu(self.input_net_xt(zxt))
        for i in range(self.n_hidden - 1):
            zy = F.elu(self.hidden_net[i](zy))

        y = self.output_net(zy)
        return y

class DomainClassifer(nn.Module):
    def __init__(self, dim_x, dim_z, n_hidden, dim_hidden):
        super(DomainClassifer, self).__init__()
        self.n_hidden = n_hidden
        self.input_net = nn.Linear(dim_x + dim_z, dim_hidden)
        self.hidden_net = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1)])
        self.output_net = nn.Linear(dim_hidden, 1)
    def forward(self, x, z):
        zx = torch.cat((x, z), 1)
        zy = F.elu(self.input_net(zx))
        for i in range(self.n_hidden - 1):
            zy = F.elu(self.hidden_net[i](zy))
        y = torch.sigmoid(self.output_net(zy))
        return y