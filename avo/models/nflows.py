import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pyro.nn import AutoRegressiveNN

from ..toy_dist import SimpleNormal
from .hvi import HVI
from .hvi_transition import HVITransition
from .utils import get_activation


class IATransform(nn.Module):
    def __init__(self, in_dim, h_dim, rand_perm=True, activation='ReLU', num_stable=False):
        super(IATransform, self).__init__()
        if rand_perm:
            self.permutation = torch.randperm(in_dim, dtype=torch.int64)
            nonlinearity = get_activation(activation)
        else:
            self.permutation = torch.arange(in_dim, dtype=torch.int64)
        self.AR = AutoRegressiveNN(input_dim=in_dim, hidden_dims=h_dim, nonlinearity=nonlinearity) #nn.ELU(0.6))
        self.num_stable = num_stable
    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        m, log_s = self.AR(x)
        if self.num_stable:
            s = torch.sigmoid(torch.exp(log_s))
            x = (1-s) * m + s * x#[:, self.permutation]
            log_det = torch.log(s).sum(axis=1)# - 0.000001
        else:
            x = torch.exp(log_s) * x + m
            log_det = log_s.sum(axis=1)
        return x, -log_det

class IAF(nn.Module):
    def __init__(self, in_dim, h_dim, depth, activation="ReLU", num_stable=False):
        super(IAF, self).__init__()
        """
        in_dim - inout dimention
        h_dim - hidden dimention of the autoregressive NN
        depth - number of IA transformations
        """
        # YOUR CODE HERE
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.depth = depth
        self.IAFs = nn.ModuleList([
            IATransform(in_dim=in_dim, h_dim=h_dim, rand_perm=True, activation=activation, num_stable=num_stable) for i in range(depth)
        ])
    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        # YOUR CODE HERE
        log_det = 0
        for i, AF in enumerate(self.IAFs):
            x, d = AF(x)
            log_det += d
        return x, log_det

class Householder(nn.Module):
    def __init__(self, v_dim, activation="ReLU", num_stable=False):
        super(Householder, self).__init__()
        """
        in_dim = v_dim
        in_dim - inout dimention
        h_dim - hidden dimention of the autoregressive NN
        depth - number of IA transformations
        """
        self.v_dim = v_dim
        self.v = nn.Parameter(torch.Tensor(v_dim))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        torch.init.kaiming_uniform_(self.v, a=math.sqrt(5))
    def forward(self, z, v=None):
        """
        Return transformed sample and log determinant
        Note v could be just learning parameter or NN, that uses hidden information, so if v is not None we use it.
        """
        batch, _ = z.shape
        if v is None:
            v = self.v.repeat(batch, 1)
        else:
            v = v.repeat(batch, 1)
        if num_stable:
            vvT = torch.bmm( v.unsqueeze(2), v.unsqueeze(1) )
            vvTz = torch.bmm( vvT, z.unsqueeze(2) ).squeeze(2)
        else:  
            # v * v_T
            vvT = (v * v).sum(1) # memory inefficient, however fast, could use author instead  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
            # v * v_T * z
            vvTz =  (vvT * z).sum(1)  # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum( v * v, 1 ) # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand( norm_sq.size(0), v.size(1) ) # expand sizes : B x L
        # calculate new z
        z = z - 2 * vvTz / norm_sq # z - 2 * v * v_T  * z / norm2(v)
        log_det = 0
        return z, log_det

class HouseholderELBO(HVI):
    def __init__(self, input_dimension, depth, AR_depth, target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._depth = depth
        self._alphas = np.linspace(1. / depth, 1., depth)
        self._initial_target = SimpleNormal(target.device)
        self._transitions = nn.ModuleList([Householder(v_dim=input_dimension) for _ in range(depth)])

    def annealed_loss(self, alpha, z):
        return (1 - alpha) * self._initial_target.E(z) + alpha * self._target.E(z)

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        previous_log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for alpha, transition in zip(self._alphas, self._transitions):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = transition(x)
            annealed_loss = self.annealed_loss(alpha, x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth

class IAFELBO(HVI):
    def __init__(self, input_dimension, depth, AR_depth, target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._depth = depth
        self._alphas = np.linspace(1. / depth, 1., depth)
        self._initial_target = SimpleNormal(target.device)
        self._transitions = nn.ModuleList([IAF(in_dim=input_dimension, h_dim=hidden_dimension, depth=AR_depth) for _ in range(depth)])

    def annealed_loss(self, alpha, z):
        return (1 - alpha) * self._initial_target.E(z) + alpha * self._target.E(z)

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        previous_log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for alpha, transition in zip(self._alphas, self._transitions):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = transition(x)
            annealed_loss = self.annealed_loss(alpha, x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth

class HouseholderAVO(HVI):
    def __init__(self, input_dimension, depth, target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._depth = depth
        self._alphas = np.linspace(1. / depth, 1., depth)
        self._initial_target = SimpleNormal(target.device)
        self._transitions = nn.ModuleList([Householder(v_dim=input_dimension) for _ in range(depth)])

    def annealed_loss(self, alpha, z):
        return (1 - alpha) * self._initial_target.E(z) + alpha * self._target.E(z)

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        previous_log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for alpha, transition in zip(self._alphas, self._transitions):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = transition(x)
            annealed_loss = self.annealed_loss(alpha, x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth

class IAFAVO(HVI):
    def __init__(self, input_dimension, depth, AR_depth,  target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._depth = depth
        self._alphas = np.linspace(1. / depth, 1., depth)
        self._initial_target = SimpleNormal(target.device)
        self._transitions = nn.ModuleList([IAF(in_dim=input_dimension, h_dim=hidden_dimension, depth=AR_depth) for _ in range(depth)])

    def annealed_loss(self, alpha, z):
        return (1 - alpha) * self._initial_target.E(z) + alpha * self._target.E(z)

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        previous_log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for alpha, transition in zip(self._alphas, self._transitions):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = transition(x)
            annealed_loss = self.annealed_loss(alpha, x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth
