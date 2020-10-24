import torch
import numpy as np
from .hvi_transition import HVITransition
from .hvi import HVI
from ..toy_dist import SimpleNormal
import torch.nn as nn


class HVIAVO(HVI):
    def __init__(self, input_dimension, depth, target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100, transitions=None, is_flow=False):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k, is_flow)
        """
        in_dim - input dimension
        h_dim - hidden dimension of transition
        depth - number of transitions
        """
        self._depth = depth
        self._alphas = np.linspace(1. / depth, 1., depth)
        self._initial_target = SimpleNormal(target.device)
        if transitions is None:
            self._transitions = nn.ModuleList([HVITransition(input_dimension, hidden_dimension) for _ in range(depth)])
        else:
            self._transitions = nn.ModuleList(transitions)

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