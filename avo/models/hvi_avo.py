import pytorch_lightning as pl
import torch
import numpy as np
from .hvi_transition import HVITransition
from .hvi import HVI
from ..toy_dist import SimpleNormal
import torch.nn as nn

from ..toy_dist import ToyD

class AnnealedTarget(ToyD):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self.name = 'four-mode mixture annealed'
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = ToyD.E(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann

class HVIAVO(HVI):
    def __init__(self, input_dimension, depth, targets=AnnealedTarget, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, targets[-1], lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - input dimension
        h_dim - hidden dimension of transition
        depth - number of transitions
        """
        self._depth = depth
        self._transitions = nn.ModuleList([HVITransition(input_dimension, hidden_dimension) for _ in range(depth)])
        
        self._initial_target = SimpleNormal(targets[-1].device)
        self._transitions_targets = transitions_targets = [
            targets(i) for i in np.linspace(0., 1., depth)]

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        previous_log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for i, target in enumerate(self._transitions_targets):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = self._transitions[i](x)
            annealed_loss = target.E(x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth

