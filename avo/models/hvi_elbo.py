import torch
import torch.nn as nn

from .hvi import HVI
from .hvi_transition import HVITransition


class HVIELBO(HVI):
    def __init__(self, input_dimension, depth, target, lr=1e-3, batch_size=64, hidden_dimension=2, beta1=0.999,
                 beta2=0.999, optimizer="adam", k=100):
        super().__init__(input_dimension, target, lr, batch_size, beta1, beta2, optimizer, k)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._transitions = nn.ModuleList([HVITransition(input_dimension, hidden_dimension) for _ in range(depth)])

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        loss = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for transition in self._transitions:
            x, log_probability = transition(x)
            loss += log_probability
        loss += self._target.E(x)
        return x, loss
