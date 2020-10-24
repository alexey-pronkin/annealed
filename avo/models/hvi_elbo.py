import torch
import torch.nn as nn

from .hvi import HVI
from .hvi_transition import HVITransition


class HVIELBO(HVI):
    def __init__(self, input_dimension, depth, target, lr=1e-3, batch_size=64,
         hidden_dimension=2, beta1=0.999, beta2=0.999, optimizer="adam", k=100, 
          transitions=None, is_flow=False):
        super().__init__(
            input_dimension=input_dimension, target=target, lr=lr, batch_size=batch_size, beta1=beta1, beta2=beta2,
             optimizer=optimizer, k=k, is_flow=is_flow,)
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._transitions = nn.ModuleList([HVITransition(input_dimension, hidden_dimension) for _ in range(depth)])
        if transitions is None:
            self._transitions = nn.ModuleList([HVITransition(input_dimension, hidden_dimension) for _ in range(depth)])
        else:
            self._transitions = nn.ModuleList(transitions)

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
