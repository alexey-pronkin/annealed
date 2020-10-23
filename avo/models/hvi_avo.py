import pytorch_lightning as pl
import torch
import numpy as np
from .hvi_transition import HVITransition
from .hvi import HVI
from ..toy_dist import SimpleNormal
import torch.nn as nn

from ..toy_dist import ToyD


class AnnealedTarget(ToyD):
    def __init__(self, alpha=0, device="cpu"):
        super().__init__()
        self.name = "four-mode mixture annealed"
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = ToyD.E(x)
        ann = (1.0 - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class HVIAVO(HVI):
    def __init__(
        self,
        input_dimension,
        depth,
        target,
        lr=1e-3,
        batch_size=64,
        hidden_dimension=2,
        beta1=0.999,
        beta2=0.999,
        optimizer="adam",
        k=100,
        transitions=None,
        is_flow=False,
    ):
        super().__init__(
            input_dimension=input_dimension,
            target=target,
            lr=lr,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            k=k,
            is_flow=is_flow,
        )
        """
        in_dim - input dimension
        h_dim - hidden dimension of transition
        depth - number of transitions
        """
        self._depth = depth
        self._alphas = np.linspace(1.0 / depth, 1.0, depth)
        self._initial_target = SimpleNormal(target.device)
        self.normal = torch.distributions.normal.Normal(0, 1)
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
        previous_log_probability = torch.sum(self.normal.log_prob(x), dim=1)
        loss = torch.sum(self.normal.log_prob(x), dim=1)
        for i, target in enumerate(self._transitions_targets):
            previous_log_probability = previous_log_probability.detach()
            x = x.detach()
            x, log_probability = self._transitions[i](x)
            annealed_loss = target.E(x)
            loss += annealed_loss + log_probability + previous_log_probability
            previous_log_probability += log_probability
        return x, loss / self._depth

