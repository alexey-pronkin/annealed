import torch.nn as nn
import torch
from .refinement_operator_module import RefinementOperatorModule


class HVITransition(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=2, activation="ReLU", context_dimension=0):
        super().__init__()
        self._forward_module = RefinementOperatorModule(
            input_dimension,
            hidden_dimension,
            activation,
            context_dimension=context_dimension
        )
        self._reverse_module = RefinementOperatorModule(
            input_dimension,
            hidden_dimension,
            activation,
            context_dimension=context_dimension
        )
    def log_forward_transition(self, previous_z, z=None, x=None):
        mu, sigma = self._forward_module(previous_z, x)
        if z is None:
            epsilon = torch.randn_like(previous_z)
            z = mu + epsilon * sigma
        return z, torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(z), dim=1)

    def log_reverse_transition(self, z, previous_z=None, x=None):
        mu, sigma = self._reverse_module(z, x)
        if previous_z is None:
            epsilon = torch.randn_like(z)
            previous_z = mu + epsilon * sigma
        return previous_z, torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(previous_z), dim=1)

    def forward(self, previous_z, x=None):
        z, log_forward = self.log_forward_transition(previous_z, x=x)
        previous_z, log_reverse = self.log_reverse_transition(z, previous_z, x=x)
        return z, log_forward - log_reverse

    def forward_reverse(self, z):
        previous_z, log_reverse = self.log_reverse_transition(z)
        z, log_forward = self.log_forward_transition(previous_z, z)
        return previous_z, log_forward - log_reverse
