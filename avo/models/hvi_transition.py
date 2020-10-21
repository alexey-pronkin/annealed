import torch.nn as nn
import torch
from .refinement_operator_module import RefinementOperatorModule


class HVITransition(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=2):
        super().__init__()
        self._forward_module = RefinementOperatorModule(input_dimension, hidden_dimension)
        self._reverse_module = RefinementOperatorModule(input_dimension, hidden_dimension)

    def log_forward_transition(self, previous_z, z=None):
        mu, sigma = self._forward_module(previous_z)
        if z is None:
            epsilon = torch.randn_like(previous_z)
            z = mu + epsilon * sigma
        return z, torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(z), dim=1)

    def log_reverse_transition(self, z, previous_z=None):
        mu, sigma = self._reverse_module(z)
        if previous_z is None:
            epsilon = torch.randn_like(z)
            previous_z = mu + epsilon * sigma
        return previous_z, torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(previous_z), dim=1)

    def forward(self, previous_z):
        z, log_forward = self.log_forward_transition(previous_z)
        previous_z, log_reverse = self.log_reverse_transition(z, previous_z)
        return z, log_forward - log_reverse

    def forward_reverse(self, z):
        previous_z, log_reverse = self.log_reverse_transition(z)
        z, log_forward = self.log_forward_transition(previous_z, z)
        return previous_z, log_forward - log_reverse
