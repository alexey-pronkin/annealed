import torch.nn as nn
import torch
from .refinement_operator_module import RefinementOperatorModule


class AVOTransition(nn.Module):
    def __init__(self, input_dimension, initial_target, final_target, alpha):
        super().__init__()
        self._initial_target = initial_target
        self._final_target = final_target
        self._alpha = alpha

        self._forward_module = RefinementOperatorModule(input_dimension)
        self._reverse_module = RefinementOperatorModule(input_dimension)

    def log_forward_transition(self, z, previous_z):
        mu, sigma = self._forward_module(previous_z)
        return torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(z), dim=1)

    def log_reverse_transition(self, z, previous_z):
        mu, sigma = self._reverse_module(z)
        return torch.sum(torch.distributions.normal.Normal(mu, sigma).log_prob(previous_z), dim=1)

    def conditional_sample(self, previous_z):
        mu, sigma = self._forward_module(previous_z)
        epsilon = torch.randn_like(previous_z)
        z = mu + epsilon * sigma
        return z

    def log_annealed_distribution(self, z):
        return -(1 - self._alpha) * self._initial_target.E(z) - self._alpha * self._final_target.E(z)

    def forward(self, previous_z, log_previous_probability):
        previous_z = previous_z.detach()
        log_previous_probability = log_previous_probability.detach()
        z = self.conditional_sample(previous_z)

        log_forward = self.log_forward_transition(z, previous_z)
        log_reverse = self.log_reverse_transition(z, previous_z)
        log_annealed = self.log_annealed_distribution(z)

        # Because log_previous_probability is detached, it is not needed in loss function
        loss = -log_annealed - log_reverse + log_forward + log_previous_probability
        log_probability = log_previous_probability + log_forward - log_reverse
        return loss, z, log_probability
