import torch.nn as nn
from .vae import VAE
import torch
import numpy as np
import torch.nn


class VaeGaussian(VAE):
    def __init__(self, input_dimension=28, latent_dimension=40, hidden_dimensions=(300, 300)):
        super().__init__(input_dimension, latent_dimension, hidden_dimensions)
        self._mu_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)
        self._logvar_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)

    def generate_z(self, x):
        hidden_x = self.encoder(x)
        z_mu = self._mu_linear(hidden_x)
        z_logvar = self._logvar_linear(hidden_x)
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
        return z, z_mu, z_logvar

    def forward(self, x):
        z, z_mu, z_logvar = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        scale = x.size()[0]
        kl_part = self.kl(z_mu, z_logvar) / scale
        x: torch.Tensor
        nll_part = self.nll_part_loss(reconstructed_x, x) / scale
        beta = self.calculate_beta()
        loss = kl_part * beta + nll_part
        # entropy = -torch.sum(0.5 * (np.log(2 * np.pi) + 1 + z_logvar))
        return reconstructed_x, loss, nll_part, kl_part

    def log_importance_weight(self, x):
        z, z_mu, z_logvar = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        log_p_x = -torch.sum(self._sample_loss(reconstructed_x, x), dim=[1, 2, 3])
        log_p_z = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(z), dim=1)
        log_q_z = torch.sum(torch.distributions.normal.Normal(z_mu, torch.exp(0.5 * z_logvar)).log_prob(z),
                            dim=1)
        return log_p_x + log_p_z - log_q_z

    @staticmethod
    def kl(z_mean, z_logvar):
        kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
        return kl_divergence_element.sum()
