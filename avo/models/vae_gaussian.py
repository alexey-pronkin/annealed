import torch.nn as nn
from .vae import VAE
import torch
import torch.nn


class VaeGaussian(VAE):
    def __init__(self, input_dimension=28, latent_dimension=40, hidden_dimensions=(300, 300)):
        super().__init__(input_dimension, latent_dimension, hidden_dimensions)
        self._mu_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)
        self._logvar_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)

    def forward(self, x):
        hidden_x = self.encoder(x)
        z_mu = self._mu_linear(hidden_x)
        z_logvar = self._logvar_linear(hidden_x)
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon

        reconstructed_x = self.decoder(z)
        scale = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        kl_part = self.kl(z_mu, z_logvar) / scale
        x: torch.Tensor
        nll_part = self.nll_part_loss(reconstructed_x, x) / scale
        loss = kl_part + nll_part
        self.log("kl_part", kl_part)
        self.log("nll_part", nll_part)
        self.log("loss", loss)
        return reconstructed_x, loss

    @staticmethod
    def kl(z_mean, z_logvar):
        kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
        return kl_divergence_element.sum()
