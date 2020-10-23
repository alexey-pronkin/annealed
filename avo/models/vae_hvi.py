import torch.nn as nn
from .vae import VAE
import torch
import torch.nn
from .hvi_transition import HVITransition


class VAEHVI(VAE):
    def __init__(self, input_dimension=28, latent_dimension=40, hidden_dimensions=(300, 300), depth=5,
                 transition_hidden_dimension=40, transitions=None, beta=1, gamma=0):
        super().__init__(input_dimension, latent_dimension, hidden_dimensions, beta=beta, gamma=gamma)
        self._mu_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)
        self._logvar_linear = nn.Linear(hidden_dimensions[-1], latent_dimension)
        if transitions is None:
            self._transitions = nn.ModuleList([HVITransition(latent_dimension, transition_hidden_dimension,
                                          "LeakyReLU", hidden_dimensions[-1]) for _ in range(depth)])
        else:
            self._transitions = nn.ModuleList(transitions)

    def generate_z(self, x):
        hidden_x = self.encoder(x)
        z_mu = self._mu_linear(hidden_x)
        z_logvar = self._logvar_linear(hidden_x)
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
        entropy = torch.sum(torch.distributions.normal.Normal(z_mu, torch.exp(0.5 * z_logvar)).log_prob(z),
                            dim=1)
        for transition in self._transitions:
            z, log_probability = transition(z, hidden_x)
            entropy += log_probability
        return z, entropy

    def forward(self, x):
        z, entropy = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        scale = x.size()[0]
        kl_part = entropy - torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(z), dim=1)
        kl_part = torch.sum(kl_part) / scale
        x: torch.Tensor
        nll_part = self.nll_part_loss(reconstructed_x, x) / scale
        beta = self.calculate_beta()
        loss = kl_part * beta + nll_part
        return reconstructed_x, loss, nll_part, kl_part

    def log_importance_weight(self, x):
        z, entropy = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        log_p_x = -torch.sum(self._sample_loss(reconstructed_x, x), dim=[1, 2, 3])
        log_p_z = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(z), dim=1)
        log_q_z = entropy
        return log_p_x + log_p_z - log_q_z
