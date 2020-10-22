import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .vae_decoder import VaeDecoder
from .vae_encoder import VaeEncoder


# noinspection PyArgumentList
class VAE(pl.LightningModule):
    def __init__(self, input_dimension=28, latent_dimension=40, hidden_dimensions=(300, 300), lr=1e-3):
        super(VAE, self).__init__()
        self.encoder = VaeEncoder(input_dimension=input_dimension, hidden_dimensions=hidden_dimensions)
        self.decoder = VaeDecoder(hidden_dimensions=hidden_dimensions, latent_dimension=latent_dimension,
                                  output_dimension=input_dimension)
        self._latent_dimension = latent_dimension
        self._lr = lr
        self._loss = nn.BCEWithLogitsLoss(reduction="sum")
        self._sample_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def training_step(self, batch, batch_index):
        x, loss = self.forward(batch[0])
        self.log("loss", loss, on_step=True)
        return loss

    def test_step(self, batch, batch_index):
        loss = self.calculate_nll(batch[0])
        self.log("loss", loss, on_step=True)
        return loss

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def log_p_z(z):
        return torch.sum(torch.distributions.normal.Normal(0, 1.).log_prob(z), dim=1)

    def reconstruct_x(self, x):
        x_mean, _ = self.forward(x)
        return x_mean

    def log_importance_weight(self, x):
        raise NotImplementedError

    def calculate_nll(self, x, sample_count=100):
        x = torch.repeat_interleave(x, sample_count, dim=0)
        log_prediction = self.log_importance_weight(x).reshape(-1, sample_count)
        return -torch.mean(torch.logsumexp(log_prediction, dim=1) - np.log(sample_count))

    def generate_x(self, n=25, device="cpu"):
        eps = torch.randn(size=(n, self._latent_dimension), device=device)
        x = self.decoder(eps)
        return x > 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer

    def nll_part_loss(self, x, target):
        return self._loss(x, target)
