import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .vae_decoder import VaeDecoder
from .vae_encoder import VaeEncoder


# noinspection PyArgumentList
class VAE(pl.LightningModule):
    def __init__(self, input_dimension=28, latent_dimension=40, hidden_dimensions=(300, 300), lr=1e-3,
                 beta=1, gamma=0, optim_config=None, encoder_config=None):
        super(VAE, self).__init__()
        if encoder_config is None:
            encoder_config = {}
        self.encoder = VaeEncoder(input_dimension=input_dimension, hidden_dimensions=hidden_dimensions,
                                  **encoder_config)
        self.decoder = VaeDecoder(hidden_dimensions=hidden_dimensions, latent_dimension=latent_dimension,
                                  output_dimension=input_dimension, **encoder_config)
        self._latent_dimension = latent_dimension
        self._lr = lr
        self._loss = nn.BCEWithLogitsLoss(reduction="sum")
        self._sample_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.init_params()
        self._current_beta = beta
        self._gamma = gamma
        if optim_config is None:
            optim_config = {}
        self._optim_config = optim_config

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def calculate_beta(self):
        return self._current_beta

    def update_beta(self):
        self._current_beta += self._gamma
        self._current_beta = np.clip(self._current_beta, 0, 1)

    def on_train_epoch_end(self, outputs) -> None:
        super().on_train_epoch_end(outputs)
        self.update_beta()

    def training_step(self, batch, batch_index):
        x, loss, nll_part, kl_part = self.forward(batch[0])
        self.log("train_loss", loss)
        self.log("train_kl", kl_part)
        self.log("train_nll", nll_part)
        self.log("train_elbo", kl_part + nll_part)
        return loss

    def validation_step(self, batch, batch_index):
        x, loss, nll_part, kl_part = self.forward(batch[0])
        self.log("val_loss", loss)
        self.log("val_kl", kl_part)
        self.log("val_nll", nll_part)
        self.log("val_elbo", kl_part + nll_part)
        return kl_part + nll_part

    def test_step(self, batch, batch_index):
        loss = self.calculate_nll(batch[0])
        self.log("loss", loss)
        return loss

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def log_p_z(z):
        return torch.sum(torch.distributions.normal.Normal(0, 1.).log_prob(z), dim=1)

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
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
        return torch.sigmoid(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, **self._optim_config)
        return optimizer

    def nll_part_loss(self, x, target):
        return self._loss(x, target)
