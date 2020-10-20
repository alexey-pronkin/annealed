import pytorch_lightning as pl
import torch
import numpy as np
from .avo_transition import AVOTransition
from ..toy_dist import SimpleNormal
import torch.nn as nn


class HVIAVO(pl.LightningModule):
    def __init__(self, input_dimension, depth, target, lr=1e-3, batch_size=64, beta1=0.999, beta2=0.999,
                 optimizer="adam"):
        super().__init__()
        """
        in_dim - inout dimension
        h_dim - hidden dimension of the auto regressive NN
        depth - number of IA transformations
        """
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._optimizer = optimizer
        self._target = target
        self._batch_size = batch_size
        self._input_dimension = input_dimension
        alphas = np.linspace(1. / depth, 1., depth)
        initial_target = SimpleNormal(target.device)
        self._transitions = nn.ModuleList([AVOTransition(input_dimension, initial_target, target, alpha)
                                           for alpha in alphas])

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        full_loss = 0
        log_probability = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(x), dim=1)
        for transition in self._transitions:
            loss, x, log_probability = transition(x, log_probability)
            full_loss += loss
        return x, full_loss

    def sample(self, sample_count):
        z = torch.randn((sample_count, self._input_dimension), device=self.device)
        z, _ = self.forward(z)
        return z

    def data_loader(self, iteration_count):
        for i in range(iteration_count):
            z = torch.randn((self._batch_size, self._input_dimension))
            yield z.to(self.device)

    # noinspection PyTypeChecker
    def loss(self, batch):
        x, loss = self.forward(batch)
        return torch.mean(loss)

    def training_step(self, batch, batch_index):
        loss = self.loss(batch)
        self.log("loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        if self._optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, betas=(self._beta1, self._beta2))
        elif self._optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self._lr)
        else:
            optimizer = None
        return optimizer
