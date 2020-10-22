import numpy as np
import pytorch_lightning as pl
import torch


class HVI(pl.LightningModule):
    def __init__(self, input_dimension, target, lr=1e-3, batch_size=64, beta1=0.999, beta2=0.999,
                 optimizer="adam", k=100, is_flow=False):
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
        self._k = k
        self._is_flow = is_flow

    def forward(self, x):
        raise NotImplementedError

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

    # noinspection PyUnresolvedReferences
    def test_step(self, batch, batch_index):
        x, entropy = self.forward(batch)

        if not self._is_flow:
            expanded_x = torch.repeat_interleave(x, self._k, dim=0)

            full_log_probability = 0
            for transition in reversed(self._transitions):
                expanded_x, log_probability = transition.forward_reverse(expanded_x)
                full_log_probability += log_probability
            full_log_probability += torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(expanded_x), dim=1)
            full_log_probability = full_log_probability.reshape(-1, self._k)

            entropy = torch.logsumexp(full_log_probability, dim=1) - np.log(self._k)

        energy = self._target.E(x)
        loss = torch.mean(energy + entropy)
        self.log("energy", torch.mean(energy))
        self.log("entropy", torch.mean(entropy))
        self.log("loss", loss)
        return loss

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
