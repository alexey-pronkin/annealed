import torch.nn as nn


class RefinementOperatorModule(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()

        self._linear_sigma = nn.Linear(input_dimension, input_dimension)
        self._linear_m = nn.Linear(input_dimension, input_dimension)
        self._linear_g = nn.Linear(input_dimension, input_dimension)
        self._linear_h = nn.Linear(input_dimension, input_dimension)

        self._sigma_activation = nn.Softplus()
        self._g_activation = nn.Sigmoid()
        self._h_activation = nn.ReLU()

    def init_weights(self):
        for parameter in self.parameters():
            nn.init.normal_(parameter)

    def forward(self, z):
        m = self._linear_m(z)
        g = self._g_activation(self._linear_g(z))
        h = self._h_activation(self._linear_h(z))
        mu = g * m + (1 - g) * z
        sigma = self._sigma_activation(self._linear_sigma(h))
        return mu, sigma
