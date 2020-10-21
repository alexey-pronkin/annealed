import torch.nn as nn


class RefinementOperatorModule(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=2):
        super().__init__()

        self._linear_h = nn.Linear(input_dimension, hidden_dimension)
        self._linear_sigma = nn.Linear(hidden_dimension, input_dimension)
        self._linear_m = nn.Linear(hidden_dimension, input_dimension)
        self._linear_g = nn.Linear(hidden_dimension, input_dimension)

        self._sigma_activation = nn.Softplus()
        self._g_activation = nn.Sigmoid()
        self._h_activation = nn.ReLU()

    def forward(self, z):
        h = self._h_activation(self._linear_h(z))
        m = self._linear_m(h)
        g = self._g_activation(self._linear_g(h))
        mu = g * m + (1 - g) * z
        sigma = self._sigma_activation(self._linear_sigma(h))
        return mu, sigma
