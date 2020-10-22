import torch.nn as nn


class VaeEncoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimensions=(300, 300)):
        super().__init__()
        modules = []

        previous_dimension = input_dimension * input_dimension
        for hidden_dimension in hidden_dimensions:
            modules.append(nn.Linear(previous_dimension, hidden_dimension))
            modules.append(nn.BatchNorm1d(hidden_dimension))
            modules.append(nn.LeakyReLU())
            previous_dimension = hidden_dimension
        self._encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self._encoder(x)
