import torch.nn as nn


class VaeDecoder(nn.Module):
    def __init__(self, latent_dimension=40, hidden_dimensions=(300, 300), output_dimension=28):
        super().__init__()
        modules = []

        previous_dimension = latent_dimension
        for hidden_dimension in reversed(hidden_dimensions):
            modules.append(nn.Linear(previous_dimension, hidden_dimension))
            modules.append(nn.BatchNorm1d(hidden_dimension))
            modules.append(nn.LeakyReLU())
            previous_dimension = hidden_dimension
        modules.append(nn.Linear(previous_dimension, output_dimension * output_dimension))
        self._output_dimension = output_dimension
        self._decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self._decoder(x)
        return x.reshape(x.shape[0], 1, self._output_dimension, self._output_dimension)