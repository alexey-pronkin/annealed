from torch import nn


def get_activation(activation):
    if activation == "ReLU":
        nonlinearity = nn.ReLU()
    elif activation == "ELU":
        nonlinearity = nn.ELU(0.4)
    elif activation == "Tanh":
        nonlinearity = nn.Tanh()
    elif activation == "Softplus":
        nonlinearity = nn.Softplus()
    else:
        nonlinearity = nn.ReLU()
    return nonlinearity
