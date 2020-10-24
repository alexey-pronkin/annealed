import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pyro.nn import AutoRegressiveNN

from ..toy_dist import SimpleNormal
from .hvi_avo import HVIAVO
from .hvi_elbo import HVIELBO
from .utils import get_activation


class IATransform(nn.Module):
    def __init__(self, in_dim, h_dim, rand_perm=True, activation="ReLU", num_stable=False):
        super(IATransform, self).__init__()
        if rand_perm:
            self.permutation = torch.randperm(in_dim, dtype=torch.int64)
            nonlinearity = get_activation(activation)
        else:
            self.permutation = torch.arange(in_dim, dtype=torch.int64)
        # if len(h_dim)==1:
        #     self.AR = nn.ModuleList([
        #         nn.Linear(in_features=in_dim, out_features=h_dim),
        #         nonlinearity(),
        #         nn.Linear(in_features=h_dim, out_features=in_dim)]
        #         )
        # else:
        #     self.AR = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Linear(in_features=in_dim, out_features=out_dim),
        #             nonlinearity()) for in_dim, out_dim in zip([in_dim, h_dim[:-1]], [h_dim, in_dim])])# AutoRegressiveNN(input_dim=in_dim, hidden_dims=h_dim, nonlinearity=nonlinearity)  # nn.ELU(0.6))
        # TODO: make forward
        self.AR = AutoRegressiveNN(input_dim=in_dim, hidden_dims=h_dim, nonlinearity=nonlinearity, permutation=self.permutation)  # nn.ELU(0.6))
        self.num_stable = num_stable

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """
        m, log_s = self.AR(x)
        if self.num_stable:
            s = torch.sigmoid(torch.exp(log_s))
            x = (1 - s) * m + s * x  # [:, self.permutation]
            log_det = torch.log(s).sum(axis=1)  # - 0.000001
        else:
            x = torch.exp(log_s) * x + m
            log_det = log_s.sum(axis=1)
        return x, -log_det


class IAF(nn.Module):
    def __init__(self, in_dim, h_dim, depth, activation="ReLU", num_stable=False):
        super(IAF, self).__init__()
        """
        in_dim - inout dimention
        h_dim - hidden dimention of the autoregressive NN
        depth - number of IA transformations
        """

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.depth = depth
        self.IAFs = nn.ModuleList(
            [
                IATransform(in_dim=in_dim, h_dim=h_dim, rand_perm=True, activation=activation, num_stable=num_stable)
                for i in range(depth)
            ]
        )

    def forward(self, x):
        """
        Return transformed sample and log determinant
        """

        log_det = 0
        for i, AF in enumerate(self.IAFs):
            x, d = AF(x)
            log_det += d
        return x, log_det


class Householder(nn.Module):
    def __init__(self, input_dimension, activation="ReLU", num_stable=True):
        super(Householder, self).__init__()
        """
        input_dimension - inout dimention = v_dim
        h_dim - hidden dimention of the autoregressive NN
        depth - number of IA transformations
        """

        self.v_dim = input_dimension
        self.v = nn.Parameter(torch.Tensor(self.v_dim))
        self.reset_parameters()
        self.num_stable = num_stable

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.v, a=-1, b=1)

    def forward(self, z, v=None):
        """
        Return transformed sample and log determinant
        Note v could be just learning parameter or NN, that uses hidden information, so if v is not None we use it.
        """
        batch, _ = z.shape
        if v is None:
            v = self.v.repeat(batch, 1)
        else:
            v = v.repeat(batch, 1)
        if self.num_stable:
            vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
            vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(2)
        else:
            # v * v_T
            vvT = (v * v).sum(
                1
            )  # memory inefficient, however fast, could use author instead  
            # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

            # v * v_T * z
            vvTz = (vvT * z).sum(
                1
            )  # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum(v * v, 1)  # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1))  # expand sizes : B x L
        # calculate new z
        z = z - 2 * vvTz / norm_sq  # z - 2 * v * v_T  * z / norm2(v)
        log_det = 0
        return z, log_det


class HouseholderELBO(HVIELBO):
    def __init__(
        self,
        input_dimension,
        depth,
        target,
        lr=1e-3,
        batch_size=64,
        hidden_dimension=2,
        beta1=0.999,
        beta2=0.999,
        optimizer="adam",
        k=100,
        transitions=None,
        is_flow=True,
    ):
        transitions = [Householder(input_dimension) for _ in range(depth)]
        super().__init__(
            input_dimension=input_dimension,
            depth=depth,
            target=target,
            lr=lr,
            batch_size=batch_size,
            hidden_dimension=2,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            k=k,
            transitions=transitions,
            is_flow=is_flow,
        )

    def get_losses():
        pass


class IAFELBO(HVIELBO):
    def __init__(
        self,
        input_dimension,
        depth,
        target,
        lr=1e-3,
        batch_size=64,
        hidden_dimension=2,
        beta1=0.999,
        beta2=0.999,
        optimizer="adam",
        k=100,
        transitions=None,
        is_flow=True,
        AR_dimensions=[128],
        activation="ReLU",
        num_stable=False,
    ):
        transitions = [
            IATransform(
                in_dim=hidden_dimension,
                h_dim=AR_dimensions,
                rand_perm=True,
                activation=activation,
                num_stable=num_stable,
            )
            for _ in range(depth)
        ]
        super().__init__(
            input_dimension=input_dimension,
            depth=depth,
            target=target,
            lr=lr,
            batch_size=batch_size,
            hidden_dimension=2,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            k=k,
            transitions=transitions,
            is_flow=is_flow,
        )

    def get_losses():
        pass


class HouseholderAVO(HVIAVO):
    def __init__(
        self,
        input_dimension,
        depth,
        target,
        lr=1e-3,
        batch_size=64,
        hidden_dimension=2,
        beta1=0.999,
        beta2=0.999,
        optimizer="adam",
        k=100,
        transitions=None,
        is_flow=True,
    ):
        transitions = [Householder(input_dimension) for _ in range(depth)]
        super().__init__(
            input_dimension=input_dimension,
            depth=depth,
            target=target,
            lr=lr,
            batch_size=batch_size,
            hidden_dimension=hidden_dimension,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            k=k,
            transitions=transitions,
            is_flow=is_flow,
        )

    def get_losses():
        pass


class IAFAVO(HVIAVO):
    def __init__(
        self,
        input_dimension,
        depth,
        target,
        lr=1e-3,
        batch_size=64,
        hidden_dimension=2,
        beta1=0.999,
        beta2=0.999,
        optimizer="adam",
        k=100,
        transitions=None,
        is_flow=True,
        AR_dimensions=[128],
        activation="ReLU",
        num_stable=False,
    ):
        transitions = [
            IATransform(
                in_dim=input_dimension,
                h_dim=AR_dimensions,
                rand_perm=True,
                activation=activation,
                num_stable=num_stable,
            )
            for _ in range(depth)
        ]
        super().__init__(
            input_dimension=input_dimension,
            depth=depth,
            target=target,
            lr=lr,
            batch_size=batch_size,
            hidden_dimension=hidden_dimension,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            k=k,
            transitions=transitions,
            is_flow=is_flow,
        )

    def get_losses():
        pass

