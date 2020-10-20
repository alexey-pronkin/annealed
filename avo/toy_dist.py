import numpy as np
from avo.normal_dist import batch_mvn
import torch


class Target:
    '''
     x (torch tensor (B,D)) batch of points
    '''

    def __init__(self, device):
        self.cached_grid = None  # plot trivial round
        self.device = device

    def E(self, x):
        '''
         Out:
             energy E from p = exp(-E(x)) (torch tensor (B,))
        '''
        raise NotImplementedError

    def E_ratio(self, x_s, x_p):
        return -(self.E(x_p) - self.E(x_s))

    def get_grad_E(self, x):
        '''
         Out:
             grad of E from p = exp(-E(x)) (torch tensor (B,D))
        '''
        x = x.clone().requires_grad_()
        E = self.E(x)
        E.sum().backward()
        return x.grad

    def plot2d_pdf(self, ax, bounds=((-6, 6), (-6, 6)),  n_points=150):
        bounds_x = bounds[0]
        bounds_y = bounds[1]

        x = np.linspace(*bounds_x, n_points)
        y = np.linspace(*bounds_y, n_points)
        levels = np.zeros((len(x), len(y)))
        if self.cached_grid is not None:
            levels = self.cached_grid
        else:
            for i in range(len(x)):
                for j in range(len(y)):
                    levels[i, j] = np.exp(
                        -self.E(torch.tensor([x[i], y[j]], device=self.device, dtype=torch.float).unsqueeze(0)).data.cpu().numpy())
            self.cached_grid = levels
        levels /= np.sum(levels)
        ax.contour(x, y, levels.T)
        ax.title.set_text(self.name)


class SimpleNormal(Target):
    def __init__(self, device):
        super().__init__(device)

    def E(self, x):
        return batch_mvn.E(x, 0, 1, True)

class PickleRick(Target):
    def __init__(self, mu, L, diag=False):
        Target.__init__(self,  L.device)
        self.name = 'Pickle Rick'
        self.diag = diag
        self.L = L
        self.inv_L = 1. / L if diag else torch.inverse(L)
        self.mu = mu

    def E(self, x):
        return batch_mvn.E(x, self.mu, self.inv_L, self.diag)


class ThreeMixture(Target):
    def __init__(self, mu, L, diag=False):
        Target.__init__(self, L.device)
        self.name = 'Three Mixture'
        self.K = mu.size(0)
        self.diag = diag
        self.L = L
        self.inv_L = 1. / L if diag else torch.inverse(L)
        self.mu = mu

    def E(self, x):
        vec = torch.zeros((x.size(0), self.K), device=self.L.device)
        for k in range(self.K):
            vec[:, k] = -batch_mvn.E(x, self.mu[k].unsqueeze(0), self.inv_L[k])
        return -torch.logsumexp(vec, dim=1)


class Banana(Target):
    def __init__(self, mu, L, diag=False):
        Target.__init__(self,  L.device)
        self.name = 'Banana'
        self.diag = diag
        self.L = L
        self.inv_L = 1. / L if diag else torch.inverse(L)
        self.mu = mu

    def E(self, x):
        y = -x.clone()
        y[:, 1] = x[:, 1] + 0.5 * x[:, 0] ** 2 + 1
        return batch_mvn.E(y, self.mu, self.inv_L, self.diag)
