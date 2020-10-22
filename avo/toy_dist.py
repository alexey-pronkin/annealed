import numpy as np
import torch
import math

from avo.normal_dist import batch_mvn


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

    def plot2d_pdf(self, ax, bounds=((-6, 6), (-6, 6)), n_points=150):
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
                        -self.E(torch.tensor([x[i], y[j]], device=self.device, dtype=torch.float).unsqueeze(
                            0)).data.cpu().numpy())
            self.cached_grid = levels
        levels /= np.sum(levels)
        ax.contour(x, y, levels.T)
        ax.title.set_text(self.name)


class MixtureTarget(Target):
    def __init__(self, target1, target2, alpha):
        super().__init__(target1.device)
        self._target1 = target1
        self._target2 = target2
        self._alpha = alpha
        self.name = "mixture"

    def E(self, x):
        return self._target1.E(x) * self._alpha + self._target2.E(x) * (1 - self._alpha)


class SimpleNormal(Target):
    def __init__(self, device):
        super().__init__(device)

    def E(self, x):
        return batch_mvn.E(x, 0, 1, True)


class PickleRick(Target):
    def __init__(self, mu, L, diag=False):
        Target.__init__(self, L.device)
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
        Target.__init__(self, L.device)
        self.name = 'Banana'
        self.diag = diag
        self.L = L
        self.inv_L = 1. / L if diag else torch.inverse(L)
        self.mu = mu

    def E(self, x):
        y = -x.clone()
        y[:, 1] = x[:, 1] + 0.5 * x[:, 0] ** 2 + 1
        return batch_mvn.E(y, self.mu, self.inv_L, self.diag)



class ToyA(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'toy dist a'

    def E(self, z):
        a = -0.5 * ((torch.norm(z, dim=1) - 2) / 0.4)**2  # check dim
        vec = torch.stack(
            (-0.5 * ((z[:, 0] - 2) / 0.6)**2,
             -0.5 * ((z[:, 0] + 2) / 0.6)**2)
        )
        return -torch.logsumexp(vec, dim=0) - a


class ToyB(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'toy dist b'

    def E(self, z):
        a = -0.5 * (2 * (torch.norm(0.5 * z, dim=1) - 2))**2  # check dim
        vec = torch.stack(
            (-0.5 * ((z[:, 0] - 2) / 0.6)**2,
             -0.5 * (2 * torch.sin(z[:, 0]))**2,
             -0.5 * ((z[:, 0] + z[:, 1] + 2.5) / 0.6)**2)
        )
        return -torch.logsumexp(vec, dim=0) - a


class ToyC(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'toy dist c'

    def E(self, z):
        return (2. - (z[:, 0]**2 + z[:, 1]**2 * 0.5)**0.5)**2


class DFunction(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'D'

    def E(self, x):
        part1 = torch.sum(torch.distributions.normal.Normal(
            torch.tensor([-2., 0], device=self.device), 0.2).log_prob(x), dim=1) + np.log(0.1)
        part2 = torch.sum(torch.distributions.normal.Normal(
            torch.tensor([2., 0], device=self.device), 0.2).log_prob(x), dim=1) + np.log(0.3)
        part3 = torch.sum(torch.distributions.normal.Normal(
            torch.tensor([0, 2.], device=self.device), 0.2).log_prob(x), dim=1) + np.log(0.4)
        part4 = torch.sum(torch.distributions.normal.Normal(
            torch.tensor([0, -2.], device=self.device), 0.2).log_prob(x), dim=1) + np.log(0.2)
        return -torch.logsumexp(torch.stack([part1, part2, part3, part4], dim=0), dim=0)


class ToyD(DFunction):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'four-mode mixture of Gaussian as a true target energy'


class ToyE(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'toy dist e'

    def E(self, z):
        w1 = torch.sin(math.pi * 0.5 * z[:, 0])
        w2 = 3 * torch.exp(-0.5 * (z[:, 0] - 2)**2)
        # return - 0.5 * ((z[:, 1] - 2) / 0.4)**2 - 0.1 * z[:, 0]**2
        a = -0.5 * ((z[:, 1] - w1) / 0.4)**2
        return - a + 0.1 * z[:, 0]**2


class ToyF(Target):
    def __init__(self, device='cpu'):
        Target.__init__(self, device)
        self.name = 'toy dist f'

    def E(self, z):
        w1 = torch.sin(math.pi * 0.5 * z[:, 0])
        w2 = 3 * torch.exp(-0.5 * (z[:, 0] - 2)**2)
        vec = torch.stack((
            -0.5 * ((z[:, 1] - w1) / 0.35)**2,
            -0.5 * ((z[:, 1] - w1 + w2) / 0.35)**2,
        ))
        return -torch.logsumexp(vec, dim=0) + 0.05 * z[:, 0] ** 2




class AnnealedA(ToyA):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = super().E(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class AnnealedB(ToyB):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = super().E(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class AnnealedC(ToyC):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = super().E(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class AnnealedD(ToyD):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E(self, x):
        ret = super().E(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class AnnealedE(ToyE):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E0(self, z):
        w1 = torch.sin(math.pi * 0.5 * z[:, 0])
        w2 = 3 * torch.exp(-0.5 * (z[:, 0] - 2)**2)
        # return - 0.5 * ((z[:, 1] - 2) / 0.4)**2 - 0.1 * z[:, 0]**2
        a = -0.5 * ((z[:, 1] - w1) / 0.4)**2
        return - a + 0.1 * z[:, 0]**2

    def E(self, x):
        # ret = super().E(x)
        ret = self.E0(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


class AnnealedF(ToyF):
    def __init__(self, alpha=0, device='cpu'):
        super().__init__()
        self._initial_target = SimpleNormal(device)
        self.alpha = alpha

    def E0(self, z):
        w1 = torch.sin(math.pi * 0.5 * z[:, 0])
        w2 = 3 * torch.exp(-0.5 * (z[:, 0] - 2)**2)
        vec = torch.stack((
            -0.5 * ((z[:, 1] - w1) / 0.35)**2,
            -0.5 * ((z[:, 1] - w1 + w2) / 0.35)**2,
        ))
        return -torch.logsumexp(vec, dim=0) + 0.05 * z[:, 0] ** 2

    def E(self, x):
        # ret = super().E(x)
        ret = self.E0(x)
        ann = (1. - self.alpha) * self._initial_target.E(x) + self.alpha * ret
        return ann


targets = [ToyA(), ToyB(), ToyC(), ToyD(), ToyD(), ToyE(), ToyF()]

anntargets = [
    AnnealedA,
    AnnealedB,
    AnnealedC, 
    AnnealedD, 
    AnnealedE, 
    AnnealedF,
]

def genTransitionalTargets(target, depth=4):
    alphas = np.linspace(0., 1., depth)
    return [target(i) for i in alphas]
