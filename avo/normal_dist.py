import torch
import math


class batch_mvn:
    '''
    mu (torch tensor (B,D)): batch of mean
    L (torch tensor (D,D)): batch corrsepondend Choltetsky factor, low triangular
    L (torch tensor (D,)): batch corrsepondend Choltetsky factor, diagonal
    x (torch tensor (B,D)) batch of points
    '''

    @staticmethod
    def rvs(mu, L, diag=False):
        '''
        Out:
            (torch tensor (B,D)) batch from N(x|mu, LL^T)
        '''
        if diag:
            eps = torch.randn_like(mu)
            sample = mu + eps * L
        else:
            eps = torch.randn_like(mu).unsqueeze(-1)
            sample = mu.unsqueeze(-1) + torch.einsum('ijk,ikl->ijl', [L.expand(mu.size(0), -1, -1), eps])
            sample = sample.squeeze(-1)
        return sample

    @staticmethod
    def E(x, mu, L_inv, diag=False):
        '''
        Out:
            (torch tensor (B,)) batch of energy N(x|mu, LL^T)
        '''
        if diag:
            vec = (x - mu) / L_inv
        else:
            # vec, _ = torch.triangular_solve((x-mu).unsqueeze(2), L, upper=False) # for RHMC
            vec = torch.einsum('ijk,ikl->ijl', [L_inv.expand(x.size(0), -1, -1), (x - mu).unsqueeze(2)])
            vec = vec.squeeze(2)
        return 0.5 * torch.sum(vec ** 2, dim=1)

    @staticmethod
    def logpdf(x, mu, L_inv, diag=False):
        '''
        Out:
            (torch tensor (B,)) batch of log  N(x|mu, LL^T)
        '''
        e = batch_mvn.E(x, mu, L_inv, diag)
        if diag:
            log_det = -torch.sum(torch.log(L_inv))
        else:
            log_det = -torch.einsum('jj', [torch.log(L_inv)])
        return - e - log_det - mu.size(1) * 0.5 * torch.log(torch.Tensor(2. * math.pi))
