from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class HKRPair(nn.Module):
    '''
    Hilbert Kernel Regularization
    REF: https://github.com/xpeng9719/defend_mi
    '''

    def __init__(self, weights, n_modality=3, sigma=5.):
        super(HKRPair, self).__init__()
        self.weights = weights
        self.sigma = sigma
        self.n_modality = n_modality
  
        self.all_components = list(
            itertools.combinations(range(n_modality), 2))
    def layer_regularization(self, x):
        """One layer regularization function"""
        batch_size = x.size(0)
        n_base = batch_size // self.n_modality
        x = x.view(self.n_modality, n_base, -1)
        loss = 0.0
        for (i, j) in self.all_components:
            loss += hsic_normalized_cca(x[i], x[j], self.sigma )
        return loss

    def forward(self, intermediate_outputs: list = []):
        """Forward a list of layer"""
        return -sum([w * self.layer_regularization(ft) for w, ft in zip(self.weights, intermediate_outputs) if w > 0])

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def coco_kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(H, torch.mm(Kx, H))

    return Kxc


def coco_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    K = coco_kernelmat(x, sigma=sigma)
    L = coco_kernelmat(y, sigma=sigma, ktype=ktype)

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res


def coco_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    coco_hx_val = coco_normalized_cca(hidden, h_data, sigma=sigma)
    coco_hy_val = coco_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return coco_hx_val, coco_hy_val


def kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))


    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)
    return Kxc


def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy


def hsic_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    hsic_hx_val = hsic_normalized_cca(hidden, h_data, sigma=sigma)
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)
    return hsic_hx_val, hsic_hy_val

if __name__ == '__main__':
    print("Test HKR")
    a = torch.rand(128, 65*64*32)*100
    b = torch.rand(128, 65*64*32)*100
    reg = hsic_normalized_cca(a, a, 5)
    print(f"Test on the same input pair (x, x): {reg:.4f}")
    reg = hsic_normalized_cca(a, b, 5)
    print(f"Test on the different input pair (x, y): {reg:.4f}")
    reg_loss = HKRPair(weights=[1], n_modality=3, sigma=5)
    ft_list = [torch.rand(3*128, 65*64*32)*100]
    out = reg_loss(ft_list)
    print(f"Test the module: {out:.4f}")