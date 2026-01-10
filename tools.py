import torch
import os
import matplotlib.pyplot as plt


def is_spd(P):
    is_symmetric = torch.allclose(P, P.T, atol=1e-8)

    eigvals_P = torch.linalg.eigvals(P).real
    is_positive_definite = torch.all(eigvals_P > 0)

    return is_symmetric and is_positive_definite


def mapto_SPD_cone(cov, mu, beta, jitter=1e-7):

    beta = torch.tensor(beta).view(1,1).to(mu.device)

    P_11 = cov + (beta * (torch.outer(mu, mu)))

    if mu.dim() == 1:
        mu = mu.unsqueeze(-1)

    top = torch.cat((P_11, mu), dim=1)

    bottom = torch.cat((mu.T, beta), dim=1)
    P = torch.cat((top, bottom), dim=0)

    P = (P + P.T) / 2.0
    P = P + torch.eye(P.size(0)).to(mu.device) * jitter

    assert is_spd(P.clone().detach()), "The P matrix is not SPD!"

    return P


def plot_stats(res_dict, ylabel, dir, filename):
    for k in res_dict.keys():
        v = res_dict[k]
        if None in v:
            continue
        plt.plot(range(1, len(v)+1), v, label=k)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(dir, filename), bbox_inches='tight', dpi=300)
    plt.close()