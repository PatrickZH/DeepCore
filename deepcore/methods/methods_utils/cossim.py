import numpy as np
import torch


def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def cossim_pair_np(v1):
    num = np.dot(v1, v1.T)
    norm = np.linalg.norm(v1, axis=1)
    denom = norm.reshape(-1, 1) * norm
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def cossim(v1, v2):
    num = torch.matmul(v1, v2.T)
    denom = torch.norm(v1, dim=1).view(-1, 1) * torch.norm(v2, dim=1)
    res = num / denom
    res[torch.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def cossim_pair(v1):
    num = torch.matmul(v1, v1.T)
    norm = torch.norm(v1, dim=1)
    denom = norm.view(-1, 1) * norm
    res = num / denom
    res[torch.isneginf(res)] = 0.
    return 0.5 + 0.5 * res