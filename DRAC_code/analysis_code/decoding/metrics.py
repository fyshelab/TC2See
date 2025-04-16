from typing import Sequence, Union
import random
import numpy as np
import torch
import torch.nn.functional as F

from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from pylablib.core.utils.funcargparse import is_sequence


def pearsonr(Y, Y_pred, dim=0):
    Y = Y.to(torch.float64)
    Y_pred = Y_pred.to(torch.float64)

    Y = Y - Y.mean(dim=dim, keepdim=True)
    Y_pred = Y_pred - Y_pred.mean(dim=dim, keepdim=True)

    Y = Y / torch.norm(Y, dim=dim, keepdim=True)
    Y_pred = Y_pred / torch.norm(Y_pred, dim=dim, keepdim=True)

    return (Y * Y_pred).sum(dim=dim).mean().item()


def r2_score(Y, Y_pred, dim=0, cast_dtype=torch.float64, reduction: str = 'mean'):
    in_dtype = Y.dtype
    if cast_dtype:
        Y = Y.to(cast_dtype)
        Y_pred = Y_pred.to(cast_dtype)

    ss_res = ((Y - Y_pred) ** 2).sum(dim=dim)
    ss_tot = ((Y - Y.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)

    r2 = 1 - ss_res / ss_tot
    if reduction == 'mean':
        r2 = r2.mean()
    if cast_dtype:
        r2 = r2.to(in_dtype)
    return r2


def squared_euclidean_distance(Y1, Y2):
    Y1_squared = (Y1 ** 2).sum(dim=-1)
    Y2_squared = (Y2 ** 2).sum(dim=-1)
    Y1_dot_Y2 = torch.einsum('... i, ... i -> ...', Y1, Y2)

    # recall (y1 - y2)^2 = y1^2 + y2^2 - 2y1*y2
    squared_distance = Y1_squared + Y2_squared - 2 * Y1_dot_Y2
    return squared_distance


def mean_squared_distance(Y1, Y2):
    return squared_euclidean_distance(Y1, Y2) / Y1.shape[1]


def cosine_distance(Y1, Y2):
    Y1 = Y1 / Y1.norm(dim=-1, keepdim=True)
    Y2 = Y2 / Y2.norm(dim=-1, keepdim=True)

    cos_sim = F.cosine_similarity(Y1, Y2, dim=-1)

    return 1 - cos_sim
    # return 1. - torch.einsum('... i, ... i -> ...', Y1, Y2)


def two_versus_two(distances, stimulus_ids=None):
    different = distances + distances.T

    distances_diag = torch.diag(distances)
    same = distances_diag[None, :] + distances_diag[:, None]

    comparison = same < different
    upper_triangle_ids = np.triu_indices(distances.shape[0], k=1)
    comparison = comparison[upper_triangle_ids]
    
    if stimulus_ids is not None:
        same_stimulus = stimulus_ids[None, :] == stimulus_ids[:, None]
        same_stimulus = same_stimulus[upper_triangle_ids]
        comparison = comparison[~same_stimulus]

    return comparison.float().mean()


def two_versus_two_slow(distances, stimulus_ids=None):
    N = distances.shape[0]
    results = []
    for i in range(N):
        for j in range(i + 1, N):
            if stimulus_ids is not None and stimulus_ids[i] == stimulus_ids[j]:
                continue
            s1 = distances[i, i]
            s2 = distances[j, j]
            d1 = distances[i, j]
            d2 = distances[j, i]
            results.append((s1 + s2) < (d1 + d2))
    return np.array(results).mean()


def top_knn_test(
        x: ArrayLike,
        y: ArrayLike,
        k: Union[int, Sequence[int]],
        metric: str = 'euclidean'
):
    neighbors = NearestNeighbors(metric=metric)

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    if not is_sequence(k):
        k = [k]

    neighbors.fit(x)

    nearest_ids = neighbors.kneighbors(y, n_neighbors=np.max(k), return_distance=False)
    N = x.shape[0]
    target_ids = np.arange(N)[:, None]
    accuracy = [
        np.any(nearest_ids[:, :int(some_k)] == target_ids, axis=1).mean()
        for some_k in k
    ]
    return accuracy
