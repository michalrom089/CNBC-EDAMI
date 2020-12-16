import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def calc_distance_ndarray(data, dist_method="euclidean"):
    """  Calculates distance   

    Args:
        data : pd.DataFrame
        dist_method : str or callable

    Returns:
        np.ndarray :
            shape(nd.array) = [len(data), len(data)]
    """
    dist = cdist(data, data, "euclidean")
    return dist


def eps_neighborhood(dist_ndarr, p_idx, eps=0.3):
    """ Epsilon neighborhood

    Args:
        dist_ndarr : np.ndarray
        p_idx : int
            index of the considered point
        eps : float
            minimum distance

    Return:
        np.array :
            points indx that meet the condition of min eps
    """

    # get distance array for considered point
    dist = dist_ndarr[p_idx, :]
    q = np.argwhere(dist <= eps)

    return q


def k_neighborhood(dist_ndarr, p_idx, k):
    """ k-neighborhood or kNN(p)

    Args:
        dist_ndarr : np.ndarray
        p_idx : int
            index of the considered point
        k : int
            density of a point

    Return:
        np.array :
            points indx that meet the condition kNN(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 2 
    """

    assert k > 0

    # get distance array for considered point
    dist = dist_ndarr[p_idx, :]
    # sort indexes of dist in a ascending order
    dist_idx_asc = np.argsort(dist)

    # return first k indexes
    return dist_idx_asc[:k]


def punctured_k_neighborhood(dist_ndarr, p_idx, k):
    """ punctured_k_neighborhood(p)

    Args:
        dist_ndarr : np.ndarray
        p_idx : int
            index of the considered point
        k : int
            density of a point

    Return:
        np.array :
            points indx that meet the condition punctured_k_neighborhood(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 3 
    """

    assert k > 0

    k_n = k_neighborhood(dist_ndarr, p_idx, k=k)

    # remove p_idx from sorted indexes
    return k_n[k_n != p_idx]


def reversed_k_neighborhood(dist_ndarr, p_idx, k):
    """ reversed_k_neighborhood(p)

    Args:
        dist_ndarr : np.ndarray
        p_idx : int
            index of the considered point
        k : int
            density of a point

    Return:
        np.array :
            points indx that meet the condition reversed_k_neighborhood(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 4
    """

    assert k > 0
    r_k_n = list()

    # brute force
    for q_idx in range(len(dist_ndarr)):
        # skip p_idx
        if q_idx == p_idx:
            continue

        p_k_n = punctured_k_neighborhood(dist_ndarr, q_idx, k=k)
        if p_idx in p_k_n:
            r_k_n.append(q_idx)

    # remove p_idx from sorted indexes
    return np.array(r_k_n)


def neighborhood_df(dist_ndarr, p_idx, k):
    """ neighborhood-based  density  factor  of  a  point

    Args:
        dist_ndarr : np.ndarray
        p_idx : int
            index of the considered point
        k : int
            density of a point

    Return:
        float:
            ndf value

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 5 
    """
    r_k_n = reversed_k_neighborhood(dist_ndarr, p_idx, k)
    p_k_n = punctured_k_neighborhood(dist_ndarr, p_idx, k)

    return len(r_k_n)/len(p_k_n)
