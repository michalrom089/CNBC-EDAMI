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


def eps_neighborhood(dist_ndarr, eps=0.3):
    """ Epsilon neighborhood

    Args:
        dist_ndarr : np.ndarray
        eps : float
            minimum distance

    Return:
        np.array[int] :
            points indx that meet the condition of min eps
    """

    # get distance array for considered point
    q = np.argwhere(dist_ndarr <= eps)

    return q


def k_neighborhood(dist_ndarr, k):
    """ k-neighborhood or kNN(p)

    Args:
        dist_ndarr : np.ndarray
        k : int
            density of a point

    Return:
        np.array[int] :
            points indx that meet the condition kNN(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 2 
    """

    assert k > 0

    # sort indexes of dist in a ascending order
    dist_idx_asc = np.argsort(dist_ndarr)
    # return first k indexes
    k_idx = dist_idx_asc[:, :k]

    return k_idx


def punctured_k_neighborhood(k_n):
    """ punctured_k_neighborhood(p)

    Args:
        k_n : np.ndarray
            k neighborhood 

    Return:
        np.array[int] :
            points indx that meet the condition punctured_k_neighborhood(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 3 
    """

    # remove point from their k_neighborhood from sorted indexes
    return k_n[:, 1:]


def reversed_k_neighborhood(p_k_n):
    """ reversed_k_neighborhood(p)

    Args:
        p_k_n : np.ndarray
            punctured k neighborhood

    Return:
        np.array[list] :
            points indx that meet the condition reversed_k_neighborhood(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 4
    """

    s = p_k_n.shape
    r_k_n = np.empty(s[0], dtype=object)

    for p_idx, pp_k_n in enumerate(p_k_n):
        # reversed_k_neighborhood for p_idx
        for el in pp_k_n:
            if not r_k_n[el]:
                r_k_n[el] = list()
            r_k_n[el].append(p_idx)

    # remove p_idx from sorted indexes
    return r_k_n


def neighborhood_dense_factor(p_k_n, r_k_n):
    """ neighborhood-based  density  factor  of  a  point

    Args:
        p_k_n : np.ndarray[int]
            punctured k neighborhood
        r_k_n : np.array[list]
            reversed k neighborhood

    Return:
        nd.array[float]:
            ndf value

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 5 
    """
    len_r = np.array(list(map(lambda x: len(x), r_k_n)), dtype=float)
    len_p = p_k_n.shape[1]

    return len_r / len_p
