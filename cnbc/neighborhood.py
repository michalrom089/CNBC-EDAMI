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
        np.array :
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
        np.array :
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
        np.array :
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
        np.array :
            points indx that meet the condition reversed_k_neighborhood(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 4
    """

    s = p_k_n.shape
    r_k_n = np.zeros(shape=(s[0]*s[1], 2), dtype=int)
    print(s)
    # brute force
    for p_idx, pp_k_n in enumerate(p_k_n):
        # reversed_k_neighborhood for p_idx
        pr_k_n = np.array(list(zip(pp_k_n, [p_idx]*len(pp_k_n))))

        start_idx = p_idx * s[1]
        end_idx = (p_idx+1) * s[1]

        r_k_n[start_idx:end_idx, :] = pr_k_n

    # remove p_idx from sorted indexes
    return r_k_n


def neighborhood_dense_factor(dist_ndarr, p_idx, k):
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
