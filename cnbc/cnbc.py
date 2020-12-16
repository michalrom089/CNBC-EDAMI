import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def make_distance_df(data):
    dist = cdist(data, data, "euclidean")
    return pd.DataFrame(data=dist)


def eps_neighborhood(data, p_idx, eps=0.3, dist_method="euclidean"):
    """ Epsilon neighborhood

    Args:
        data : pd.DataFrame
        p_idx : int
            index of the considered point
        eps : float
            minimum distance

    Return:
        np.array :
            points indx that meet the condition of min eps
    """

    dist = cdist(data, [data.iloc[p_idx]], metric=dist_method)
    # reduce dimensions (cdist produces ''matrix'')
    dist = dist[:, 0]

    q = np.argwhere(dist < eps)
    q_without_p = np.delete(q, p_idx)

    return q_without_p


def k_neighborhood(data, p_idx, k, dist_method="euclidean"):
    """ k-neighborhood or kNN(p)

    Args:
        data : pd.DataFrame
        p_idx : int
            index of the considered point
        k : int
            number of points satisfying the kNN(p) condition

    Return:
        np.array :
            points indx that meet the condition kNN(p) condition

    @see: http://ceur-ws.org/Vol-1269/paper113.pdf Section 3, Definition 2 
    """

    assert k > 0

    dist = cdist(data, [data.iloc[p_idx]], metric=dist_method)
    # reduce dimensions (cdist produces ''matrix'')
    dist = dist[:, 0]
    # sort indexes of dist in a ascending order
    dist_idx_asc = np.argsort(dist)
    # remove p_idx from sorted indexes
    dist_idx_asc = np.delete(dist_idx_asc, p_idx)

    # return first k indexes
    return dist_idx_asc[:k]


def kknn(data, p):
    """ punctured k+-neighborhood
    """
    pass


def rknn(data, p):
    """ reversed  punctured k+-neighborhood  of  a  point p
    """
    pass


def ndf(data, p):
    pass


if __name__ == "__main__":
    iris = pd.read_csv("./iris_csv.csv")
    cl = iris["class"]

    data = iris[["sepallength", "sepalwidth", "petallength", "petalwidth"]]
    eps_n = k_neighborhood(data.head(), 0, 2)

    print(eps_n)
