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

    dist = cdist(data, [data.iloc[p_idx]])
    # reduce dimensions (cdist produces ''matrix'')
    dist = dist[:, 0]

    q = np.argwhere(dist < eps)
    q_without_p = np.delete(q, p_idx)

    return q_without_p


def rknn(data, p):
    """ reversed  punctured k+-neighborhood  of  a  point p
    """
    pass


def kknn(data, p):
    """ punctured k+-neighborhood
    """
    pass


def ndf(data, p):
    pass


if __name__ == "__main__":
    iris = pd.read_csv("./iris_csv.csv")
    cl = iris["class"]

    data = iris[["sepallength", "sepalwidth", "petallength", "petalwidth"]]
    eps_n = eps_neighborhood(data, 0)

    print(eps_n)
