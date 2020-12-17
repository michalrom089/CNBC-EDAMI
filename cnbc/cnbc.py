from .neighborhood import calc_distance_ndarray, neighborhood_dense_factor

import numpy as np
from scipy.spatial.distance import cdist


class CNBC():

    def __init__(self, data):
        self.data = data

    def calc_distance_ndarray(self, dist_method="euclidean"):
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


def cnbc(data, k):
    """ Neighborhood-Based Clustering with Constraints 

        data : pd.DataFrame
            data to be clustered
        k : int
            density of a point
    """
    dist_method = "euclidean"
    dist_ndarr = calc_distance_ndarray(data, dist_method=dist_method)
    cluster_id = 0

    data["Cluster"] = "unclassified"

    # calculate NDF for each point
    data["NDF"] = list(map(lambda idx: neighborhood_dense_factor(
        dist_ndarr, idx, k), data.index.values))

    print(data.head())
