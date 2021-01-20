from copy import copy
from enum import Enum

import numpy as np
import pandas as pd

from nbc.neighborhood import calc_distance_ndarray, k_neighborhood
from nbc.neighborhood import punctured_k_neighborhood, reversed_k_neighborhood
from nbc.neighborhood import neighborhood_dense_factor


class PointClass(Enum):
    NOISE = 0
    UNCLASSIFIED = 1


class NBC():
    """ Neighborhood-Based Clustering with Constraints"""

    def __init__(self, k=4, dist_method="euclidean"):
        """
            Args:
                k : int
                    density of a point
                dist_method : str or callable
                    distance method
        """
        self.k = k
        self.dist_method = dist_method

    def fit(self, data):
        """
            Args:
                data : pd.DataFrame
                    data to be clustered
            Return:
                pd.Series: 
                    clusters

        """

        # calculate all neighborhood metrics
        dist_ndarr = calc_distance_ndarray(data, self.dist_method)
        k_n = k_neighborhood(dist_ndarr, k=self.k)
        p_k_n = punctured_k_neighborhood(k_n)
        r_k_n = reversed_k_neighborhood(p_k_n)
        ndf = neighborhood_dense_factor(p_k_n, r_k_n)

        # label all points as UNCLASSIFIED
        cluster = pd.Series([PointClass.UNCLASSIFIED]*len(data))
        dp_set = set()

        cluster_id = 0

        # for each unclassified point that is dense
        for p in np.argwhere(self.is_dense(ndf))[:, 0]:

            if not cluster[p] == PointClass.UNCLASSIFIED:
                continue

            cluster[p] = cluster_id
            
            dp_set.clear()

            # for each point from (punctured_k_neighborhood(p))
            for q in k_n[p, :]:
                if not cluster[q] == PointClass.UNCLASSIFIED:
                    continue

                cluster[q] = cluster_id
                if ndf[q] >= 1:
                    dp_set.add(q)

            while(len(dp_set) != 0):
                s = dp_set.pop()

                # for each unclassified point from (punctured_k_neighborhood(s) \ deffered_points)
                for t in k_n[s, :]:
                    if not cluster[t] == PointClass.UNCLASSIFIED:
                        continue

                    cluster[t] = cluster_id
                    if ndf[t] >= 1:
                        dp_set.add(t)

            cluster_id = cluster_id + 1

        # mark all unlcassified as noise
        cluster[cluster == PointClass.UNCLASSIFIED] = PointClass.NOISE

        return cluster

    def is_dense(self, ndf):
        return ndf >= 1
