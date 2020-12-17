from .neighborhood import calc_distance_ndarray, k_neighborhood, punctured_k_neighborhood, reversed_k_neighborhood, neighborhood_dense_factor

import numpy as np
import pandas as pd

UNCLASSIFIED = "unclassified"


class CNBC():
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
        """

        # calculate all neighborhood metrics
        dist_ndarr = calc_distance_ndarray(data, self.dist_method)
        k_n = k_neighborhood(dist_ndarr, k=self.k)
        p_k_n = punctured_k_neighborhood(k_n)
        r_k_n = reversed_k_neighborhood(p_k_n)
        ndf = neighborhood_dense_factor(p_k_n, r_k_n)

        cluster = pd.Series([UNCLASSIFIED]*len(data))
        deffered_p = set()
        dp_set = set()

        cluster_id = 0

        # TODO include must_link and cannot_link

        # for each unclassified point that is dense
        for p in np.argwhere(self.is_dense(ndf))[:, 0]:

            if cluster[p] != UNCLASSIFIED:
                continue

            cluster[p] = cluster_id

            dp_set.clear()

            # for each point from (punctured_k_neighborhood(p) \ deffered_points)
            for q in [el for el in p_k_n[p, :] if el not in deffered_p]:
                cluster[q] = cluster_id
                if ndf[q] >= 1:
                    dp_set.add(q)
                # TODO add all point r from must_link(q) such r.ndf >=1 to dp_set

            while(len(dp_set) == 0):
                s = dp_set.pop()

                # for each unclassified point from (punctured_k_neighborhood(s) \ deffered_points)
                for t in [el for el in p_k_n[s, :] if el not in deffered_p]:
                    if cluster[p] != UNCLASSIFIED:
                        continue

                    cluster[t] = cluster_id
                    if ndf[t] >= 1:
                        dp_set.add(t)
                    # TODO add all point u from must_link(t) such u.ndf >=1 to dp_set

            cluster_id = cluster_id + 1

        return cluster

    def is_dense(self, ndf):
        return ndf >= 1
