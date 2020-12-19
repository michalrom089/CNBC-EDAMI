from copy import copy
from enum import Enum

import numpy as np
import pandas as pd

from cnbc.neighborhood import calc_distance_ndarray, k_neighborhood
from cnbc.neighborhood import punctured_k_neighborhood, reversed_k_neighborhood
from cnbc.neighborhood import neighborhood_dense_factor


class PointClass(Enum):
    NOISE = 0
    UNCLASSIFIED = 1
    DEFFERED = 2


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

    def fit(self, data, must_link, cannot_link):
        """
            Args:
                data : pd.DataFrame
                    data to be clustered
                must_link : np.array[tuple]
                    array of must link pairs
                cannot_link : np.array[tuple]
                    array of cannot link pairs
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
        deffered_p = set()
        dp_set = set()

        cluster_id = 0

        for q in self.unpack_link(cannot_link):
            # label q as deffered
            cluster[q] = PointClass.DEFFERED
            deffered_p.add(q)

            # label all points from the p_k_n(q) as deffered
            # for p in p_k_n[q, :]:
            #     cluster[p] = PointClass.DEFFERED
            #     deffered_p.add(p)

        # for each unclassified point that is dense
        for p in np.argwhere(self.is_dense(ndf))[:, 0]:

            if not cluster[p] == PointClass.UNCLASSIFIED:
                continue

            cluster[p] = cluster_id

            dp_set.clear()
            dp_set.update(self.get_involved_link_point(p, must_link))

            # for each point from (punctured_k_neighborhood(p) \ deffered_points)
            for q in k_n[p, :]:
                if not cluster[q] == PointClass.UNCLASSIFIED:
                    continue

                cluster[q] = cluster_id
                if ndf[q] >= 1:
                    dp_set.add(q)
                # add all point r from must_link(q) such r.ndf >=1 to dp_set
                dp_set.update(self.get_involved_link_point(q, must_link))

            while(len(dp_set) != 0):
                s = dp_set.pop()

                # for each unclassified point from (punctured_k_neighborhood(s) \ deffered_points)
                for t in k_n[s, :]:
                    if not cluster[t] == PointClass.UNCLASSIFIED:
                        continue

                    cluster[t] = cluster_id
                    if ndf[t] >= 1:
                        dp_set.add(t)

                    # all all point u from must_link(t)
                    dp_set.update(self.get_involved_link_point(t, must_link))

            cluster_id = cluster_id + 1

        # mark all unlcassified as noise
        cluster[cluster == PointClass.UNCLASSIFIED] = PointClass.NOISE

        self.assign_deffered_point_to_clusters(
            data, deffered_p, cannot_link, k_n, ndf, cluster)

        return cluster

    def assign_deffered_point_to_clusters(self, data, deffered_p, cannot_link, k_n, ndf, cluster):
        """ Assign deffered points to clusters

            Args:
                data: pd.DataFrame
             s       data to be clustered
                deffered_p : set()
                    deffered points
                cannot_link : np.array[tuple]
                    array of cannot link pairs
        """

        # save current deffered points in auxiliar var
        a_deffered_p = set()
        t_deffered_p = copy(deffered_p)
        while(True):
            a_deffered_p.clear()

            for p in t_deffered_p:
                assert p in t_deffered_p
                assert cluster[p] == PointClass.DEFFERED

                for q in k_n[p, :]:
                    if self.is_dense(ndf[q]) and cluster[q] != PointClass.NOISE and cluster[q] != PointClass.DEFFERED:

                        p_target_cluster = set(cluster[cluster == cluster[q]].index)
                        p_cannot_link = self.get_involved_link_point(p, cannot_link)

                        # if there are no points in the target cluster that cannot be linked
                        is_target_cluster_linkable = not p_target_cluster & p_cannot_link

                        if is_target_cluster_linkable:
                            cluster[p] = cluster[q]
                            a_deffered_p.add(p)

                            break
                t_deffered_p = t_deffered_p - a_deffered_p

            if len(a_deffered_p) == 0:
                break

        # Mark all deffered points as noise
        cluster[t_deffered_p] = PointClass.NOISE

    def is_dense(self, ndf):
        return ndf >= 1

    def unpack_link(self, link):
        """ Converts link object to plain list

            E.g.:
                [(1,2), (3,4)] => [1,2,3,4]
        """
        return [item for t in link for item in t]

    def get_involved_link_point(self, p_idx, link):
        # filter elements from the link object that contains p_idx
        l1 = filter(lambda x: p_idx in x, link)
        # get the second element (not p_idx)
        return set(map(lambda x: x[0] if x[0] != p_idx else x[1], l1))
