from .neighborhood import calc_distance_ndarray, neighborhood_dense_factor


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
