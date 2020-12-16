import pandas as pd
from cnbc.cnbc import *


def test_eps_neighborhood():
    # Assign
    p_idx = 0
    data = [
        [0.1, 0.2],
        [0.9, 0.3],
        [2, 3],
        [0.1, 0.1],
        [0.1, 0.3],
        [1, 3],
    ]
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    eps_n = eps_neighborhood(dist_ndarr, p_idx, eps=0.3)

    # Assert
    assert len(eps_n) == 3
    assert p_idx in eps_n
    assert 3 in eps_n
    assert 4 in eps_n


def test_k_neighborhood():
    # Assign
    p_idx = 0
    data = [
        [0.1, 0.2],
        [0.1, 0.3],
        [2, 3],
        [0.1, 5],
        [0.9, 0.3],
        [0.001, 0.1],
    ]
    k = 2
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    k_n = k_neighborhood(dist_ndarr, p_idx, k=k)

    # Assert
    assert len(k_n) == k
    assert p_idx in k_n
    assert 1 in k_n


def test_punctured_k_neighborhood():
    # Assign
    p_idx = 0
    data = [
        [0.1, 0.2],
        [0.1, 0.3],
        [2, 3],
        [0.1, 5],
        [0.9, 0.3],
        [0.001, 0.1],
    ]
    k = 2
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    p_k_n = punctured_k_neighborhood(dist_ndarr, p_idx, k=k)

    # Assert
    assert len(p_k_n) == k-1
    assert not p_idx in p_k_n
    assert 1 in p_k_n


def test_reversed_k_neighborhood():
    # Assign
    p_idx = 1
    data = [
        [0, 0],
        [1, 1],
        [5, 6],
        [2, 3],
        [2, 3],
        [1, 7],
    ]
    k = 3
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    r_k_n = reversed_k_neighborhood(dist_ndarr, p_idx, k=k)

    # Assert
    assert len(r_k_n) == 3
    assert not p_idx in r_k_n
    assert 0 in r_k_n
    assert 3 in r_k_n
    assert 4 in r_k_n


def test_reversed_k_neighborhood():
    # Assign
    p_idx = 1
    data = [
        [0, 0],
        [1, 1],
        [5, 6],
        [2, 3],
        [2, 3],
        [1, 7],
    ]
    k = 3
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    r_k_n = reversed_k_neighborhood(dist_ndarr, p_idx, k=k)

    # Assert
    assert len(r_k_n) == 3
    assert not p_idx in r_k_n
    assert 0 in r_k_n
    assert 3 in r_k_n
    assert 4 in r_k_n


def test_ndf():
    # Assign
    p_idx = 0
    data = [
        [0, 0],
        [1, 1],
        [5, 6],
        [2, 3],
        [2, 3],
        [1, 7],
    ]
    k = 3
    df = pd.DataFrame(data=data)
    dist_ndarr = calc_distance_ndarray(df, dist_method="euclidean")

    # Act
    ndf = neighborhood_df(dist_ndarr, p_idx, k=k)

    # Assert
    assert type(ndf) == float
    assert ndf == 0.5
