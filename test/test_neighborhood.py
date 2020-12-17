import pandas as pd
from cnbc.neighborhood import *


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
    eps_n = eps_neighborhood(dist_ndarr, eps=0.3)
    
    # Assert

    # get eps_neighborhood for p_idx point
    peps_n = eps_n[eps_n[:, 0] == p_idx, 1]
    assert len(peps_n) == 3
    assert p_idx in peps_n
    assert 3 in peps_n
    assert 4 in peps_n

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
    k_n = k_neighborhood(dist_ndarr, k=k)

    # Assert
    
    # get eps_neighborhood for p_idx point
    pk_n = k_n[p_idx, :]
    assert len(pk_n) == k
    assert p_idx in pk_n
    assert 1 in pk_n


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
    k_n = k_neighborhood(dist_ndarr, k=k)
    p_k_n = punctured_k_neighborhood(k_n)

    # Assert
    
    # get eps_neighborhood for p_idx point
    pp_k_n = p_k_n[p_idx, :]
    assert len(pp_k_n) == k-1
    assert not p_idx in pp_k_n
    assert 1 in pp_k_n


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
    k_n = k_neighborhood(dist_ndarr, k=k)
    p_k_n = punctured_k_neighborhood(k_n)
    r_k_n = reversed_k_neighborhood(p_k_n)

    # Assert
    assert len(r_k_n) == 3
    assert not p_idx in r_k_n
    assert 0 in r_k_n
    assert 3 in r_k_n
    assert 4 in r_k_n


def test_neighborhood_dense_factor():
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
    ndf = neighborhood_dense_factor(dist_ndarr, p_idx, k=k)

    # Assert
    assert type(ndf) == float
    assert ndf == 0.5
