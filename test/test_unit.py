import pandas as pd
from cnbc.cnbc import eps_neighborhood


def test_eps_neighbiorhood():
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

    # Act
    eps_n = eps_neighborhood(df, p_idx, eps=0.3, dist_method="euclidean")

    # Assert
    assert len(eps_n) == 2
    assert 3 in eps_n
    assert 4 in eps_n
