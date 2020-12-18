from cnbc.cnbc import *


def test_cnbc(iris_sample):
    # Assign
    X, y = iris_sample
    k = 4

    # Act
    cl = CNBC(k=k).fit(X, [], [])

    assert len(cl) == len(y)


def test_cnbc_must_link(iris_sample):
    # Assign
    X, y = iris_sample
    k = 4
    idx0 = 0
    idx1 = 3
    must_link = [[idx0, idx1]]

    # Act
    cl = CNBC(k=k).fit(X, [], [])
    cl_must_link = CNBC(k=k).fit(X, must_link=must_link, cannot_link=[])

    print(cl_must_link)

    assert len(cl) == len(y)
    assert cl[idx0] != cl[idx1]
    assert cl_must_link[idx0] == cl_must_link[idx1]
    assert cl_must_link[idx0] != "deffered"
