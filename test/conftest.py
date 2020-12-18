import pandas as pd
import pytest
import os


@pytest.fixture
def iris_sample():
    data = pd.read_csv(f"{os.path.dirname(__file__)}/data/iris_sample.csv")

    X = data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
    y = data["class"]

    return (X, y)
