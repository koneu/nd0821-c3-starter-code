import pytest
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model

LOW_INCOME = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 185908,
    "education": "HS-grad",
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
}

HIGH_INCOME = {
    "age": 51,
    "workclass": "Private",
    "fnlgt": 108435,
    "education": "Masters",
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 47,
    "native-country": "United-States",
}


@pytest.fixture(scope="module")
def client_with_model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("model")

    df = pd.read_csv("starter/data/census.csv", skipinitialspace=True)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    train_df, _ = train_test_split(df, test_size=0.20, random_state=42)
    X, y, encoder, lb = process_data(train_df, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)

    joblib.dump(model, tmp_path / "trained_model.pkl")
    joblib.dump(encoder, tmp_path / "encoder.pkl")
    joblib.dump(lb, tmp_path / "lb.pkl")

    with patch("starter.main.MODEL_DIR", tmp_path):
        from importlib import reload
        import starter.main
        reload(starter.main)
        from starter.main import app
        with TestClient(app) as c:
            yield c


def test_get_root(client_with_model):
    response = client_with_model.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_post_inference_low_income(client_with_model):
    response = client_with_model.post("/inference", json=LOW_INCOME)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_inference_high_income(client_with_model):
    response = client_with_model.post("/inference", json=HIGH_INCOME)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_get_metrics(client_with_model):
    response = client_with_model.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"precision", "recall", "f1"}
    assert all(0.0 <= body[k] <= 1.0 for k in body)


def test_get_slices_missing(client_with_model):
    with patch("starter.main.SLICE_OUTPUT") as mock_path:
        mock_path.exists.return_value = False
        response = client_with_model.get("/slices")
    assert response.status_code == 200
    assert "error" in response.json()


def test_get_slices(client_with_model, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("slices") / "slice_output.txt"
    tmp.write_text("feature  value  n  precision  recall  f1\n")
    with patch("starter.main.SLICE_OUTPUT", tmp):
        response = client_with_model.get("/slices")
    assert response.status_code == 200
    assert "slices" in response.json()
