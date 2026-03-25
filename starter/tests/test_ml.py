import tempfile
import os
import numpy as np
import pandas as pd
import pytest

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, save_model, load_model


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 35, 45, 55],
        "workclass": ["Private", "Self-emp", "Private", "Gov"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    })


@pytest.fixture
def processed_train(sample_df):
    X, y, encoder, lb = process_data(
        sample_df,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )
    return X, y, encoder, lb


def test_process_data_training(sample_df):
    X, y, encoder, lb = process_data(
        sample_df,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )
    assert X.shape[0] == len(sample_df)
    assert len(y) == len(sample_df)
    assert encoder is not None
    assert lb is not None


def test_process_data_inference(sample_df, processed_train):
    _, _, encoder, lb = processed_train
    X, y, _, _ = process_data(
        sample_df,
        categorical_features=["workclass"],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert X.shape[0] == len(sample_df)


def test_process_data_no_label(sample_df, processed_train):
    _, _, encoder, lb = processed_train
    X, y, _, _ = process_data(
        sample_df.drop("salary", axis=1),
        categorical_features=["workclass"],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert len(y) == 0


def test_train_model(processed_train):
    X, y, _, _ = processed_train
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")


def test_inference(processed_train):
    X, y, _, _ = processed_train
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_save_and_load_model(processed_train):
    X, y, _, _ = processed_train
    model = train_model(X, y)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_model(model, path)
        loaded = load_model(path)
        assert np.array_equal(inference(model, X), inference(loaded, X))
    finally:
        os.unlink(path)
