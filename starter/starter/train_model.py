# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

import pandas as pd
from collections import namedtuple
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model, compute_slice_metrics

# Add code to load in the data.

data = pd.read_csv("starter/data/census.csv", skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train_df, test_df = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

ProcessedData = namedtuple("ProcessedData", ["X", "y", "encoder", "lb"])

train = ProcessedData(*process_data(
    train_df, categorical_features=cat_features, label="salary", training=True
))

# Proces the test data with the process_data function.

test = ProcessedData(*process_data(
    test_df, categorical_features=cat_features, label="salary", training=False, encoder=train.encoder, lb=train.lb
))

# Train and save a model.

model = train_model(train.X, train.y)

save_model(model, "starter/model/trained_model.pkl")
save_model(train.encoder, "starter/model/encoder.pkl")
save_model(train.lb, "starter/model/lb.pkl")

# Sanity check the results

preds = inference(model, test.X)

precision, recall, fbeta = compute_model_metrics(test.y, preds)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {fbeta:.3f}")

compute_slice_metrics(
    model, test_df, cat_features, "salary", train.encoder, train.lb,
    output_path="slice_output.txt"
)
