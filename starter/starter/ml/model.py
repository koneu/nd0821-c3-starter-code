import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def save_model(model, path):
    """ Save a trained model to disk.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    path : str
        Path to save the model to.
    """
    joblib.dump(model, path)


def load_model(path):
    """ Load a trained model from disk.

    Inputs
    ------
    path : str
        Path to load the model from.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    return joblib.load(path)


def compute_slice_metrics(model, df, categorical_features, label, encoder, lb, output_path="slice_output.txt"):
    """ Compute model performance on slices of categorical features.

    For each unique value of each categorical feature, computes precision,
    recall and F1 on the subset of data with that value and writes results
    as a formatted table to a text file.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    df : pd.DataFrame
        Dataframe containing features and label.
    categorical_features : list[str]
        List of categorical feature names to slice on.
    label : str
        Name of the label column.
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer from training.
    output_path : str
        Path to write the slice output file.
    """
    from ml.data import process_data

    W_FEATURE, W_VALUE, W_N, W_PRECISION, W_RECALL, W_F1 = 20, 30, 6, 10, 8, 8

    header = (
        f"{'feature':<{W_FEATURE}} {'value':<{W_VALUE}} {'n':>{W_N}} "
        f"{'precision':>{W_PRECISION}} {'recall':>{W_RECALL}} {'f1':>{W_F1}}\n"
    )
    separator = "-" * len(header) + "\n"
    big_separator = "=" * len(header) + "\n"

    def _write_row(f, feature, value, subset):
        X, y, _, _ = process_data(
            subset,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        f.write(
            f"{feature:<{W_FEATURE}} {str(value):<{W_VALUE}} {len(subset):>{W_N}} "
            f"{precision:>{W_PRECISION}.3f} {recall:>{W_RECALL}.3f} {fbeta:>{W_F1}.3f}\n"
        )

    with open(output_path, "w") as f:
        f.write(header)
        f.write(big_separator)
        for feature in categorical_features:
            for value in sorted(df[feature].unique()):
                _write_row(f, feature, value, df[df[feature] == value])
            _write_row(f, feature, "[ALL]", df)
            f.write(separator)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)
