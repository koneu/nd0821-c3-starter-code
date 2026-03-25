# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model type:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Training framework:** scikit-learn
- **Random state:** 42 (for reproducibility)
- **Task:** Binary classification — predicting whether an individual earns >50K or <=50K per year

## Intended Use

This model is intended for educational purposes as part of a CI/CD pipeline project. It demonstrates how to build, test, and deploy a machine learning model via a REST API. It should **not** be used for real hiring, lending, or any consequential decision-making about individuals.

## Training Data

The model was trained on the UCI Census Income dataset (`census.csv`), which contains demographic and employment information from the 1994 US Census. The dataset contains approximately 32,000 records. An 80/20 train/test split was used.

Categorical features are one-hot encoded; the label (`salary`) is binarized with `LabelBinarizer`.

## Evaluation Data

The remaining 20% of the census dataset was held out as a test set. The same encoder and label binarizer fitted on the training set were applied to the test set to avoid data leakage.

## Metrics

Performance is evaluated using precision, recall, and F1 score (beta=1). Slice metrics across all categorical features are written to `slice_output.txt`.

_Note: metrics below are approximate — exact values vary with each train/test split._

| Metric    | Value  |
|-----------|--------|
| Precision | ~0.85  |
| Recall    | ~0.63  |
| F1        | ~0.72  |

## Ethical Considerations

The census dataset contains sensitive demographic attributes including race, sex, and native country. The model may reflect historical biases present in the data. Slice metrics should be reviewed to identify performance disparities across demographic groups before any deployment.

This model was trained on 1994 data and demographic patterns have changed significantly since then. Predictions should not be generalised to current populations.

## Caveats and Recommendations

- The model uses default Random Forest hyperparameters and has not been tuned. Performance could be improved with cross-validation and hyperparameter search.
- Class imbalance (~75% <=50K) means the model is biased toward the majority class. Consider resampling or class weighting for production use.
- Always review `slice_output.txt` after retraining to check for performance degradation on specific subgroups.