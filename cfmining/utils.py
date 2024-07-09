import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import torch
import joblib
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde as kde
from cfmining.criteria import *


VAL_RATIO = 1 / 7
TEST_RATIO = 3 / 10
SEED = 0

class FakeOutlierDetection:
    def predict(self, X):
        return [1]

class OutlierWrap:
    def __init__(self, X, outlier_clf, percentile):
        self.outlier_clf = outlier_clf
        self._percentile = percentile
        preds = self.outlier_clf.predict(X)
        # build interpolator that returns a threshold for each percentile
        kde_estimator = kde(preds)
        cdf = np.cumsum(kde_estimator(np.linspace(0, 1, 101)))
        self.percentile_interp = interp1d(
            cdf,
            np.linspace(0, 1, 101),
            copy=False,
            bounds_error=False,
            assume_sorted=True,
        )
        self._threshold = self.percentile_interp(100 * (1 - percentile))

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        self._percentile = value
        self._threshold = self.percentile_interp(100 * (1 - value))

    def predict(self, X):
        pred = self.outlier_clf.predict(X)
        pred = np.where(pred < self._threshold, 1, -1)
        return pred

    def score(self, X):
        pred = self.outlier_clf.predict(X)
        return pred


def get_data_model(dataset, model_name="LGBMClassifier"):
    """Helper function to load the dataset and model."""
    if dataset in ["german", "taiwan"]:
        df = pd.read_csv(f"../data/{dataset}.csv")
    elif dataset == "german_small":
        df = pd.read_csv(f"../data/german.csv")
    if dataset == "german":
        X = df.drop("GoodCustomer", axis=1)
        Y = df["GoodCustomer"]
    if dataset == "german_small":
        X = df.drop("GoodCustomer", axis=1)
        X = X[["LoanAmount", "LoanDuration", "OwnsHouse", "is_male"]]
        Y = df["GoodCustomer"]
    elif dataset == "taiwan":
        X = df.drop("NoDefaultNextMonth", axis=1)
        Y = df["NoDefaultNextMonth"]

    X_train, X_test, Y_train, _ = train_test_split(
        X, Y, test_size=TEST_RATIO, random_state=SEED, shuffle=True
    )

    outlier_detection = joblib.load(f"../models/{dataset}/IsolationForest.pkl")
    model = joblib.load(f"../models/{dataset}/{model_name}.pkl")
    denied_individ = model.predict(X_test) == 0
    individuals = X_test.iloc[denied_individ].reset_index(drop=True)

    if dataset == "taiwan":
        X_train = X_train.astype(int)
        individuals = individuals.astype(int)

    return X_train, Y_train, model, outlier_detection, individuals


def diversity_metric(solutions):
    """Measure the diversity metric of solutions."""
    l1_dist_matrix = np.abs(solutions[:, None] - solutions[None, :]).sum(axis=2).astype(np.float64)
    e = np.eye(len(solutions), dtype=np.float64) * 1e-4
    l1_dist_matrix += e
    K = 1 / (1 + l1_dist_matrix)
    return np.linalg.det(K)


def proximity_metric(individual, solutions):
    """Measure the proximity metric of solutions, i.e., the sum of L1 distances."""
    return np.sum(np.abs(individual - solutions), axis=1)


def sparsity_metric(individual, solutions):
    """Measure the sparsity metric of solutions, i.e., the number of changes."""
    return np.sum(individual != solutions, axis=1)


class DeepPipeExplainer:
    """Wrap class that handles pipeline with shap.DeepExplaienr"""

    def __init__(self, pipeline, background_data):
        self.preprocess = pipeline[:-1]
        self.model = pipeline[-1].model
        self.background_data = self.preprocess.transform(background_data)

        if type(self.background_data) == pd.DataFrame:
            self.background_data = self.background_data.values
        self.explainer = shap.DeepExplainer(
            self.model, torch.Tensor(self.background_data)
        )

    def __call__(self, X):
        X = self.preprocess.transform(X)
        if type(X) == pd.DataFrame:
            X = X.values
        values = self.explainer.shap_values(torch.Tensor(X))[:, :, 1]
        # return an explanation object
        return shap.Explanation(values)

    def explain_row(self, X):
        X = self.preprocess.transform(X)
        if type(X) == pd.DataFrame:
            X = X.values
        return self.explainer.shap_values(torch.Tensor(X))
