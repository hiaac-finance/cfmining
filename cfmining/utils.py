import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import torch
import joblib
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde as kde
from cfmining.criteria import *


def diversity_metric(solutions):
    """Measure the diversity metric of solutions."""
    l1_dist_matrix = (
        np.abs(solutions[:, None] - solutions[None, :]).sum(axis=2).astype(np.float64)
    )
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
