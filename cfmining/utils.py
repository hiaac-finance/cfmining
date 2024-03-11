import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

VAL_RATIO = 1 / 7
TEST_RATIO = 3 / 10
SEED = 0


class OutlierWrap:
    def __init__(self, outlier_clf, threshold):
        self.outlier_clf = outlier_clf
        self.threshold = threshold

    def predict(self, X):
        pred = self.outlier_clf.predict(X)
        pred = np.where(pred < self.threshold, 1, -1)
        return pred


def get_data_model(dataset, model_name="LGBMClassifier"):
    """Helper function to load the dataset and model."""
    df = pd.read_csv(f"../data/{dataset}.csv")
    if dataset == "german":
        X = df.drop("GoodCustomer", axis=1)
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

    return X_train, Y_train, model, outlier_detection, individuals


def diversity_metric(solutions):
    """Measure the diversity metric of solutions."""
    l1_dist_matrix = np.abs(solutions[:, None] - solutions[None, :])
    print(l1_dist_matrix.shape)
    l1_dist_matrix += np.eye(len(solutions)) * 1e-4
    K = 1 / (1 + l1_dist_matrix)
    return np.linalg.det(K)


def proximity_metric(individual, solutions):
    """Measure the proximity metric of solutions, i.e., the sum of L1 distances."""
    return np.sum(np.abs(individual - solutions), axis=1)


def sparsity_metric(individual, solutions):
    """Measure the sparsity metric of solutions, i.e., the number of changes."""
    return np.sum(individual != solutions, axis=1)
