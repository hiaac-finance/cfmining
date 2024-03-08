import numpy as np


class OutlierWrap:
    def __init__(self, outlier_clf, threshold):
        self.outlier_clf = outlier_clf
        self.threshold = threshold

    def predict(self, X):
        pred = self.outlier_clf.predict(X)
        pred = np.where(pred < self.threshold, 1, -1)
        return pred


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
