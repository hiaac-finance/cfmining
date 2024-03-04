import numpy as np


class OutlierWrap:
    def __init__(self, outlier_clf, threshold):
        self.outlier_clf = outlier_clf
        self.threshold = threshold

    def predict(self, X):
        pred = self.outlier_clf.predict(X)
        pred = np.where(pred < self.threshold, 1, -1)
        return pred
