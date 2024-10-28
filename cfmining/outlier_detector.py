import numpy as np
from sklearn.base import OutlierMixin
from isotree import IsolationForest as Isof


class FakeOutlierDetector(OutlierMixin):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.ones(X.shape[0])


class IsolationForest:
    def __init__(self, contamination=0.1, **kwargs):

        self._contamination = contamination
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.model = Isof(**self.kwargs)
        self.model.fit(X)
        self.scores = self.model.predict(X)
        self.threshold = np.percentile(self.scores, 100 * (1 - self.contamination))

    @property
    def contamination(self):
        return self._contamination

    @contamination.setter
    def contamination(self, value):
        self._contamination = value
        self.threshold = np.percentile(self.scores, 100 * (1 - value))

    def predict(self, X):
        """Return 1 for inliers, -1 for outliers."""
        scores = self.model.predict(X)
        return np.where(scores > self.threshold, -1, 1)
