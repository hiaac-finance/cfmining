import numpy as np
from sklearn.base import OutlierMixin
from isotree import IsolationForest as Isof
from .models import AutoEncoder


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
    
class AE_OutlierDetector:
    def __init__(self, contamination=0.1, categoric_features = [], **kwargs):
        self._contamination = contamination
        self.categoric_features = categoric_features
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.model = AutoEncoder(categoric_features = self.categoric_features, **self.kwargs)
        self.model.fit(X)
        self.losses = self.model.loss(X)
        self.threshold = np.percentile(self.losses, 100 * (1 - self.contamination))

    @property
    def contamination(self):
        return self._contamination

    @contamination.setter
    def contamination(self, value):
        self._contamination = value
        self.threshold = np.percentile(self.losses, 100 * (1 - value))

    def predict(self, X):
        """Return 1 for inliers, -1 for outliers."""
        scores = self.model.loss(X)
        return np.where(scores > self.threshold, -1, 1)
