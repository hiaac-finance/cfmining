# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
import pandas as pd
import sklearn
import shap
from functools import lru_cache
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from cfmining.utils import TreePipeExplainer, DeepPipeExplainer

def mean_plus_dev_error(y_ref, y_pred, dev=2):
    err = abs(y_ref - y_pred)
    return np.mean(err) + dev * np.std(err)


def mean_error(y_ref, y_pred):
    err = abs(y_ref - y_pred)
    return np.mean(err)


def metric(clf, X, y):
    return mean_error(y, clf.predict_proba(X)[:, 1])


def replace(X, X_rep, col):
    newX = X.copy()
    newX[:, col] = X_rep[:, col]
    return newX


def calc_imp(X, y, individual, clf, action_set, repetitions=1):
    X_base = np.concatenate([X.values for i in range(repetitions)], axis=0)
    grid_ = action_set.feasible_grid(
        individual,
        return_actions=False,
        return_percentiles=False,
        return_immutable=True,
    )
    # X_rep = np.array([[np.random.choice(action._grid) for action in action_set] for i in range(X_base.shape[0])])
    X_rep = np.array(
        [
            [np.random.choice(grid_[action.name]) for action in action_set]
            for i in range(X_base.shape[0])
        ]
    )
    importance = np.array(
        [
            metric(clf, replace(X_base, X_rep, col), clf.predict_proba(X_base)[:, 1])
            for col in range(X.shape[1])
        ]
    )
    return importance


class GeneralClassifier:
    """Wrapper to general type of classifer.
    It estimates the importance of the features and assume the classifier as non-monotone.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    """

    def __init__(self, classifier, outlier_clf=None, X=None, threshold=0.5):
        self.clf = classifier
        self.threshold = threshold
        self.use_predict_max = True
        self.feature_names = X.columns.tolist()

        # calculate importances
        X100 = X.sample(100 if len(X) > 1000 else min(100, X.shape[0]))

        def predict_proba_aux(x):
            if type(x) != pd.DataFrame:
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x = pd.DataFrame(data=x, columns=X.columns)
            return self.clf.predict_proba(x)[:, 1]

        explainer = shap.Explainer(predict_proba_aux, X100)
        shap_values = explainer(X100)
        self.importances = np.abs(shap_values.values).mean(0)

        self.outlier_clf = outlier_clf

    @property
    def feat_importance(self):
        return self.importances

    @property
    def monotone(self):
        return False

    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value) >= self.threshold:
            return True
        else:
            return False

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        if isinstance(value, np.ndarray):
            value = value.reshape(1, -1)
            value = pd.DataFrame(data=value, columns=self.feature_names)
        return self.clf.predict_proba(value)[0, 1]

    def predict_outlier(self, value):
        """Predicts if the sample is an outlier. (Placeholder)"""
        if self.outlier_clf is None:
            return 1

        if value.ndim == 1:
            value = value.reshape(1, -1)
        return self.outlier_clf.predict(value)


class MonotoneClassifier(GeneralClassifier):
    """Wrapper to general type of classifer.
    It estimates the importance of the features and assume the classifier as monotone.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    outlier_classifier : sklearn type classifer,
        General classifier with predict, return 1 if the sample is a outlier.
    X : numpy array,
        Input Samples
    threshold : float,
        User defined threshold for the classification.
    """

    def __init__(self, classifier, outlier_clf=None, X=None, threshold=0.5):
        super().__init__(classifier, outlier_clf, X, threshold)
        self.use_prob_max = True

        # calculate action_max
        min_values = X.values.min(0)
        max_values = X.values.max(0)
        self.action_max = []
        sample = X.iloc[[0]].values
        for i in range(X.shape[1]):
            sample_min_value = sample.copy()
            sample_min_value[0, i] = min_values[i]
            sample_min_value = pd.DataFrame(data=sample_min_value, columns=X.columns)
            prob_min_value = classifier.predict_proba(sample_min_value)[0, 1]
            sample_max_value = sample.copy()
            sample_max_value[0, i] = max_values[i]
            sample_max_value = pd.DataFrame(data=sample_max_value, columns=X.columns)
            prob_max_value = classifier.predict_proba(sample_max_value)[0, 1]

            self.action_max.append(
                min_values[i] if prob_min_value > prob_max_value else max_values[i]
            )

    @property
    def monotone(self):
        return True

    def predict_max(self, value, open_vars, n_changes=None):
        """Calculates probability of achieving desired classification."""
        assert type(value) == np.ndarray
        value_copy = value.copy()
        for i in open_vars:
            value_copy[i] = self.action_max[i]
        if value_copy.ndim == 1:
            value_copy = value_copy.reshape(1, -1)
        value_copy = pd.DataFrame(data=value_copy, columns=self.feature_names)

        return self.clf.predict_proba(value_copy)[0, 1]


class GeneralClassifier_Shap:
    """Wrapper to general type of classifer.
    It estimates the importance of the features using SHAP.

    Parameters
    ----------
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    outlier_classifier : sklearn type classifer,
        General classifier with predict, return 1 if the sample is an outlier.
    X : pandas DataFrame,
        Input Samples
    categorical_features : list, optional
        List of features names that are categorical, by default []
    method_predict_max : str, optional
        Must be in ["shap", "monotone"], by default "shap"
    shap_explainer : str, optional
        Shap explainer utilized, must be in ["permutation", "tree", "custom"], by default "permutation"
    explainer : object that return shap explanations , optional
        Explainer to be utilized if "shap_explainer"="custom" , by default None
    threshold : float, optional
        Threhsold utilized for prediction, by default 0.5
    """

    def __init__(
        self,
        classifier,
        outlier_classifier,
        X=None,
        categorical_features=[],
        method_predict_max="shap",
        shap_explainer="permutation",
        explainer=None,
        threshold=0.5,
    ):
        self.clf = classifier
        self.outlier_clf = outlier_classifier
        self.feature_names = X.columns.tolist()
        self.categorical_features = categorical_features
        self.method_predict_max = method_predict_max
        self.threshold = threshold
        self.use_predict_max = True

        X100 = X.sample(1000 if len(X) > 1000 else min(100, X.shape[0]))
        if shap_explainer == "permutation":
            predict_proba = lambda x: self.clf.predict_proba(x)[:, 1]
            self.explainer = shap.Explainer(predict_proba, X100)
        elif shap_explainer == "tree":
            self.explainer = shap.TreeExplainer(
                self.clf,
                X100,
                model_output="probability",
                feature_perturbation="interventional",
            )
        elif shap_explainer == "tree_pipe":
            self.explainer = TreePipeExplainer(
                self.clf,
                X100,
                model_output="probability",
                feature_perturbation="interventional",
            )
        elif shap_explainer == "deep":
            self.explainer = shap.DeepExplainer(
                self.clf,
                X100,
            )
        elif shap_explainer == "deep_pipe":
            self.explainer = DeepPipeExplainer(
                self.clf,
                X100,
            )
        elif shap_explainer == "custom":
            self.explainer = explainer

        self.shap_values = self.explainer(X100)
        self.importances = np.abs(self.shap_values.values).mean(0)
        self.calculate_categorical_importances(X100)

        if method_predict_max == "shap":
            self.shap_max = self.shap_values.values.max(0)
            self.idx_sort_shap_max = np.argsort(self.shap_max)[::-1].tolist()

        elif method_predict_max == "monotone":
            min_values = X.values.min(0)
            max_values = X.values.max(0)

            sample = X.iloc[[0]].values
            self.action_max = []
            for i in range(X.shape[1]):
                sample_min_value = sample.copy()
                sample_min_value[0, i] = min_values[i]
                sample_min_value = pd.DataFrame(
                    data=sample_min_value, columns=self.feature_names
                )
                prob_min_value = self.clf.predict_proba(sample_min_value)[0, 1]
                sample_max_value = sample.copy()
                sample_max_value[0, i] = max_values[i]
                sample_max_value = pd.DataFrame(
                    data=sample_max_value, columns=self.feature_names
                )
                prob_max_value = self.clf.predict_proba(sample_max_value)[0, 1]

                if prob_min_value > prob_max_value:
                    self.action_max.append(min_values[i])
                else:
                    self.action_max.append(max_values[i])
        self.shap_explanation = lru_cache(maxsize=10000)(self.shap_explanation)
        self._predict_proba = lru_cache(maxsize=10000)(self._predict_proba)
        self._predict_outlier = lru_cache(maxsize=10000)(self._predict_outlier)

    @property
    def feat_importance(self):
        return self.importances

    @property
    def monotone(self):
        return False

    def calculate_categorical_importances(self, X):
        """Calculate the feature importance of categorical features based on shap values."""

        cat_importances = {}
        for col in self.categorical_features:
            cat_importances[col] = {}
            col_idx = self.feature_names.index(col)
            feature_values = X[col].unique()
            for value in feature_values:
                idx = np.where((X[col] == value).values)[0]
                shap_values_feat = self.shap_values.values[idx, col_idx]
                cat_importances[col][value] = shap_values_feat.mean()

        self.cat_importances = cat_importances

    def update_categorical_grid(self, grid):
        """Update the grid of categorical feature values.

        Parameters
        ----------
        grid : dict,
            Dictionary with the grid of actions (features).

        """
        for col in self.categorical_features:
            imp = self.cat_importances[col]
            # check if there is value not in imp
            for v in grid[col]:
                if v not in imp:
                    imp[v] = -np.inf

            imp = list(imp.items())
            imp = sorted(imp, key=lambda x: x[1], reverse=True)
            grid[col] = [x for x, _ in imp]
        return grid

    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value) >= self.threshold:
            return True
        else:
            return False

    def _predict_proba(self, value):
        """Cached function to calculate the probability."""
        value = pd.DataFrame(data=[value], columns=self.feature_names)
        return self.clf.predict_proba(value)[0, 1]

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        return self._predict_proba(tuple(value))

    def _predict_outlier(self, value):
        """Cache function to predict if the sample is an outlier."""
        value = np.array([value])
        return self.outlier_clf.predict(value)[0]

    def predict_outlier(self, value):
        """Predicts if the sample is an outlier."""
        return self._predict_outlier(tuple(value))

    def shap_explanation(self, value):
        """Calculates the shap explanation for a specific sample."""
        value = pd.DataFrame(data=[value], columns=self.feature_names)
        return self.explainer(value)[0].values

    def clear_cache(self):
        """Clears the cache of the shap explanation."""
        self.shap_explanation.cache_clear()
        self._predict_proba.cache_clear()
        self._predict_outlier.cache_clear()

    def set_pivot(self, pivot):
        self.pivot = pivot
        self.pivot_explanation = self.shap_explanation(tuple(pivot))

    def predict_max(self, value, open_vars, n_changes=None):
        """Calculates probability of achieving desired classification."""

        if self.method_predict_max == "shap":
            shap_individual = self.shap_explanation(tuple(value))
            prob = self.predict_proba(value)

            if n_changes is not None:
                open_vars_big = [i for i in self.idx_sort_shap_max if i in open_vars]
                open_vars_big = open_vars_big[:n_changes]
                prob = (
                    prob
                    - shap_individual[open_vars_big].sum()
                    + self.shap_max[open_vars_big].sum()
                )
            else:
                prob = (
                    prob
                    - shap_individual[open_vars].sum()
                    + self.shap_max[open_vars].sum()
                )

            return prob

        elif self.method_predict_max == "monotone":
            value_copy = value.copy()
            for i in open_vars:
                value_copy[i] = self.action_max[i]
            return self.predict_proba(value_copy)

    def estimate_predict_max(self, value, open_vars, n_changes=None):
        """Estimates the maximal probability of a partial sample."""
        prob = self.predict_proba(value)
        if n_changes is not None:
            # get index of n_changes most important features
            open_vars_big = [i for i in self.idx_sort_shap_max if i in open_vars]
            open_vars_big = open_vars_big[:n_changes]
            prob = (
                prob
                - self.pivot_explanation[open_vars_big].sum()
                + self.shap_max[open_vars_big].sum()
            )
        else:
            prob = (
                prob
                - self.pivot_explanation[open_vars].sum()
                + self.shap_max[open_vars].sum()
            )

        if prob < self.threshold:
            prob = self.predict_max(value, open_vars, n_changes)

        return prob


class LinearClassifier:
    """Wrapper to linear classifers.
    Calculates the importance using the coeficients from linear classification and assume the classifier as monotone.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    """

    def __init__(self, classifier, X=None, y=None, metric=None, threshold=0.5):
        self.clf = classifier
        self.threshold = threshold
        if type(self.clf) is sklearn.pipeline.Pipeline:
            try:
                coeficients = self.clf["clf"].coef_[0] / self.clf["std"].scale_
            except KeyError:
                print("sdadsa")
                coeficients = self.clf["clf"].coef_[0]
        else:
            coeficients = self.clf.coef_[0]

        self.importances = abs(coeficients * X.std(axis=0))

    @property
    def feat_importance(self):
        return self.importances

    @property
    def monotone(self):
        return True

    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value) >= self.threshold:
            return True
        else:
            return False

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        return self.clf.predict_proba([value])[0, 1]


class LinearRule:
    def __init__(self, coef, Xtrain, threshold=0):
        self.coef = coef
        self.threshold = threshold
        self.Xtrain = Xtrain

    @property
    def feat_importance(self):
        return abs(self.coef * self.Xtrain.mean(axis=0))

    @property
    def monotone(self):
        return True

    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.coef @ value >= self.threshold:
            return True
        else:
            return False

    def predict_proba(self, value):
        """Predicts if the probability is higher than threshold"""
        return self.coef @ value


from cfmining.predictors_utils import TreeExtractor


class TreeClassifier(GeneralClassifier, TreeExtractor):
    """Wrapper tree based classifiers.
    It extracts the information from the branching to speed-up the prediction based on the reference sample.
    It estimates the importance of the features and assume the classifier as non-monotone
    but explores the structure of the tree to calculate the maximal probability of a branch in the optimization procedure.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    use_predict_max : bool,
        Set if the wrapper is allowed to calculate the maximal probability of a optimization branch.
    clf_type : float (sklearn or lightgbm),
        Family of tree classifier.
    """

    def __init__(
        self,
        classifier,
        X=None,
        y=None,
        metric=None,
        threshold=0.5,
        use_predict_max=False,
        clf_type="sklearn",
    ):
        super().__init__(classifier, X, y, metric, threshold)
        self.clf = classifier
        self.clf_type = clf_type
        self.threshold = threshold
        self.use_predict_max = use_predict_max

    def fit(self, individual, action_set):
        self.names = list(action_set.df["name"])
        grid_ = action_set.feasible_grid(
            individual,
            return_actions=False,
            return_percentiles=False,
            return_immutable=True,
        )
        grid_ = {idx: grid_[name] for idx, name in enumerate(grid_)}
        self.extract_tree(grid_)

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            for leaf in leaves_tree:
                for v, name in zip(value[leaf["used_features"]], leaf["used_features"]):
                    if v not in leaf["variables"][name]:
                        break
                else:
                    prediction += leaf["prediction"]
                    break
        if self.clf_type == "sklearn":
            return prediction / self.n_estimators
        elif self.clf_type == "lightgbm":
            return 1 / (1 + np.exp(-prediction))

    def predict_max_(self, value, fixed_vars):
        """Calculates the maximal probability of a optimization branch."""
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            prob = self.lowest_value
            for leaf in leaves_tree:
                for v, name in zip(value[fixed_vars], fixed_vars):
                    if v not in leaf["variables"][name]:
                        break
                else:
                    prob = max(prob, leaf["prediction"])
            prediction += prob
        if self.clf_type == "sklearn":
            return prediction / self.n_estimators
        elif self.clf_type == "lightgbm":
            return 1 / (1 + np.exp(-prediction))

    def predict_max(self, value, fixed_vars):
        """Calculates the maximal probability of a optimization branch."""
        fixed = set(fixed_vars)
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            prob = -np.inf
            for leaf in leaves_tree:
                feat_u = list(fixed.intersection(leaf["used_features"]))
                for v, name in zip(value[feat_u], feat_u):
                    if v not in leaf["variables"][name]:  # and name in fixed_vars:
                        break
                else:
                    prob = max(prob, leaf["prediction"])
            prediction += prob
        if self.clf_type == "sklearn":
            return prediction / self.n_estimators
        elif self.clf_type == "lightgbm":
            return 1 / (1 + np.exp(-prediction))


class MonotoneTree(TreeClassifier):
    """Wrapper tree based classifiers.
    It extracts the information from the branching to speed-up the prediction based on the reference sample.
    It estimates the importance of the features and assume the classifier as monotone it should only be used with lightgbm with monotonicity constraint.
    but explores the structure of the tree to calculate the maximal probability of a branch in the optimization procedure.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    use_predict_max : bool,
        Set if the wrapper is allowed to calculate the maximal probability of a optimization branch.
    clf_type : float (sklearn or lightgbm),
        Family of tree classifier.
    """

    @property
    def monotone(self):
        return True

