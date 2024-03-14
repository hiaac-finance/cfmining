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

    def __init__(self, classifier, X=None, y=None, metric=None, threshold=0.5):
        self.clf = classifier
        self.threshold = threshold
        from eli5.sklearn import PermutationImportance
        from sklearn.metrics import roc_auc_score, mean_absolute_error

        def deviat(clf, X, y):
            if metric is None:
                # return mean_absolute_error(y, clf.predict_proba(X)[:,1])
                return roc_auc_score(y, clf.predict_proba(X)[:, 1])
            else:
                return metric(y, clf.predict_proba(X)[:, 1])

        perm = PermutationImportance(self.clf, scoring=deviat, n_iter=10).fit(X, y)
        # perm = PermutationImportance(self.clf, scoring=deviat).fit(X, classifier.predict_proba(X)[:,1])
        self.importances = abs(perm.feature_importances_)

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
        return self.clf.predict_proba([value])[0, 1]


class GeneralClassifier_Shap:
    """Wrapper to general type of classifer.
    It estimates the importance of the features using SHAP.

    Parameters
    ----------

    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    threshold : float,
        User defined threshold for the classification.
    """

    def __init__(
        self,
        classifier,
        outlier_classifier,
        X=None,
        categorical_features=[],
        method_predict_max="shap",
        tree = False,
        threshold=0.5,
    ):
        self.clf = classifier
        self.outlier_clf = outlier_classifier
        self.threshold = threshold
        self.feature_names = X.columns.tolist()
        self.n_features = X.shape[1]
        self.categorical_features = categorical_features
        self.method_predict_max = method_predict_max
        self.tree = tree
        self.use_log_odds = True if not tree else False
        def predict_proba(x):
            x = pd.DataFrame(x, columns=self.feature_names)
            p = self.clf.predict_proba(x)[:, 1]
            if self.use_log_odds:
                log_odds = np.log(p / (1 - p))
                return log_odds
            return p
    
        X100 = X.sample(100)
        if not tree:
            self.explainer = shap.Explainer(predict_proba, X100)
        else:
            self.explainer = shap.TreeExplainer(self.clf, X100, model_output="probability", feature_perturbation="interventional")

        self.shap_values = self.explainer(X)
        self.calculate_categorical_importances(X)
        self.importances = np.abs(self.shap_values.values).mean(0)
        if method_predict_max == "shap":
            self.shap_max = self.shap_values.values.max(0)
            if self.use_log_odds:
                self.shap_max = np.exp(self.shap_max) / (1 + np.exp(self.shap_max))
        elif method_predict_max == "monotone":
            min_values = X.values.min(0)
            max_values = X.values.max(0)

            sample = X.iloc[[0]].values
            self.action_max = []
            for i in range(self.n_features):
                sample_min_value = sample.copy()
                sample_min_value[0, i] = min_values[i]
                prob_min_value = self.clf.predict_proba(sample_min_value)[0, 1]
                sample_max_value = sample.copy()
                sample_max_value[0, i] = max_values[i]
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
    
    def update_categorical_grid(self, col, grid):
        """Update the grid of categorical feature values for a specific feature."""
        imp = self.cat_importances[col]
        value = [imp[v] if v in imp else -np.inf for v in grid]
        new_grid = grid.copy()
        new_grid = [x for _, x in sorted(zip(value, new_grid))]
        return new_grid
                
    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value) >= self.threshold:
            return True
        else:
            return False

    def _predict_proba(self, value):
        """Cached function to calculate the probability."""
        value = np.array([value])
        return self.clf.predict_proba(value)[0, 1]

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
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
        value = np.array([value])
        return self.explainer(value)[0].values
    
    def clear_cache(self):
        """Clears the cache of the shap explanation."""
        self.shap_explanation.cache_clear()
        self._predict_proba.cache_clear()
        self._predict_outlier.cache_clear()


    def predict_max(self, value, open_vars):
        """Calculates probability of achieving desired classification."""
        
        if self.method_predict_max == "shap":
            shap_individual = self.shap_explanation(tuple(value))
            prob = self.predict_proba(value)
            if self.use_log_odds:
                shap_individual = np.exp(shap_individual) / (1 + np.exp(shap_individual))
            return (
                prob - shap_individual[open_vars].sum() + self.shap_max[open_vars].sum()
            )
        elif self.method_predict_max == "monotone":
            value_copy = value.copy()
            for i in open_vars:
                value_copy.iloc[0, i] = self.action_max[i]
            return self.clf.predict_proba(value_copy)[:, 1]


class MonotoneClassifier(GeneralClassifier):
    """Wrapper to general type of classifer.
    It estimates the importance of the features and assume the classifier as monotone.

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

    @property
    def monotone(self):
        return True


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


"""
class TreeClassifier(GeneralClassifier):
    def __init__(self, classifier, individual, action_set, 
                 X=None, y=None, metric=None, threshold=0.5,
                 use_predict_max=False, general=None):
        from actionsenum.mip_algorithms import recursive_tree_
 
        self.clf = classifier
        self.threshold = threshold

        self.names = list(action_set.df['name'])
        grid_ = action_set.feasible_grid(individual, return_actions=False, return_percentiles=False, return_immutable=True)
        leaves = [recursive_tree_(tree_, self.names, grid_, 0, tree_name=i)
                  for i, tree_ in enumerate(classifier.estimators_)]

        for leaves_tree in leaves:
            for leaf in leaves_tree:
                leaf['variables'] = {var:set(leaf['variables'][var]) for var in leaf['variables']}
                leaf['used_feat'] = list(leaf['used_feat'])[::-1]
                leaf['used_idx'] = [self.names.index(name) for name in leaf['used_feat']]

        self.leaves = leaves

        if general is not None:
            self.importances = general.importances
        else:
            self.importances = classifier.feature_importances_
        self.use_predict_max = use_predict_max

    def predict_proba(self, value):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            for leaf in leaves_tree:
                active_leaf = True
                for v, name in zip(value[leaf['used_idx']], leaf['used_feat']):
                    if v not in leaf['variables'][name]:
                        active_leaf=False
                        break
                if active_leaf:
                    prediction+=leaf['prediction']
                    break
        return prediction/n_estimators

    def predict_max(self, value, fixed_vars):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            prob = 0
            for leaf in leaves_tree:
                active_leaf = True
                for v, idx, name in zip(value[leaf['used_idx']], leaf['used_idx'], leaf['used_feat']):
                    if v not in leaf['variables'][name] and idx in fixed_vars:
                        active_leaf=False
                        break
                if active_leaf:
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        #print(fixed_vars, prediction/n_estimators)
        return prediction/n_estimators


class TreeClassifier2(GeneralClassifier):
    def __init__(self, classifier,
                 X=None, y=None, metric=None, threshold=0.5,
                 use_predict_max=False):
 
        super().__init__(classifier, X, y, metric, threshold)
        self.clf = classifier
        self.threshold = threshold
        self.use_predict_max = use_predict_max

    def fit(self, individual, action_set):
        from actionsenum.mip_algorithms import recursive_tree_
        self.names = list(action_set.df['name'])
        grid_ = action_set.feasible_grid(individual, return_actions=False, return_percentiles=False, return_immutable=True)
        leaves = [recursive_tree_(tree_, self.names, grid_, 0, tree_name=i)
                  for i, tree_ in enumerate(self.clf.estimators_)]

        for leaves_tree in leaves:
            for leaf in leaves_tree:
                leaf['variables'] = {var:set(leaf['variables'][var]) for var in leaf['variables']}
                leaf['used_feat'] = set(leaf['used_feat'])
                leaf['used_idx'] = [self.names.index(name) for name in leaf['used_feat']]

        self.leaves = leaves

    def predict_proba(self, value):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            for leaf in leaves_tree:
                active_leaf = True
                for v, name in zip(value[leaf['used_idx']], leaf['used_feat']):
                    if v not in leaf['variables'][name]:
                        active_leaf=False
                        break
                if active_leaf:
                    prediction+=leaf['prediction']
                    break
        return prediction/n_estimators

    def predict_max(self, value, fixed_vars):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            prob = 0
            for leaf in leaves_tree:
                active_leaf = True
                for v, idx, name in zip(value[leaf['used_idx']], leaf['used_idx'], leaf['used_feat']):
                    if v not in leaf['variables'][name] and idx in fixed_vars:
                        active_leaf=False
                        break
                if active_leaf:
                    bla = leaf['tree'], leaf['name']
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        return prediction/n_estimators
"""
