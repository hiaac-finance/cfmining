import numpy as np
import pandas as pd
import copy
import dice_ml
from nice import NICE
import json
import cfmining.algorithms as alg
from cfmining.criteria import (
    PercentileCalculator,
    PercentileCriterion,
    PercentileChangesCriterion,
    NonDomCriterion,
)


class Bruteforce:
    """Wrapper function for Bruteforce algorithm, fit expects an individual and can be called multiple times.


    Parameters
    ----------
    action_set : ActionSet
        Action set for searching counterfactuals
    model : GeneralClassifier
        Classifier with methods for predicting probability and feature importance
    criteria : string in ["percentile", "percentile_change", "non_dom"]
        Criteria for comparing multi-objective solutions
    max_changes : int
        Maximum number of changes to consider counterfactuals
    """

    def __init__(self, action_set, model, criteria, max_changes):
        self.action_set = action_set
        self.model = model
        if criteria == "percentile":
            perc_calc = PercentileCalculator(action_set=action_set)
            self.compare = lambda ind: PercentileCriterion(ind, perc_calc)
        elif criteria == "percentile_change":
            perc_calc = PercentileCalculator(action_set=action_set)
            self.compare = lambda ind: PercentileChangesCriterion(ind, perc_calc)
        elif criteria == "non_dom":
            self.compare = lambda ind: NonDomCriterion(ind)

        self.max_changes = max_changes

    def fit(self, individual):
        m = alg.BruteForce(
            self.action_set,
            individual,
            self.model,
            max_changes=self.max_changes,
            compare=self.compare(individual),
        )
        m.fit()
        self.solutions = m.solutions
        if len(self.solutions) > 0:
            if isinstance(self.solutions[0], np.ndarray):
                self.solutions = [s.tolist() for s in self.solutions]
        return self


class MAPOCAM:
    """Wrapper function for MAPOCAM algorithm, fit expects an individual and can be called multiple times.


    Parameters
    ----------
    action_set : ActionSet
        Action set for searching counterfactuals
    model : GeneralClassifier
        Classifier with methods for predicting probability and feature importance
    criteria : string in ["percentile", "percentile_change", "non_dom"]
        Criteria for comparing multi-objective solutions
    max_changes : int
        Maximum number of changes to consider counterfactuals
    """

    def __init__(self, action_set, model, criteria, max_changes):
        self.action_set = copy.deepcopy(action_set)
        for feat in self.action_set:
            feat.flip_direction = 1
            feat.update_grid()

        self.model = model
        if criteria == "percentile":
            perc_calc = PercentileCalculator(action_set=action_set)
            self.compare = lambda ind: PercentileCriterion(ind, perc_calc)
        elif criteria == "percentile_change":
            perc_calc = PercentileCalculator(action_set=action_set)
            self.compare = lambda ind: PercentileChangesCriterion(ind, perc_calc)
        elif criteria == "non_dom":
            self.compare = lambda ind: NonDomCriterion(ind)

        self.max_changes = max_changes

    def fit(self, individual):
        m = alg.MAPOCAM(
            self.action_set,
            individual,
            self.model,
            max_changes=self.max_changes,
            compare=self.compare(individual),
        )
        m.fit()
        self.solutions = m.solutions
        if len(self.solutions) > 0:
            if isinstance(self.solutions[0], np.ndarray):
                self.solutions = [s.tolist() for s in self.solutions]
        return self


class Dice:
    """Wrapper function for Dice algorithm, fit expects an individual and can be called multiple times.


    Parameters
    ----------
    data : DataFrame
        Dataframe with model features
    Y : np.ndarray or pd.Series
        Classifier target
    model : GeneralClassifier
        Sklearn classifier
    n_cfs : int
        Number of counterfactuals to generate
    mutable_features: list
        List of features that can be used on conterfactuals
    sparsity_weight: float
        Parameter, weight for sparsity in optimization problem
    """

    def __init__(
        self,
        data,
        Y,
        model,
        n_cfs,
        mutable_features,
        continuous_features,
        sparsity_weight=0.2,
    ):
        self.total_CFs = n_cfs
        self.sparsity_weight = sparsity_weight
        self.mutable_features = mutable_features
        self.features = data.columns.tolist()

        class ModelWrap:
            def __init__(self, model):
                self.model = model

            def predict_proba(self, x):
                object_columns = x.select_dtypes(include=["object"]).columns.tolist()

                def convert(x):
                    if isinstance(x, str):
                        x = int(x) if x.isdigit() else float(x)
                    return x

                for col in object_columns:
                    x[col] = x[col].apply(convert)

                return self.model.predict_proba(x)

        dice_model = dice_ml.Model(
            model=ModelWrap(model), backend="sklearn", model_type="classifier"
        )
        data_extended = data.copy()
        data_extended["target"] = Y
        dice_data = dice_ml.Data(
            dataframe=data_extended,
            continuous_features=continuous_features,
            outcome_name="target",
        )
        self.exp = dice_ml.Dice(dice_data, dice_model)

    def fit(self, individual):
        if type(individual) == np.ndarray:
            individual = pd.DataFrame(data=[individual], columns=self.features)
        dice_exp = self.exp.generate_counterfactuals(
            individual,
            total_CFs=self.total_CFs,
            desired_class="opposite",
            sparsity_weight=self.sparsity_weight,
            features_to_vary=self.mutable_features,
        )
        solutions = json.loads(dice_exp.to_json())["cfs_list"][0]
        self.solutions = [solution[:-1] for solution in solutions]
        if len(self.solutions) > 0:
            if isinstance(self.solutions[0], np.ndarray):
                self.solutions = [s.tolist() for s in self.solutions]
        return self


class Nice:
    """Wrapper function for Nice algorithm, fit expects an individual and can be called multiple times.


    Parameters
    ----------
    data : DataFrame
        Dataframe with model features
    Y : np.ndarray or pd.Series
        Classifier target
    model : GeneralClassifier
        Sklearn classifier
    cat_features: list
        List of categorical features
    """

    def __init__(self, data, Y, model, cat_features):
        predict_fn = lambda x: model.predict_proba(x)

        features = data.columns.tolist()
        num_features = [feat for feat in features if feat not in cat_features]
        self.cat_features = [features.index(feat) for feat in cat_features]
        self.num_features = [features.index(feat) for feat in num_features]

        self.exp = NICE(
            X_train=data.values,
            predict_fn=predict_fn,
            y_train=Y,
            cat_feat=self.cat_features,
            num_feat=self.num_features,
            distance_metric="HEOM",
            num_normalization="minmax",
            optimization="proximity",
            justified_cf=True,
        )

    def fit(self, individual):
        if individual.ndim == 1:
            individual = individual[None, :]
        self.solutions = self.exp.explain(individual).tolist()
        if len(self.solutions) > 0:
            if isinstance(self.solutions[0], np.ndarray):
                self.solutions = [s.tolist() for s in self.solutions]
        return self
