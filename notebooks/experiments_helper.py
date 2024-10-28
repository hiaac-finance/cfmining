import os
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import time
import joblib
import sys
from sklearn.model_selection import train_test_split
import pathos.multiprocessing as mp

sys.path.append("../")
from cfmining.criteria import PercentileCalculator, PercentileCriterion, NonDomCriterion
from cfmining.utils import diversity_metric
from cfmining.action_set import ActionSet
from cfmining.datasets import *


VAL_RATIO = 1 / 7
TEST_RATIO = 3 / 10
SEED = 0


def get_data_model(dataset_name, model_name="LGBMClassifier"):
    """Helper function to load the dataset and model."""
    if dataset_name == "german":
        dataset = GermanCredit(use_categorical=False)
    elif dataset_name == "german_cat":
        dataset = GermanCredit(use_categorical=True)
    elif dataset_name == "taiwan":
        dataset = Taiwan(use_categorical=False)
    elif dataset_name == "taiwan_cat":
        dataset = Taiwan(use_categorical=True)
    elif dataset_name == "adult":
        dataset = Adult(use_categorical=False)
    elif dataset_name == "adult_cat":
        dataset = Adult(use_categorical=True)

    X, Y = dataset.load_data()
    X_train, X_test, _, _ = train_test_split(
        X, Y, test_size=TEST_RATIO, random_state=SEED, shuffle=True
    )

    outlier_detection = joblib.load(f"../models/{dataset}/IsolationForest.pkl")
    model = joblib.load(f"../models/{dataset}/{model_name}.pkl")
    denied_individ = model.predict(X_test) == 0
    individuals = X_test.iloc[denied_individ].reset_index(drop=True)

    return dataset, X_train, model, outlier_detection, individuals

def get_action_set(dataset, X_train, default_step_size = 0.1):
    action_set = ActionSet(
        X = X_train,
        default_step_size = default_step_size,
        mutable_features = dataset.mutable_features,
        default_step_type = "relative"
    )
    return action_set


def run_experiments(
    method,
    individuals,
    model,
    output_file=None,
    n_jobs = 1,
):
    results = []

    if not output_file is None:
        folder = "/".join(output_file.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    if n_jobs == 1:
        for i in tqdm(range(len(individuals))):
            individual = individuals.iloc[i]
            try:
                model.clear_cache()
            except:
                pass
            start = time.time()
            method.fit(individual.values)
            end = time.time()

            solutions = method.solutions

            results.append(
                {
                    "individual": individual.values.tolist(),
                    "prob": model.predict_proba(individual.values),
                    "time": end - start,
                    "n_solutions": len(method.solutions),
                    "solutions": solutions,
                }
            )

    else:
        ...
    
    results = pd.DataFrame(results)
    if output_file is not None:
        results.to_csv(output_file, index=False)
    return results


def summarize_results(results, dataset, outlier_percentile=0.05):
    X_train, _, _, _, _ = get_data_model(dataset)
    outlier_detection = joblib.load(f"../models/{dataset}/IsolationForest_test.pkl")
    outlier_detection.percentile = outlier_percentile
    perc_calc = PercentileCalculator(X=X_train.astype(np.float64))
    # verify if "individual" and "solutions" are strings
    if type(results["individual"].iloc[0]) == str:
        results["individual"] = results["individual"].apply(literal_eval)
        results["solutions"] = results["solutions"].apply(literal_eval)
    results_df = []
    costs = []
    n_changes = []
    outliers = []
    outliers_score = []
    for i in range(len(results)):
        individual = results["individual"].iloc[i]
        solutions = results["solutions"].iloc[i]
        if len(individual) == 1:
            individual = individual[0]
        percentile_criteria = PercentileCriterion(individual, perc_calc)

        if len(solutions) == 0:
            results_df.append(
                {
                    "costs": None,
                    "n_changes": None,
                    "diversity": None,
                    "outlier": None,
                    "outliers_score": None,
                    "n_solutions": 0,
                    "time": results["time"].iloc[i],
                }
            )
            continue

        # print([percentile_criteria.f(s) for s in solutions])
        costs = np.mean([percentile_criteria.f(s) for s in solutions])
        n_changes = []
        for s in solutions:
            n_changes_ = sum(
                [1 for i in range(len(individual)) if individual[i] != s[i]]
            )
            n_changes.append(n_changes_)
        n_changes = np.mean(n_changes)
        outliers = np.mean(
            [outlier_detection.predict(np.array(s)[None, :]) == -1 for s in solutions]
        )
        outliers_score = np.mean(
            [outlier_detection.score(np.array(s)[None, :]) for s in solutions]
        )
        if len(solutions) == 1:
            diversity = 0
        else:
            diversity = diversity_metric(np.array(solutions))

        results_df.append(
            {
                "costs": costs,
                "n_changes": n_changes,
                "outlier": outliers,
                "outliers_score": outliers_score,
                "diversity": diversity,
                "n_solutions": len(solutions),
                "time": results["time"].iloc[i],
            }
        )

    return pd.DataFrame(results_df)


def format_df_table(df, agg_column, columns):
    df_mean = (
        df.groupby(agg_column)
        .agg(dict([(c, "mean") for c in columns]))
        .reset_index()
        .round(3)
    )
    df_std = (
        df.groupby(agg_column)
        .agg(dict([(c, "std") for c in columns]))
        .reset_index()
        .round(3)
    )
    df_90p = (
        df.groupby(agg_column)
        .agg(dict([(c, lambda x: np.nanpercentile(x, 95)) for c in columns]))
        .reset_index()
        .round(3)
    )

    for col in columns:
        df_mean[col] = (
            df_mean[col].astype("str")
            + " (+-"
            + df_std[col].astype("str")
            + ") | "
            + df_90p[col].astype("str")
        )
    return df_mean
