import numpy as np
import pandas as pd
from ast import literal_eval
import joblib
import sys

sys.path.append("../")
from cfmining.criteria import PercentileCalculator, PercentileCriterion, NonDomCriterion
from cfmining.utils import get_data_model, diversity_metric


def run_experiments(
        method,
        individuals, 
        model, 
        output_file = None,
    ):
    results = []

    if not output_file is None:
        folder = "/".join(output_file.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok = True)

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
        
        results.append({
            "individual" : individual.values.tolist(),
            "prob" : model.predict_proba(individual.values),
            "time" : end - start,
            "n_solutions" : len(method.solutions),
            "solutions" : solutions,
        })

        #print(f"Prob. max counter: {method.prob_max_counter} | Prob: {results[-1]['prob']:.2f}")
        if output_file is not None:
            pd.DataFrame(results).to_csv(output_file, index=False)

        

    results = pd.DataFrame(results)
    if output_file is not None:
        results.to_csv(output_file, index=False)
    return results


def summarize_results(results, dataset, outlier_percentile=0.05):
    X_train, _, _, _ = get_data_model(dataset)
    outlier_detection = joblib.load(f"../models/{dataset}/IsolationForest_test.pkl")
    outlier_detection.percentile = outlier_percentile
    perc_calc = PercentileCalculator(X=X_train)
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
                    "diversity" : None,
                    "outlier": None,
                    "outliers_score": None,
                }
            )
            continue

        costs = np.mean([percentile_criteria.f(s) for s in solutions])
        n_changes = []
        for s in solutions:
            n_changes_ = sum(
                [1 for i in range(len(individual)) if individual[i] != s[i]]
            )
            n_changes.append(n_changes_)
        n_changes = np.mean(n_changes)
        outliers = np.sum(
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
                "diversity" : diversity
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

    for col in columns:
        df_mean[col] = (
            df_mean[col].astype("str") + " (+-" + df_std[col].astype("str") + ")"
        )
    return df_mean
