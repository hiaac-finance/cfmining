# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""

import numpy as np
import time
from sortedcontainers import SortedDict

from .criteria import *


class MAPOCAM:
    """Class that finds a set of Pareto-optimal counterfactual antecedents
    using branch-and-bound based tree search.

    Parameters
    ----------

    action_set : ActionSet type class,
        Contains the discretization of the features to find the counterfactual antecedents.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find counterfactual antecedents
        that change the outcome.
    classifier : predictor.Classifier type class
        Wrapper to help on finding the counterfactual antecedent.
    max_changes : int, optional (default=3)
        Maximal number of changes of a counterfactual antecedent.
    compare : criteria type class.
        Class that evaluate the cost of a counterfactual antecedent.
    time_limit : float, optional (default=np.inf)
        Maximum time to run the algorithm.

    Attributes
    ----------

    solutions : set of numpy arrays.
        Set of Pareto-optimal counterfactual antecedent.

    Notes
    -----

    See the original paper: [1]_.

    References
    ----------

    .. [1] Raimundo, M.M.; Nonato, L.G.; Poco, J.
    “Mining Pareto-Optimal Counterfactual Antecedents with a Branch-and-Bound Model-Agnostic Algorithm”.
    Submitted to IEEE Transactions on Knowledge and Data Engineering.

    Methods
    -------

    """

    eps = 0.0000001

    def __init__(
        self,
        action_set,
        pivot,
        classifier,
        max_changes=3,
        compare=None,
        clean_suboptimal=True,
        warm_solutions=None,
        recursive=False,
        time_limit=np.inf,
    ):
        self.names = list(action_set.df["name"])
        self.d = len(self.names)
        self.max_changes = max_changes
        self.clf = classifier
        self.action_set = action_set
        self.recursive = recursive
        self.time_limit = time_limit

        assert type(pivot) is np.ndarray, "pivot should be a numpy array"
        self.pivot = pivot

        self.feat_direction = np.array(
            [action_set[feat_name].flip_direction for feat_name in self.names]
        )
        self.mutable_features = [
            i for (i, name) in enumerate(self.names) if self.action_set[name].mutable
        ]


        self.feas_grid = action_set.feasible_grid(
            pivot, return_percentiles=False, return_actions=False, return_immutable=True
        )

        self.sequence = np.argsort(classifier.feat_importance)[::-1]
        # self.sequence = np.argsort(classifier.feat_importance*action_range)[::-1]

        if compare is None:
            self.compare = NonDomCriterion(self.feat_direction)
        else:
            self.compare = compare

        self.clean_suboptimal = clean_suboptimal
        self.solutions = []
        if warm_solutions is not None:
            for old_sol in warm_solutions:
                solution = self.feat_direction * np.maximum(
                    self.feat_direction * old_sol, self.feat_direction * pivot
                )
                if self.clf.predict_proba(solution) >= self.clf.threshold - self.eps:
                    self.update_solutions(solution)

    def update_solutions(self, solution):
        if not self.recursive:
            for key in self.calls:
                if self.compare.greater_than(self.calls[key][0], solution):
                    del self.calls[key]

        if self.clean_suboptimal:
            solutions = []
            new_optimal = True
            self.keep_solutions = np.ones(len(self.solutions))
            for idx, old_sol in enumerate(self.solutions):
                old_better = self.compare.greater_than(solution, old_sol)
                new_better = self.compare.greater_than(old_sol, solution)
                if new_better and not old_better:
                    self.keep_solutions[idx] = 0
                if old_better:
                    new_optimal = False
                    # #print('sda')
                    break
                # new_optimal = new_optimal and not old_better
            self.solutions = [
                old_sol
                for idx, old_sol in enumerate(self.solutions)
                if self.keep_solutions[idx] == 1
            ]
            if new_optimal:
                self.solutions += [solution]
        else:
            self.solutions += [solution]

    def find_candidates(self, solution=None, size=0, changes=0):
        next_idx = self.sequence[size]
        next_name = self.names[next_idx]

        for value in self.feas_grid[next_name]:
            new_size = size + 1
            new_changes = changes + (value != self.pivot[next_idx])

            new_solution = solution.copy()
            new_solution[next_idx] = value

            for old_sol in self.solutions:
                if self.compare.greater_than(new_solution, old_sol):
                    return

            new_proba = self.clf.predict_proba(new_solution)

            if new_proba >= self.clf.threshold - self.eps:
                self.update_solutions(new_solution)
                return

            if new_size >= self.pivot.size:
                continue

            if new_changes >= self.max_changes:
                continue

            if hasattr(self.clf, "predict_max") and self.clf.use_predict_max:
                open_vars = [
                    x for x in self.sequence[new_size:] if x in self.mutable_features
                ]
                fixed_vars = [
                    x for x in self.sequence if x not in open_vars
                ]
                max_prob = self.clf.predict_max(new_solution, fixed_vars)
                if max_prob < self.clf.threshold:
                    continue

            if self.recursive:
                self.find_candidates(new_solution, new_size, new_changes)
            else:
                self.idd += 1
                # self.calls[(self.d-changes+new_proba, self.idd)] = (new_solution, new_size, new_changes)
                self.calls[(new_proba, self.idd)] = [
                    new_solution,
                    new_size,
                    new_changes,
                ]

    def fit(self):
        """
        Find counterfactual antecedent given the data.
        """
        start = time.time()
        if self.recursive:
            self.find_candidates(self.pivot.copy(), 0, 0)
        else:
            self.idd = 0
            self.calls = SortedDict(
                {(self.clf.predict_proba(self.pivot), 0): [self.pivot.copy(), 0, 0]}
            )
            while len(self.calls) > 0 and time.time() - start < self.time_limit:
                # #print([key[0] for key in self.calls])
                _, call = self.calls.popitem()
                self.find_candidates(*call)

        if not self.clean_suboptimal:
            solutions = []
            for i, solution in enumerate(self.solutions):
                optimal = True
                for j, comp_sol in enumerate(self.solutions):
                    if self.compare.greater_than(solution, comp_sol):
                        if i < j or not self.compare.greater_than(comp_sol, solution):
                            optimal = False
                            break
                if optimal:
                    solutions += [solution]
            self.solutions = solutions

        return self


class ActionsEnumerator:
    """Class that finds a set of Pareto-optimal counterfactual antecedents.

    Parameters
    ----------
    action_set : ActionSet type class,
        Contains the discretization of the features to find the counterfactual antecedents.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find counterfactual antecedents
        that change the outcome.
    classifier : predictor.Classifier type class
        Wrapper to help on finding the counterfactual antecedents.
    max_changes : int, optional (default=3)
        Maximal number of changes of a counterfactual antecedent.
    compare : criteria type class.
        Class that evaluate the cost of a counterfactual antecedent.

    Attributes
    ----------
    solutions : set of numpy arrays.
        Set of Pareto-optimal counterfactual antecedents.

    Notes
    -----
    See the original paper: [1]_.

    References
    ----------
    .. [1] Raimundo, M.M.; Nonato, L.G.; Poco, J.
    “Mining Pareto-Optimal Counterfactual Antecedents with a Branch-and-Bound Model-Agnostic Algorithm”.
    Submitted to IEEE Transactions on Knowledge and Data Engineering.
    Methods
    -------
    """

    def __init__(
        self,
        action_set,
        pivot,
        classifier,
        max_changes=3,
        compare=None,
        clean_suboptimal=False,
        warm_solutions=None,
    ):
        self.names = list(action_set.df["name"])
        self.d = len(self.names)
        self.max_changes = max_changes
        self.clf = classifier
        self.action_set = action_set

        assert type(pivot) is np.ndarray, "pivot should be a numpy array"
        self.pivot = pivot

        self.sequence = np.argsort(classifier.feat_importance)[::-1]
        self.feat_direction = np.array(
            [action_set[feat_name].flip_direction for feat_name in self.names]
        )

        action_grid = action_set.feasible_grid(pivot, return_percentiles=False)
        self.max_action = pivot.copy()

        self.feas_grid = {}
        for idx, feat_name in enumerate(self.names):
            flip_dir = self.feat_direction[idx]
            self.feas_grid[feat_name] = (
                action_grid[feat_name][::flip_dir] + self.pivot[idx]
                if feat_name in action_grid and self.action_set[feat_name].mutable
                else np.array([self.pivot[idx]])
            )
            if flip_dir == 1:
                self.max_action[idx] = max(self.feas_grid[feat_name])
            else:
                self.max_action[idx] = min(self.feas_grid[feat_name])

        if compare is None:
            self.compare = NonDomCriterion(self.feat_direction)
        else:
            self.compare = compare

        self.clean_suboptimal = clean_suboptimal
        self.solutions = []
        if warm_solutions is not None:
            for old_sol in warm_solutions:
                solution = self.feat_direction * np.maximum(
                    self.feat_direction * old_sol, self.feat_direction * pivot
                )
                if self.clf.predict(solution):
                    self.update_solutions(solution)

        # self.solutions = []

    def fit(self):
        """
        Find actionable counterfactual antecedents given the data.
        """
        self.recursive_fit()
        solutions = []
        for i, solution in enumerate(self.solutions):
            optimal = True
            for j, comp_sol in enumerate(self.solutions):
                if self.compare.greater_than(solution, comp_sol):
                    if i < j or not self.compare.greater_than(comp_sol, solution):
                        optimal = False
                        break
            if optimal:
                solutions += [solution]
        self.solutions = solutions

        return self

    def update_solutions(self, solution):
        if self.clean_suboptimal:
            solutions = []
            new_optimal = True
            for old_sol in self.solutions:
                old_better = self.compare.greater_than(solution, old_sol)
                new_better = self.compare.greater_than(old_sol, solution)
                if old_better or not new_better:
                    solutions += [old_sol]
                new_optimal = new_optimal and not old_better
            if new_optimal:
                self.solutions = solutions + [solution]
        else:
            self.solutions += [solution]

    def recursive_fit(self, solution=None, size=0, changes=0):
        if solution is None:
            solution = self.pivot.copy()

        for old_sol in self.solutions:
            if self.compare.greater_than(solution, old_sol):
                return True

        ## stop search if constraint is satified
        if self.clf.predict(solution):
            self.update_solutions(solution)
            return True

        if size >= self.pivot.size:
            return False

        ## stop search if achieved maximal changes
        if changes >= self.max_changes:
            return False

        next_idx = self.sequence[size]
        next_name = self.names[next_idx]

        for value in self.feas_grid[next_name]:
            new_solution = solution.copy()
            new_solution[next_idx] = value
            new_changes = changes + (value != self.pivot[next_idx])
            if self.recursive_fit(new_solution, size + 1, new_changes):
                break

        return False


class Greedy:
    """Class that finds a single solution given a criterion.

    Parameters
    ----------
    action_set : ActionSet type class,
        Contains the discretization of the features to find the counterfactual antecedents.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find counterfactual antecedents
        that change the outcome.
    compare : criteria type class.
        Class that evaluate the cost of a counterfactual antecedent.

    Attributes
    ----------
    solution : numpy array.
        A counterfactual antecedent.

    Notes
    -----
    See the original paper: [1]_.

    References
    ----------
    .. [1] Raimundo, M.M.; Nonato, L.G.; Poco, J.
    “Mining Pareto-Optimal Counterfactual Antecedents with a Branch-and-Bound Model-Agnostic Algorithm”.
    Submitted to IEEE Transactions on Knowledge and Data Engineering.
    Methods
    -------
    """

    def __init__(self, action_set, pivot, classifier, compare):
        self.compare = compare
        self.names = list(action_set.df["name"])
        self.clf = classifier
        self.action_set = action_set
        assert type(pivot) is np.ndarray, "pivot should be a numpy array"
        self.pivot = pivot

        self.feat_direction = np.array(
            [action_set[feat_name].flip_direction for feat_name in self.names]
        )

        action_grid = action_set.feasible_grid(pivot, return_percentiles=False)
        self.feas_grid = {}

        for idx, feat_name in enumerate(self.names):
            flip_dir = self.feat_direction[idx]
            self.feas_grid[feat_name] = (
                action_grid[feat_name][::flip_dir] + self.pivot[idx]
                if feat_name in action_grid and self.action_set[feat_name].mutable
                else np.array([self.pivot[idx]])
            )

        self.solutions = []

    def calc_improv(self, action, next_action):
        cost = self.compare.f(action)
        next_cost = self.compare.f(next_action)

        score = self.clf.predict_proba(action)
        next_score = self.clf.predict_proba(next_action)

        if next_score <= score:
            return float("inf")
        elif (next_cost - cost) == 0:
            return 1 / abs(next_score - score)
        else:
            return (next_cost - cost) / abs(next_score - score)

    def fit(self):
        """
        Find counterfactual antecedents given the data.
        """
        solution = self.pivot.copy()
        score = self.clf.predict_proba(solution)

        best_improv = float("inf")
        while score < self.clf.threshold:
            best_solution = solution.copy()
            best_improv = float("inf")

            for idx, feat in enumerate(self.feas_grid):
                if solution[idx] != self.pivot[idx]:
                    start = np.searchsorted(
                        self.action_set[feat].flip_direction * self.feas_grid[feat],
                        self.action_set[feat].flip_direction * solution[idx],
                        side="left",
                    )
                else:
                    start = 0

                for act in self.feas_grid[feat][start + 1 :]:
                    new_solution = solution.copy()
                    new_solution[idx] = act
                    new_improv = self.calc_improv(solution, new_solution)
                    if new_improv < best_improv:
                        best_improv = new_improv
                        best_solution = new_solution
            # #print(best_improv)
            solution = best_solution
            score = self.clf.predict_proba(solution)
            if best_improv == float("inf"):
                break
        if best_improv == float("inf"):
            self.solution = None
        else:
            self.solution = solution


class BruteForce(MAPOCAM):
    """Class that finds a set of Pareto-optimal counterfactual antecedents using a brute force approach.

    Parameters
    ----------
    action_set : ActionSet type class,
        Contains the discretization of the features to find the counterfactual antecedents.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find counterfactual antecedents
        that change the outcome.
    classifier : predictor.Classifier type class
        Wrapper to help on finding the counterfactual antecedents.
    max_changes : int, optional (default=3)
        Maximal number of changes of a counterfactual antecedent.
    compare : criteria type class.
        Class that evaluate the cost of a reccounterfactual antecedentourse.

    Attributes
    ----------
    solutions : set of numpy arrays.
        Set of Pareto-optimal counterfactual antecedents.

    Notes
    -----
    See the original paper: [1]_.

    References
    ----------
    .. [1] Raimundo, M.M.; Nonato, L.G.; Poco, J.
    “Mining Pareto-Optimal Counterfactual Antecedents with a Branch-and-Bound Model-Agnostic Algorithm”.
    Submitted to IEEE Transactions on Knowledge and Data Engineering.
    Methods
    -------
    """

    def fit(self):
        """
        Find counterfactual antecedents given the data.
        """
        self.clean_suboptimal = True
        self.recursive = True
        self.recursive_fit()

        return self

    def recursive_fit(self, solution=None, size=0, changes=0):
        if solution is None:
            solution = self.pivot.copy()
        else:
            if self.clf.predict(solution):
                self.update_solutions(solution)

        if changes >= self.max_changes or size >= self.pivot.size:
            return

        next_idx = self.sequence[size]
        next_name = self.names[next_idx]

        for value in self.feas_grid[next_name]:
            new_solution = solution.copy()
            new_solution[next_idx] = value
            new_changes = changes + (value != self.pivot[next_idx])
            self.recursive_fit(new_solution, size + 1, new_changes)


class MAPOFCEM:
    """Class that finds a set of Pareto-optimal counterfactual antecedents
    using branch-and-bound based tree search.

    Parameters
    ----------

    action_set : ActionSet type class,
        Contains the discretization of the features to find the counterfactual antecedents.
    classifier : predictor.Classifier type class
        Wrapper to help on finding the counterfactual antecedent.
    max_changes : int, optional (default=3)
        Maximal number of changes of a counterfactual antecedent.
    compare : str or criteria type class.
        String with name of a criteria or a class that evaluate the cost of a counterfactual antecedent, if string must be in ['percentile', 'percentile_change', 'non_dom'].

    Attributes
    ----------

    solutions : set of numpy arrays.
        Set of Pareto-optimal counterfactual antecedent.

    """

    eps = 0.0000001

    def __init__(
        self,
        action_set,
        classifier,
        compare="percentile",
        estimate_predict_max=False,
        estimate_outlier=True,
        max_changes=3,
        categorical_features=None,
        outlier_contamination=0.05,
        time_limit=180,
    ):
        self.action_set = action_set
        self.clf = classifier
        self.outlier_contamination = outlier_contamination
        if hasattr(self.clf.outlier_clf, "contamination"):
            self.clf.outlier_clf.contamination = outlier_contamination
        if categorical_features is None:
            categorical_features = []
        self.categorical_features = categorical_features

        self.names = list(action_set.df["name"])
        self.d = len(self.names)
        self.mutable_features = [
            i for (i, name) in enumerate(self.names) if self.action_set[name].mutable
        ]

        if type(compare) is str:
            if compare == "percentile":
                self.perc_calc = PercentileCalculator(action_set=action_set)
                self.compare_call = lambda cfe: PercentileCriterion(cfe, self.perc_calc)
            elif compare == "percentile_change":
                self.perc_calc = PercentileCalculator(action_set=action_set)
                self.compare_call = lambda cfe: PercentileChangesCriterion(
                    cfe, self.perc_calc
                )
            elif compare == "non_dom":
                self.compare_call = lambda cfe: NonDomCriterion(cfe)
            elif compare == "euclidean":
                self.range_calc = RangeCalculator(action_set=action_set)
                self.compare_call = lambda cfe: LpDistCriterion(cfe, self.range_calc)
            elif compare == "abs_diff":
                self.range_calc = RangeCalculator(action_set=action_set)
                self.compare_call = lambda cfe: AbsDiffCriterion(cfe, self.range_calc)
            else:
                raise ValueError("compare must be a valid string")
        else:
            self.compare_call = compare

        self.sequence = np.argsort(classifier.feat_importance)[::-1]
        self.max_changes = max_changes
        if self.max_changes == np.inf:
            self.max_changes = self.d

        # maybe remove later
        self.estimate_predict_max = estimate_predict_max
        self.estimate_outlier = estimate_outlier
        self.time_limit = time_limit

    def update_solutions(self, solution):
        # remove dominated calls
        for key in self.calls:
            if self.compare.greater_than(self.calls[key][0], solution):
                del self.calls[key]

        # verify if the solution is dominated by any other solution
        # or if it dominates any other solution
        new_optimal = True
        self.keep_solutions = np.ones(len(self.solutions))
        for idx, old_sol in enumerate(self.solutions):
            old_better = self.compare.greater_than(solution, old_sol)
            new_better = self.compare.greater_than(old_sol, solution)
            if new_better and not old_better:
                self.keep_solutions[idx] = 0
            if old_better:
                new_optimal = False
                break

        self.solutions = [
            old_sol
            for idx, old_sol in enumerate(self.solutions)
            if self.keep_solutions[idx] == 1
        ]
        if new_optimal:
            self.solutions += [solution]

    def find_candidates(self, solution=None, size=0, changes=0):
        next_idx = self.sequence[size]
        next_name = self.names[next_idx]

        # Looking for a candidate following features
        for value in self.feas_grid[next_name]:
            new_solution = solution.copy()
            new_solution[next_idx] = value

            # If new solution is worse than any previous, stop
            if not next_name in self.categorical_features:
                for old_sol in self.solutions:
                    if self.compare.greater_than(new_solution, old_sol):
                        return

            # If is a solution, save it
            new_proba = self.clf.predict_proba(new_solution)
            if new_proba >= self.clf.threshold - self.eps:
                # Only save if it is not an outlier

                outlier = self.clf.predict_outlier(new_solution) == -1

                if not outlier and not next_name in self.categorical_features:
                    self.update_solutions(new_solution)
                    return
                else:
                    self.outlier_detection_counter += 1
                    # continue

            new_size = size + 1
            if new_size >= self.pivot.size:
                continue

            # If made more changes than the limit
            new_changes = changes + (value != self.pivot[next_idx])
            if new_changes >= self.max_changes:
                continue

            open_vars = [
                x for x in self.sequence[new_size:] if x in self.mutable_features
            ]

            # Calculate max probability of solution
            max_prob = 1
            if hasattr(self.clf, "predict_max"):
                if self.estimate_predict_max:
                    max_prob = self.clf.estimate_predict_max(
                        new_solution, open_vars, self.max_changes - new_changes
                    )
                else:
                    max_prob = self.clf.predict_max(
                        new_solution, open_vars, self.max_changes - new_changes
                    )

            # If max probability will be lower than the threshold, continue
            if max_prob < self.clf.threshold - self.eps:
                self.prob_max_counter += 1
                continue

            # If the partial solution is already an outlier, skip it

            if self.estimate_outlier:
                new_solution_with_nan = new_solution.copy().astype(float)
                new_solution_with_nan[open_vars] = np.nan
                outlier = self.clf.predict_outlier(new_solution_with_nan) == -1
                if outlier:
                    self.outlier_detection_counter += 1
                    continue

            self.idd += 1
            self.calls[(new_proba, self.idd)] = [
                new_solution,
                new_size,
                new_changes,
            ]

    def fit(self, pivot):
        """
        Find counterfactual antecedent given the data.

        Parameters
        ----------
        pivot : numpy type array,
            Sample that had an undesired prediction that we want to find counterfactual antecedents that change the outcome.
        """
        assert type(pivot) is np.ndarray, "pivot should be a numpy array"
        self.pivot = pivot
        if hasattr(self.clf, "set_pivot"):
            self.clf.set_pivot(pivot)
        self.compare = self.compare_call(self.pivot)

        self.feas_grid = self.action_set.feasible_grid(
            pivot, return_percentiles=False, return_actions=False, return_immutable=True
        )

        self.solutions = []
        self.prob_max_counter = 0
        self.outlier_detection_counter = 0

        self.idd = 0
        self.calls = SortedDict(
            {(self.clf.predict_proba(self.pivot), 0): [self.pivot.copy(), 0, 0]}
        )
        start = time.time()
        while len(self.calls) > 0 and time.time() - start < self.time_limit:
            _, call = self.calls.popitem()
            self.find_candidates(*call)

        self.solutions = [s.tolist() for s in self.solutions]
        return self
