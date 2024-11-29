# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""

import numpy as np
import functools
import xxhash


def np_cache_xxhash(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

    """

    def decorator(function):
        np_array_aux = None
        method_class = None

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal np_array_aux
            nonlocal method_class

            if len(args) == 1:
                np_array = args[0]
                args = ()
            elif len(args) == 2:
                np_array = args[1]
                method_class = args[0]
                args = ()

            np_array_aux = np_array
            hashable_array = xxhash.xxh32(np_array).hexdigest()
            # if method_class is not None:
            #    hashable_array += xxhash.xxh32(method_class.pivot).hexdigest()
            return cached_wrapper(hashable_array, *args, **kwargs)

        @functools.lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            nonlocal np_array_aux
            nonlocal method_class
            result = function(method_class, np_array_aux, *args, **kwargs)
            return result

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


class PercentileCalculator:
    """Class that is capable of calculating the percentile cost
    (change of the feature in percentiles).

    Parameters
    ----------

    X : numpy array,
        Contains the samples.
    action_set : ActionSet type class,
        Contains the discretization of the features to find the recourses.

    """

    def __init__(self, X=None, action_set=None, method="ustun"):
        self.method = method
        if self.method == "ustun":
            assert (
                X is not None or action_set is not None
            ), "X or action_set should not be None."
            if action_set is not None:
                self.action_set = action_set
            elif X is not None:
                from cfmining.action_set import ActionSet

                self.action_set = ActionSet(X=X)

            self.names = self.action_set.df["name"].values
        self.percentile_vec = np.vectorize(self.percentile)

    # @functools.lru_cache(maxsize=20*1024)
    @functools.lru_cache(maxsize=None)
    def percentile(self, value, feature):
        """Calculates the percentile."""
        if self.method == "ustun":
            return self.action_set[feature].percentile(value)

    def percVec(self, vec):
        # return self.percentile_vec(vec, self.names)
        # return np.array([self.percentile(value, feature) for feature, value in zip(self.names, vec)])
        return np.fromiter(map(self.percentile, vec, self.names), dtype="float")


class PercentileCriterion:
    """Class that using the percentile calculator is capable of defining
    which solution have a higher percentile cost.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    perc_calc : PercentileCalculator class,
        Percentile calculator for that set of samples.
    """

    def __init__(self, pivot, perc_calc):
        self._pivot = np.ascontiguousarray(pivot)
        self.perc_calc = perc_calc
        self.pivotP = self.perc_calc.percVec(pivot)
        # self.f.cache_clear()

    @property
    def pivot(self):
        return self._pivot
    
    @pivot.setter
    def pivot(self, value):
        self._pivot = np.ascontiguousarray(value)
        self.pivotP = self.perc_calc.percVec(self._pivot)

    # @np_cache_xxhash(maxsize=None)
    def f(self, solution):
        solutionP = self.perc_calc.percVec(solution)
        # return abs(solutionP-self.pivotP).max()
        return max(map(abs, solutionP - self.pivotP))

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_obj = self.f(new_sol)
        old_obj = self.f(old_sol)

        return new_obj >= old_obj


class PercentileChangesCriterion(PercentileCriterion):
    """Class that using the percentile calculator and number of changes
    is capable of defining if a solution is worse in both criteria.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    perc_calc : PercentileCalculator class,
        Percentile calculator for that set of samples.
    """

    # @np_cache_xxhash(maxsize=None)
    def f(self, solution):
        perc = super(PercentileChangesCriterion, self).f(solution)
        changes = sum(solution != self.pivot)

        return np.array([perc, changes])

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_objs = self.f(new_sol)
        old_objs = self.f(old_sol)

        return all(new_objs >= old_objs)


class NonDomCriterion:
    """Class that using the changes on each feature
    is capable of defining if a solution had to change more in all features.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    """

    def __init__(self, pivot):
        self.pivot = np.ascontiguousarray(pivot)

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        return all(np.abs(new_sol - self.pivot) >= np.abs(old_sol - self.pivot))


class Criterion:
    def __init__(self, pivot):
        self.pivot = np.ascontiguousarray(pivot)

    def f(self, solution):
        pass

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_obj = self.f(new_sol)
        old_obj = self.f(old_sol)

        return new_obj >= old_obj


class RangeCalculator:
    """Class that calculates the range of the features.

    Parameters
    ----------
    action_set : ActionSet,
        ActionSet with search grid.
    """

    def __init__(self, action_set):
        self.range = np.array([max(feat.grid) - min(feat.grid) for feat in action_set])


class MaxDistCriterion(Criterion):
    """Class that calculates the maximum distance between the pivot and the solution.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    range_calc : RangeCalculator,
        Range calculator for that set of samples.
    categoric_features : list,
        List of boolean values that indicates if a feature is categoric.
    """

    def __init__(self, pivot, range_calc, categoric_features=None):
        self.pivot = np.ascontiguousarray(pivot)
        self.range = range_calc.range
        if categoric_features is None:
            categoric_features = [False] * len(pivot)
        self.categoric_features = categoric_features

    def f(self, solution):
        dif = np.where(self.categoric_features, 0, (solution - self.pivot) / self.range)
        return np.linalg.norm(dif, ord=np.inf)


class CategoricDistCriterion(Criterion):
    """Class that calculates the distance between the pivot and the solution for categoric features.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    categoric_features : list,
        List of boolean values that indicates if a feature is categoric.
    """

    def __init__(self, pivot, categoric_features=None):
        self.pivot = np.ascontiguousarray(pivot)
        if categoric_features is None:
            categoric_features = [False] * len(pivot)
        self.categoric_features = categoric_features

    def f(self, solution):
        dif = np.where(self.categoric_features, solution != self.pivot, 0)
        return np.sum(dif)


class NumberChangesCriterion(Criterion):
    """Class that calculates the number of changes between the pivot and the solution.

    Parameters
    ----------

    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    """

    def __init__(self, pivot):
        self.pivot = np.ascontiguousarray(pivot)

    def f(self, solution):
        return np.sum(solution != self.pivot)


class LpDistCriterion(Criterion):
    def __init__(self, pivot, range_calc, categoric_features=None, p=2):
        self.pivot = np.ascontiguousarray(pivot)
        self.p = p
        self.range = range_calc.range
        if categoric_features is None:
            categoric_features = [False] * len(pivot)
        self.categoric_features = categoric_features

    def f(self, solution):
        dif = np.where(self.categoric_features, 0, (solution - self.pivot) / self.range)
        return np.sum(np.abs(dif))
        #return np.linalg.norm(dif, ord=self.p)

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_obj = self.f(new_sol)
        old_obj = self.f(old_sol)

        return new_obj >= old_obj


class MultiCriterion(Criterion):
    """Class that support multiple criterion in a generic way.

        Parameters
        ----------
    criterion_list : List of Criterion
        List of criterion to be evaluated.
    pivot : numpy array
        Pivot solution to be compared.
    """

    def __init__(self, criterion_list, pivot):
        self.criterion_list = criterion_list
        self.pivot = pivot
        for criterion in self.criterion_list:
            criterion.pivot = pivot

    def f(self, solution):
        return np.array([criterion.f(solution) for criterion in self.criterion_list])

    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_objs = self.f(new_sol)
        old_objs = self.f(old_sol)

        return all(new_objs >= old_objs)


