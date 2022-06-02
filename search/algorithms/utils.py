import time
import logging
import warnings
import numpy as np
import os.path as osp

import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names

from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.factory import get_performance_indicator
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization, associate_to_niches, calc_niche_count, niching
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance


def setup_logger(logpth):
    logfile = 'search-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    _format = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=_format, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def calc_hv(ref_pt, F, normalized=True):
    # calculate hypervolume on the non-dominated set of F
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_F = F[front, :]
    ref_point = 1.01 * ref_pt
    hv = get_performance_indicator("hv", ref_point=ref_point).do(nd_F)

    if normalized:
        hv = hv / np.abs(np.prod(ref_point))

    return hv


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            try:
                X[i, np.random.choice(is_false)] = True
                X[i, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def distribution_estimation(data):
    """ estimate the optimal distribution from data """
    # Get histogram of original data
    y, x = np.histogram(data, bins='auto', density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # place holder for best distribution
    distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if d not in ['levy_stable', 'studentized_range']]):
        print("{:>3} / {:<3}: {}".format(ii + 1, len(_distn_names), distribution))

        distribution = getattr(st, distribution)

        # Try to fit the distribution and ignore the failed ones
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                distributions.append((distribution, params, sse))

        except Exception:
            pass

    return sorted(distributions, key=lambda q: q[2])[0]  # return the best distribution


def rank_n_crowding_survival(X, F, n_survive):
    # NSGA-II environmental selection operator
    # modified from https://github.com/anyoptimization/pymoo/blob/c6426a721d95c932ae6dbb610e09b6c1b0e13594/
    # pymoo/algorithms/moo/nsga2.py#L71

    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    return X[survivors]


def reference_direction_survival(ref_dirs, X, F, n_survive):
    # NSGA-III environmental selection operator
    # modified from https://github.com/anyoptimization/pymoo/blob/a5e55c790c907d7dc23ec6105e1f3a506ec9a14e/
    # pymoo/algorithms/moo/nsga3.py#L123

    norm = HyperplaneNormalization(ref_dirs.shape[1])

    # calculate the fronts of the population
    fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
    non_dominated, last_front = fronts[0], fronts[-1]

    # update the hyperplane based boundary estimation
    hyp_norm = norm
    hyp_norm.update(F, nds=non_dominated)
    ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

    # consider only the population until we come to the splitting front
    I = np.concatenate(fronts)
    X, rank, F = X[I], rank[I], F[I]

    # update the front indices for the current population
    counter = 0
    for i in range(len(fronts)):
        for j in range(len(fronts[i])):
            fronts[i][j] = counter
            counter += 1
    last_front = fronts[-1]

    # associate individuals to niches
    niche_of_individuals, dist_to_niche, dist_matrix = \
        associate_to_niches(F, ref_dirs, ideal, nadir)

    # if we need to select individuals to survive
    if len(X) > n_survive:

        # if there is only one front
        if len(fronts) == 1:
            n_remaining = n_survive
            until_last_front = np.array([], dtype=int)
            niche_count = np.zeros(len(ref_dirs), dtype=int)

        # if some individuals already survived
        else:
            until_last_front = np.concatenate(fronts[:-1])
            niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[until_last_front])
            n_remaining = n_survive - len(until_last_front)

        S = niching(X[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                    dist_to_niche[last_front])

        survivors = np.concatenate((until_last_front, last_front[S].tolist()))
        X = X[survivors]

    return X