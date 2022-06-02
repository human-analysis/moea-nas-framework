import sys
sys.path.insert(0, './')

import os
import time
import json
import logging
import numpy as np

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation, get_reference_directions

from search.surrogate_models import SurrogateModel
from search.surrogate_models.utils import get_correlation
from search.evaluators.ofa_evaluator import OFAEvaluator
from search.search_spaces.ofa_search_space import SearchSpace
from search.algorithms.evo_nas import EvoNAS
from search.algorithms.utils import MySampling, BinaryCrossover, MyMutation, setup_logger, HighTradeoffPoints

__all__ = ['MSuNAS']


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: OFAEvaluator,
                 err_predictor: SurrogateModel,
                 objs='acc&flops',  # objectives to be optimized,
                 ):
        super().__init__(
            n_var=search_space.n_var, n_obj=len(objs.split('&')),
            n_constr=0, xl=search_space.lb, xu=search_space.ub, type_var=np.int)

        self.search_space = search_space
        self.evaluator = evaluator
        self.err_predictor = err_predictor
        self.objs = objs

    def _evaluate(self, x, out, *args, **kwargs):
        # use surrogate model to predict error
        features = self.search_space.features(x)
        errors = SurrogateModel.predict(self.err_predictor, features).reshape(-1, 1)

        if self.n_obj > 1:
            # exactly measure other objectives
            archs = self.search_space.decode(x)
            other_objs_stats = self.evaluator.evaluate(archs, objs=self.objs.replace('acc', ''), print_progress=False)
            other_objs = np.array([list(stats.values()) for stats in other_objs_stats])
            out["F"] = np.column_stack((errors, other_objs))
        else:
            out["F"] = errors


class SubsetSelectionProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=np.bool)

        #todo: make sure inputs "candidates" and "archive" are [N, M] matrix, where N is pop_size, M is n_obj
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # s, p = stats.kstest(np.concatenate((self.archive, self.candidates[_x])), 'uniform')
            # append selected candidates to archive then sort
            tmps = []
            for j in range(self.archive.shape[1]):
                tmp = np.sort(np.concatenate((self.archive[:, j], self.candidates[_x, j])))
                tmps.append(np.std(np.diff(tmp)))
            f[i, 0] = np.max(tmps)

            # f[i, 0] = s
            # we penalize if the number of selected candidates is not exactly K
            # g[i, 0] = (self.n_max - np.sum(_x)) ** 2
            g[i, 0] = np.sum(_x) - self.n_max  # as long as the selected individual is less than K

        out["F"] = f
        out["G"] = g


class MSuNAS(EvoNAS):
    """
    NSGANetV2: https://arxiv.org/abs/2007.10396
    """
    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: OFAEvaluator,
                 objs='acc&flops',
                 surrogate='lgb',  # surrogate model method
                 n_doe=100,  # design of experiment points, i.e., number of initial (usually randomly sampled) points
                 n_gen=8,  # number of high-fidelity evaluations per generation/iteration
                 max_gens=30,  # maximum number of generations/iterations to search
                 save_path='.tmp',   # path to the folder for saving stats
                 num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
                 resume=None,  # path to a search experiment folder to resume search
                 ):

        super().__init__(search_space, evaluator, objs, pop_size=n_doe, max_gens=max_gens)

        self.surrogate = surrogate
        self.n_doe = n_doe
        self.n_gen = n_gen
        self.num_subnets_to_report = num_subnets
        self.resume = resume
        self.ref_pt = None
        self.save_path = save_path
        self.logger = None  # a placeholder

    def _sample_initial_population(self, sample_size):
        """ a generic initialization method by uniform sampling from search space,
                note that uniform sampling from search space (x or genotype space)
                does NOT imply uniformity in architecture (pheotype) space """

        archs = self.search_space.sample(sample_size - 2)
        # add the lower and upper bound architectures for improving diversity among individuals
        archs.extend(self.search_space.decode(
            [np.array(self.search_space.lb), np.array(self.search_space.ub)]))

        return archs

    def _fit_predictors(self, archive):
        self.logger.info("fitting {} model for accuracy/error ...".format(self.surrogate))

        data = self.get_attributes(archive, _attr='arch&err')
        features = self.search_space.features(self.search_space.encode([d[0] for d in data]))
        err_targets = np.array([d[1] for d in data])
        err_predictor = SurrogateModel(self.surrogate).fit(features, err_targets, ensemble=True)

        return err_predictor

    @staticmethod
    def select_solver(n_obj,
                      _pop_size=100,
                      _crx_prob=0.9,  # crossover probability
                      _mut_eta=1.0,  # polynomial mutation hyperparameter eta
                      _seed=42  # random seed for riesz energy
                      ):
        # define operators
        sampling = get_sampling('int_lhs')
        crossover = get_crossover('int_two_point', prob=_crx_prob)
        mutation = get_mutation('int_pm', eta=_mut_eta)

        if n_obj < 2:
            # use ga, de, pso, etc.
            ea_method = get_algorithm(
                "ga", pop_size=_pop_size, sampling=sampling, crossover=crossover,
                mutation=mutation, eliminate_duplicates=True)

        elif n_obj > 2:
            # use NSGA-III
            # create the reference directions to be used for the optimization
            # # use this if you are familiar with many-obj optimization
            # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
            ref_dirs = get_reference_directions("energy", n_obj, _pop_size, seed=_seed)
            ea_method = get_algorithm(
                'nsga3', pop_size=_pop_size, ref_dirs=ref_dirs, sampling=sampling, crossover=crossover,
                mutation=mutation, eliminate_duplicates=True)

        else:
            # use NSGA-II, MOEA/D
            ea_method = get_algorithm(
                "nsga2", pop_size=_pop_size, sampling=sampling, crossover=crossover,
                mutation=mutation, eliminate_duplicates=True)

        return ea_method

    def subset_selection(self, pop, archive, K):
        # get non-dominated archs from archive
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)

        # select based on the cheap objectives, i.e., Params, FLOPs, etc.
        subset_problem = SubsetSelectionProblem(pop.get("F")[:, 1:], F[front, 1:], K)

        # define a solver
        ea_method = get_algorithm(
            'ga', pop_size=500, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        # start solving
        res = minimize(subset_problem, ea_method, termination=('n_gen', 200), verbose=True)  # set verbos=False to save screen
        # in case the number of solutions selected is less than K
        if np.sum(res.X) < K:
            for idx in np.argsort(pop.get("F")[:, 0]):
                res.X[idx] = True
                if np.sum(res.X) >= K:
                    break
        return res.X

    def _next(self, archive, err_predictor):
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self.search_space, self.evaluator, err_predictor, self.objs)

        # this problem is a regular discrete-variable single-/multi-/many-objective problem
        # which can be exhaustively searched by regular EMO algorithms such as rGA, NSGA-II/-III, MOEA/D, etc.
        ea_method = self.select_solver(problem.n_obj)

        res = minimize(problem, ea_method, termination=('n_gen', 20), verbose=True)  # set verbose=False to save screen

        # check against archive to eliminate any already evaluated subnets to be re-evaluated
        not_duplicate = np.logical_not(
            [x in [x[0] for x in self.get_attributes(archive, _attr='arch')]
             for x in self.search_space.decode(res.pop.get("X"))])

        # form a subset selection problem to short list K from pop_size
        indices = self.subset_selection(res.pop[not_duplicate], archive, self.n_gen)

        candidates = self.search_space.decode(res.pop[not_duplicate][indices].get("X"))

        return candidates

    def search(self):
        # ----------------------- setup ----------------------- #
        # create the save dir and setup logger
        self.save_path = os.path.join(
            self.save_path, self.search_space.name + "NSGANetV2-{}-{}-n_doe@{}-n_gen@{}-max_gens@{}".format(
                self.objs, self.surrogate, self.n_doe, self.n_gen, self.max_gens))

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

        # ----------------------- initialization ----------------------- #
        it_start = time.time()  # time counter
        if self.resume:
            archive = json.load(open(self.resume, 'r'))
        else:
            archive = self.initialization()

        # setup reference point for calculating hypervolume
        if self.ref_pt is None:
            self.ref_pt = np.max(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')), axis=0)

        self.logger.info("Iter 0: hv = {:.4f}, time elapsed = {:.2f} mins".format(
            self._calc_hv(archive, self.ref_pt), (time.time() - it_start) / 60))

        self.save_iteration("iter_0", archive)  # dump the initial population

        # ----------------------- main search loop ----------------------- #
        for it in range(1, self.max_gens + 1):
            it_start = time.time()  # time counter

            # construct err surrogate model from archive
            err_predictor = self._fit_predictors(archive)

            # construct an auxiliary problem of surrogate objectives and
            # search for the next set of candidates for high-fidelity evaluation
            candidates = self._next(archive, err_predictor)

            # high-fidelity evaluate the selected candidates (lower level)
            stats_dict = self._eval(candidates)

            # evaluate the performance of mIoU predictor
            err_pred = SurrogateModel.predict(
                err_predictor, self.search_space.features(self.search_space.encode(candidates)))
            err_rmse, err_r, err_rho, err_tau = get_correlation(
                err_pred, [100 - stat['acc'] for stat in stats_dict])

            # add the evaluated subnets to archive
            for cand, stats in zip(candidates, stats_dict):
                archive.append({'arch': cand, **stats})

            # print iteration-wise statistics
            hv = self._calc_hv(archive, self.ref_pt)
            self.logger.info("Iter {}: hv = {:.4f}, time elapsed = {:.2f} mins".format(
                it, hv, (time.time() - it_start) / 60))
            self.logger.info("Surrogate model {} performance:".format(self.surrogate))
            self.logger.info("For predicting mIoU: RMSE = {:.4f}, Spearman's Rho = {:.4f}, "
                             "Kendall’s Tau = {:.4f}".format(err_rmse, err_rho, err_tau))

            meta_stats = {'iteration': it, 'hv': hv, 'acc_tau': err_tau}
            self.save_iteration("iter_{}".format(it), archive, meta_stats)  # dump the current iteration

        # ----------------------- report search result ----------------------- #
        # dump non-dominated architectures from the archive first
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))
        nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_archive = [archive[idx] for idx in nd_front]
        self.save_subnets("non_dominated_subnets", nd_archive)

        # select a subset from non-dominated set in case further fine-tuning
        nd_F = np.array(self.get_attributes(nd_archive, _attr=self.objs.replace('acc', 'err')))
        selected = HighTradeoffPoints(n_survive=self.num_subnets_to_report).do(nd_F)
        self.save_subnets("high_tradeoff_subnets", [nd_archive[i] for i in selected])

    def save_iteration(self, _save_dir, archive, meta_stats=None):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        json.dump(archive, open(os.path.join(save_dir, 'archive.json'), 'w'), indent=4)
        if meta_stats:
            json.dump(meta_stats, open(os.path.join(save_dir, 'stats.json'), 'w'), indent=4)

    def save_subnets(self, _save_dir, archive):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for i, subnet in enumerate(archive):
            json.dump(subnet, open(os.path.join(save_dir, 'subnet_{}.json'.format(i + 1)), 'w'), indent=4)


if __name__ == '__main__':

    import torch
    from supernets.ofa_mbnv3 import GenOFAMobileNetV3
    from search.evaluators.ofa_evaluator import ImageNetEvaluator
    from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace

    search_space = OFAMobileNetV3SearchSpace()

    supernet = GenOFAMobileNetV3(
        n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # load checkpoints weights
    state_dicts = [
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0',
                   map_location='cpu')['state_dict'],
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2',
                   map_location='cpu')['state_dict']]

    supernet.load_state_dict(state_dicts)

    evaluator = ImageNetEvaluator(
        supernet=supernet, data_root='/home/cseadmin/datasets/ILSVRC2012/images', batchsize=200, n_workers=12)

    nas_method = MSuNAS(search_space, evaluator, objs='acc&flops&params', save_path='tmp',
                        # resume='/home/cseadmin/zhichao/neural-architecture-transfer/tmp/'
                        #        'MobileNetV3SearchSpaceNSGANetV2-acc&flops-lgb-n_doe@100-n_iter@8-max_iter@30/'
                        #        'iter_0/archive.json'
                        )
    nas_method.search()

