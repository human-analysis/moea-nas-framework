import numpy as np

from .utils import reset_seed, get_correlation

_DEBUG = False


class SurrogateModel:
    def __init__(self, method='lgb'):
        self.method = method

    def _fetch(self, **kwargs):
        if self.method == 'rbf':
            """ use with caution, might not work with one-hot encoding """
            from search.surrogate_models.rbf import RBF
            return RBF()

        elif self.method == 'rbfs':
            """ use with caution, might not work with one-hot encoding """
            from search.surrogate_models.rbf import RBFEnsemble
            return RBFEnsemble()

        elif self.method == 'mlp':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor()

        elif self.method == 'e2epp':
            """ this method is quite slow """
            from search.surrogate_models.carts import CART
            return CART(n_tree=1000)

        elif self.method == 'carts':
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor()

        elif self.method == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
            return GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())

        elif self.method == 'svr':
            from sklearn.svm import SVR
            return SVR()

        elif self.method == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge()

        elif self.method == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor()

        elif self.method == 'bayesian':
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge()

        elif self.method == 'lgb':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(objective='huber')

        else:
            raise NotImplementedError

    def _fit(self, inputs, targets, pretrained=None, **kwargs):
        model = self._fetch(**kwargs)
        if pretrained:
            return model.fit(inputs, targets, pretrained)
        else:
            return model.fit(inputs, targets)

    def fit(self, inputs, targets, ensemble=False, pretrained=None):
        if ensemble:
            return self.cross_validation(inputs, targets, pretrained)
        else:
            return self._fit(inputs, targets, pretrained)

    @staticmethod
    def predict(model, inputs):
        if isinstance(model, list):
            preds = 0
            for i, expert in enumerate(model):
                pred = expert.predict(inputs)
                preds += pred

            avg_pred = preds / len(model)
            return avg_pred
        else:
            return model.predict(inputs)

    def cross_validation(self, inputs, targets, pretrained=None, seed=25130907, num_folds=10):
        reset_seed(seed)
        perm = np.random.permutation(len(targets))

        pool, kendall_tau = [], []

        for i, test_split in enumerate(np.array_split(perm, num_folds)):

            train_split = np.setdiff1d(perm, test_split, assume_unique=True)

            if pretrained:
                expert = self._fit(inputs[train_split, :], targets[train_split], pretrained=pretrained[i])
            else:
                expert = self._fit(inputs[train_split, :], targets[train_split])

            pred = self.predict(expert, inputs[test_split, :])
            rmse, r, rho, tau = get_correlation(pred, targets[test_split])

            print("Fold {}: rmse = {:.4f}, pearson = {:.4f}, spearman = {:.4f}, kendall = {:.4f}".format(
                i, rmse, r, rho, tau))

            kendall_tau.append(tau)
            pool.append(expert)

        print("{}-fold KTau performance = {:.4f}({:.4f})".format(
            num_folds, np.mean(kendall_tau), np.std(kendall_tau)))

        return pool