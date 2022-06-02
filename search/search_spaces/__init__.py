import numpy as np
from abc import ABC, abstractmethod


class SearchSpace(ABC):
    def __init__(self, **kwargs):
        self.feat_enc = None
        # attributes below have to be filled by child class
        self.n_var = None
        self.lb = None
        self.ub = None
        self.categories = None

    @property
    def name(self):
        return NotImplementedError

    @abstractmethod
    def _sample(self, **kwargs):
        """ method to randomly create a solution """
        raise NotImplementedError

    def sample(self, n_samples, **kwargs):
        subnets = []
        for _ in range(n_samples):
            subnets.append(self._sample(**kwargs))
        return subnets

    @abstractmethod
    def _encode(self, subnet):
        """ method to convert architectural string to search decision variable vector """
        raise NotImplementedError

    def encode(self, subnets):
        X = []
        for subnet in subnets:
            X.append(self._encode(subnet))
        return np.array(X)

    @abstractmethod
    def _decode(self, x):
        """ method to convert decision variable vector to architectural string """
        raise NotImplementedError

    def decode(self, X):
        subnets = []
        for x in X:
            subnets.append(self._decode(x))
        return subnets

    @abstractmethod
    def _features(self, X):
        """ method to convert decision variable vector to feature vector for surrogate model / predictor """
        raise NotImplementedError

    def features(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim < 2:
            X = X.reshape(1, -1)
        return self._features(X)