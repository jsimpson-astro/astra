from typing import Callable, TypeAlias
import numpy as np

class BasePrior:
    def __init__(self):
        raise NotImplementedError()
        
    def eval(self, p):
        raise NotImplementedError()

class UniformPrior(BasePrior):
    """
    Uniform prior for SpectrumFitter.
    Requires two parameters: a lower and an upper bound.
    """
    def __init__(self, lower, upper):
        if lower > upper:
            self._lower = upper
            self._upper = lower
        else:
            self._lower = lower
            self._upper = upper

    def eval(self, p):
        """
        Evaluate prior probability
        """
        if self.lower <= p <= self.upper:
            return 1.
        else:
            return -np.inf

    def __repr__(self):
        return f"UniformPrior({self.lower}, {self.upper})"

    def __str__(self):
        return f"UniformPrior(lower={self.lower}, upper={self.upper})"

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

class GaussianPrior(BasePrior):
    """
    Gaussian prior for SpectrumFitter.
    Requires two parameters: the mean and standard deviation of the Gaussian.
    """
    def __init__(self, mean, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive and non-zero.")

        self._mean = mean
        self._sigma = sigma

    def eval(self, p):
        """
        Evaluate prior probability
        """
        prob = np.exp(-(p - self.mean)**2 / (2 * self.sigma**2)) 
        prob = prob / (2 * np.pi * self.sigma**2)**0.5

        return prob

    def __repr__(self):
        return f"GaussianPrior({self.mean}, {self.sigma})"

    def __str__(self):
        return f"GaussianPrior(mean={self.mean}, sigma={self.sigma})"

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

# typing for prior classes
_AnyPrior: TypeAlias = UniformPrior | GaussianPrior
_PriorOrPriors: TypeAlias = _AnyPrior | list[_AnyPrior]

