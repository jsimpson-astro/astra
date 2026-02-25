from typing import TypeAlias
import numpy as np
from scipy.stats.distributions import truncnorm

class BasePrior:
    def __init__(self):
        raise NotImplementedError()

    def eval(self, p):
        raise NotImplementedError()

    def sample(self, size=1):
        raise NotImplementedError()

    def transform(self, u):
        raise NotImplementedError()

    @property
    def bounded(self):
        raise NotImplementedError()

class UniformPrior(BasePrior):
    """
    Uniform prior for SpectrumFitter.
    Requires two parameters: a lower and an upper bound.

    Set norm=True to normalise the prior to a total p of 1 (must be a bounded prior).
    """

    def __init__(self, lower, upper, norm=False):
        if lower > upper:
            self._lower = upper
            self._upper = lower
        else:
            self._lower = lower
            self._upper = upper

        if self.lower == -np.inf or self.upper == np.inf:
            self._bounded = False
        else:
            self._bounded = True
            self._range = self.upper - self.lower

        if self.bounded is False and norm is True:
            raise ValueError("Cannot normalise an infinite prior.")
        self._norm = norm

    def eval(self, p):
        """
        Evaluate prior probability
        """
        if self.lower <= p <= self.upper:
            return 1. / self._range if self.norm else 1.
        else:
            return -np.inf

    def sample(self, size=1):
        """
        Sample from distribution
        """
        if not self.bounded:
            raise ArithmeticError("Cannot sample from infinite, unbounded uniform distribution")

        samples = np.random.uniform(size=size, low=self.lower, high=self.upper)

        return samples

    def transform(self, u):
        """
        Transform from u in the interval [0, 1) to bounds
        """
        if self.bounded is False:
            raise ArithmeticError("Cannot transform with an infinite prior.")

        x = (self._range) * u + self.lower

        return x

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

    @property
    def norm(self):
        return self._norm

    @property
    def bounded(self):
        return self._bounded

    @property
    def mean(self):
        if self.bounded:
            return (self.lower + self.upper) / 2
        else:
            raise ValueError("Cannot compute mean of an unbounded prior.")

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

    def sample(self, size=1):
        """
        Sample from distribution
        """
        samples = np.random.normal(size=size, loc=self.mean, scale=self.sigma)

        return samples

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

    @property
    def bounded(self):
        return False


class TruncnormPrior(GaussianPrior):
    """
    Truncated gaussian prior for SpectrumFitter.
    Requires two parameters: the mean and standard deviation of the Gaussian.
    """

    def __init__(self, mean, sigma, trunc=5., trunc_lower=None, trunc_upper=None):
        if sigma <= 0:
            raise ValueError("sigma must be positive and non-zero.")

        if trunc <= 0:
            raise ValueError("trunc must be positive and non-zero.")

        self._mean = mean
        self._sigma = sigma

        if (trunc_lower is None and trunc_upper is None):
            self._trunc = trunc
            trunc_lower = trunc_upper = trunc
        elif trunc_lower == trunc_upper:
            trunc = abs(trunc_upper)
            self._trunc = trunc
        else:
            trunc_lower = trunc if trunc_lower is None else trunc_lower
            trunc_upper = trunc if trunc_lower is None else trunc_upper
            self._trunc = (trunc_lower, trunc_upper)

        self._lower = mean - trunc_lower * sigma
        self._upper = mean + trunc_upper * sigma

        self._trunc_lower = trunc_lower
        self._trunc_upper = trunc_upper

    def eval(self, p):
        """
        Evaluate prior probability
        """
        prob = truncnorm.pdf(
            p, loc=self.mean, scale=self.sigma,
            a=-self.trunc_lower, b=self.trunc_upper
        )

        return prob

    def sample(self, size=1):
        """
        Sample randomly from distribution
        """
        samples = truncnorm.rvs(
            size=size, loc=self.mean, scale=self.sigma,
            a=-self.trunc_lower, b=self.trunc_upper
        )

        return samples

    def transform(self, u):
        """
        Transform from u in the interval [0, 1) to real values
        """
        x = truncnorm.ppf(
            u, loc=self.mean, scale=self.sigma,
            a=-self.trunc_lower, b=self.trunc_upper
        )

        return x

    def __repr__(self):
        if self.trunc_lower == self.trunc_upper:
            repr_ = f"TruncnormPrior({self.mean}, {self.sigma}, {self.trunc})"
        else:
            repr_ = (
                f"TruncnormPrior({self.mean}, {self.sigma}, "
                f"{self.trunc_lower}, {self.trunc_upper})"
            )
        return repr_

    def __str__(self):
        if self.trunc_lower == self.trunc_upper:
            str_ = f"TruncnormPrior(mean={self.mean}, sigma={self.sigma}, trunc={self.trunc})"
        else:
            str_ = (
                f"TruncnormPrior(mean={self.mean}, sigma={self.sigma}, "
                f"trunc_lower={self.trunc_lower}, trunc_upper={self.trunc_upper})"
            )
        return str_

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    @property
    def trunc(self):
        return self._trunc

    @property
    def trunc_lower(self):
        return self._trunc_lower

    @property
    def trunc_upper(self):
        return self._trunc_upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def bounded(self):
        return True if np.isfinite(self.lower) and np.isfinite(self.upper) else False


# typing for prior classes
_AnyPrior: TypeAlias = UniformPrior | GaussianPrior
_PriorOrPriors: TypeAlias = _AnyPrior | list[_AnyPrior]
