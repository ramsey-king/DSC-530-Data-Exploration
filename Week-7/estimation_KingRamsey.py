"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkstats2
import thinkplot

import math
import random
import numpy as np


def MeanError(estimates, actual):
    """Computes the mean error of a sequence of estimates.

    estimate: sequence of numbers
    actual: actual value

    returns: float mean error
    """
    errors = [estimate-actual for estimate in estimates]
    return np.mean(errors)


def RMSE(estimates, actual):
    """Computes the root mean squared error of a sequence of estimates.

    estimate: sequence of numbers
    actual: actual value

    returns: float RMSE
    """
    e2 = [(estimate-actual)**2 for estimate in estimates]
    mse = np.mean(e2)
    return math.sqrt(mse)


def Estimate1(n=7, m=1000):
    """Evaluates RMSE of sample mean and median as estimators.

    n: sample size
    m: number of iterations
    """
    mu = 0
    sigma = 1

    means = []
    medians = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        xbar = np.mean(xs)
        median = np.median(xs)
        means.append(xbar)
        medians.append(median)

    print('Experiment 1')
    print('rmse xbar', RMSE(means, mu))
    print('rmse median', RMSE(medians, mu))

    # additional code added to answer 7-1. This will help to explain if x and median are biased estimates of mu
    print('mean error xbar:', MeanError(means, mu))
    print('mean error median:', MeanError(medians, mu))


def Estimate2(n=7, m=1000):
    """Evaluates S and Sn-1 as estimators of sample variance.

    n: sample size
    m: number of iterations
    """
    mu = 0
    sigma = 1

    estimates1 = []
    estimates2 = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        biased = np.var(xs)
        unbiased = np.var(xs, ddof=1)
        estimates1.append(biased)
        estimates2.append(unbiased)

    print('Experiment 2')
    print('mean error biased', MeanError(estimates1, sigma**2))
    print('mean error unbiased', MeanError(estimates2, sigma**2))

    # additional code added to help explain if s-squared and s-squared n-1 yields a lower MSE.
    print('RMSE estimates1', RMSE(estimates1, sigma))
    print('RMSE estimates2', RMSE(estimates2, sigma))




def Estimate3(n=10, m=1000):
    """Evaluates L and Lm as estimators of the exponential parameter.

    n: sample size
    m: number of iterations
    """
    lam = 2

    means = []
    medians = []
    for _ in range(m):
        xs = np.random.exponential(1/lam, n)
        L = 1 / np.mean(xs)
        Lm = math.log(2) / np.median(xs)
        means.append(L)
        medians.append(Lm)

    print('Experiment 3')
    print('rmse L', RMSE(means, lam))
    print('rmse Lm', RMSE(medians, lam))
    print('mean error L', MeanError(means, lam))
    print('mean error Lm', MeanError(medians, lam))


def SimulateSample(mu=0.5, sigma=0.5, n=10, m=1000):
    """Plots the sampling distribution of the sample mean.

    mu: hypothetical population mean
    sigma: hypothetical population standard deviation
    n: sample size
    m: number of iterations
    """
    def VertLine(x, y=1):
        thinkplot.Plot([x, x], [0, y], color='0.8', linewidth=3)

    means = []
    for _ in range(m):
        xs = np.random.normal(mu, sigma, n)
        xbar = np.mean(xs)
        means.append(xbar)

    stderr = RMSE(means, mu)
    print('standard error', stderr)

    cdf = thinkstats2.Cdf(means)
    ci = cdf.Percentile(5), cdf.Percentile(95)
    print('confidence interval', ci)
    VertLine(ci[0])
    VertLine(ci[1])

    # plot the CDF
    thinkplot.Cdf(cdf)
    thinkplot.Save(root='estimation1',
                   xlabel='sample mean',
                   ylabel='CDF',
                   title='Sampling distribution')


def main():
    thinkstats2.RandomSeed(17)

    print("The Base Case: Sample size 7 and 1,000 iterations")
    Estimate1(7, 1000)
    print()

    print("Sample size 7; 50,000 iterations:")
    Estimate1(7, 50000)
    print()

    print("Sample size 70; 1,000 iterations:")
    Estimate1(70, 1000)
    print()

    print("Sample size of 70; 50,000 iterations")
    Estimate1(70, 50000)
    print()

    print("Default case:  Sample size 7 and 1,000 iterations")
    Estimate2(7, 1000)
    print()

    print("Sample size 7 and 50,000 iterations")
    Estimate2(7, 50000)
    print()

    print("Default case:  Sample size 70 and 1,000 iterations")
    Estimate2(70, 1000)
    print()

    print("Sample size 70 and 50,000 iterations")
    Estimate2(70, 50000)
    print()

    print("Default case:  sample size 10 and 1,000 iterations")
    Estimate3()
    print()

    # This simulates the default case above (n = 10, m = 1,000) for plotting purposes to determine the 90% confidence interval
    SimulateSample()

    print("Sample size is 100 and 1,000 iterations")
    Estimate3(n=100, m=1000)
    print()

    # This simulates the above example for plotting purposes to determine the 90% confidence interval
    SimulateSample(0.5, 0.5, 100, 1000)

    print("Sample size is 1,000 and 10,000 iterations")
    Estimate3(n=1000, m=10000)
    print()

    # This simulates the above example for plotting purposes to determine the 90% confidence interval
    SimulateSample(0.5, 0.5, 1000, 10000)



if __name__ == '__main__':
    main()
