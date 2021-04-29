"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import numpy as np

import hinc
import thinkplot
import thinkstats2


def InterpolateSample(df, log_upper=6.0):
    """Makes a sample of log10 household income.

    Assumes that log10 income is uniform in each range.

    df: DataFrame with columns income and freq
    log_upper: log10 of the assumed upper bound for the highest range

    returns: NumPy array of log10 household income
    """
    # compute the log10 of the upper bound for each range
    df['log_upper'] = np.log10(df.income)

    # get the lower bounds by shifting the upper bound and filling in
    # the first element
    df['log_lower'] = df.log_upper.shift(1)
    df.log_lower[0] = 3.0

    # plug in a value for the unknown upper bound of the highest range
    df.log_upper[41] = log_upper

    # use the freq column to generate the right number of values in
    # each range
    arrays = []
    for _, row in df.iterrows():
        vals = np.linspace(row.log_lower, row.log_upper, int(row.freq))
        arrays.append(vals)

    # collect the arrays into a single sample
    log_sample = np.concatenate(arrays)
    return log_sample


def Median(xs):
    cdf = thinkstats2.MakeCdfFromList(xs)
    return cdf.Value(0.5)


def PearsonMedianSkewness(xs):
    pass
    # median = Median(xs)
    # mean =


def main():
    df = hinc.ReadData()
    log_sample = InterpolateSample(df, log_upper=6.0)

    log_cdf = thinkstats2.Cdf(log_sample)

    thinkplot.Cdf(log_cdf)
    thinkplot.Show(xlabel='household income',
                   ylabel='CDF')

    cdf = thinkstats2.MakeCdfFromList(log_cdf)

    tsmedian = thinkstats2.Median(log_sample)
    print("Thinkstats median:", tsmedian)
    print("Thinkstats median converted back to dollars:", "${:,.2f}".format(10 ** tsmedian))

    tsmean = thinkstats2.Mean(log_sample)
    print("Thinkstats mean:", tsmean)

    print("Thinkstats mean converted back to dollars:", "${:,.2f}".format(10 ** tsmean))

    tsskewness = thinkstats2.Skewness(log_sample)
    print("Thinkstats skewness:", tsskewness)

    tsPearskewness = thinkstats2.PearsonMedianSkewness(log_sample)
    print("Thinkstats Pearson's skewness:", tsPearskewness)

    pdf = thinkstats2.EstimatedPdf(log_sample)
    thinkplot.Pdf(pdf, label='household income')
    thinkplot.Show(xlabel='household income', ylabel='PDF')

    print("The fraction of households below the mean is: approximately 45.06%.  This is calculated by showing the "
          "following:")
    print("The difference between the cdf Value at this percentage and the mean is:", cdf.Value(0.450603472) - tsmean)

    log_sample2 = InterpolateSample(df, log_upper=7.0)
    log_cdf2 = thinkstats2.Cdf(log_sample2)

    thinkplot.Cdf(log_cdf2)
    thinkplot.Show(xlabel='household income',
                   ylabel='CDF')

    tsmean2 = thinkstats2.Mean(log_sample2)
    print("Thinkstats mean:", tsmean2)

    print("Thinkstats mean converted back to dollars:", "${:,.2f}".format(10 ** tsmean2))

    tsskewness2 = thinkstats2.Skewness(log_sample2)

    tsPearskewness2 = thinkstats2.PearsonMedianSkewness(log_sample2)

    print("If we changed the upper bound, to say 7 or $10 million, the difference in mean in dollars is:",
          "${:,.2f}".format(10 ** tsmean2 - 10 ** tsmean))

    print("The difference in skewness:", tsskewness2 - tsskewness)

    print("The difference in Pearson's skewness:", tsPearskewness2 - tsPearskewness)


if __name__ == "__main__":
    main()
