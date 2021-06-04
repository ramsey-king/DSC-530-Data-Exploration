# DSC 530
# Final Project
# Statistical Analysis of Men's NCAA March Madness Teams
# Author:  Ramsey King
# Date:  06/04/21

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import statsmodels.api as sm

# Load the datasets to be used for the project
team_data_df = pd.read_excel("ncaa-team-data.xlsx", index_col=0)
rpi_df = pd.read_excel("RPI_Dataset.xlsx", index_col=0)

# Combine the two data frames into one final data frame for analysis
ncaa_final_df = pd.merge(team_data_df, rpi_df, how='left', left_on=['TeamID', 'year'], right_on=['TeamID', 'Year'])

# Filter the data frame to only the teams that made the NCAA men's tournament that year:
ncaa_final_df = ncaa_final_df[(ncaa_final_df["ncaa_numeric"] >= 1) & (ncaa_final_df["Year"] >= 1985) &
                              (ncaa_final_df["Year"] <= 2016)]

# Delete the respective columns from the final dataframe for analysis:
del ncaa_final_df["W-L"]
del ncaa_final_df["School"]
del ncaa_final_df["school"]
del ncaa_final_df["CommonNameSchool"]

# Now we want to filter to only teams that have made it to the Final Four and beyond during the years 1985 - 2016:
final_four_df = ncaa_final_df[(ncaa_final_df["ncaa_numeric"] >= 32) & (ncaa_final_df["Year"] >= 1985) &
                              (ncaa_final_df["Year"] <= 2016)]

# A data frame of the teams that reached the Sweet Sixteen round
ncaa_final_df_16 = ncaa_final_df[(ncaa_final_df["ncaa_numeric"] == 8) & (ncaa_final_df["Year"] >= 1985) &
                                 (ncaa_final_df["Year"] <= 2016)]

# A data frame of the teams that reached the Elite Eight round
ncaa_final_df_8 = ncaa_final_df[(ncaa_final_df["ncaa_numeric"] == 16) & (ncaa_final_df["Year"] >= 1985) &
                                (ncaa_final_df["Year"] <= 2016)]

# A data frame of the teams that reached the Final Four round
ncaa_final_df_4 = ncaa_final_df[(ncaa_final_df["ncaa_numeric"] == 32) & (ncaa_final_df["Year"] >= 1985) &
                                (ncaa_final_df["Year"] <= 2016)]


# This function makes histograms of the wins, winning percentage, RPI_Rank, Strength of Schedule, and Seed columns
def make_hists():
    final_four_df.hist(column="w")
    final_four_df.hist(column="wl")
    final_four_df.hist(column="RPI_Rank")
    final_four_df.hist(column="sos")
    final_four_df.hist(column="seed")
    plt.show()


# Here, we get the preliminary statistics from our focus variables: mean, mode, standard deviation and kurtosis
def get_stats():
    final_four_df_means = final_four_df[["w", "wl", "RPI_Rank", "sos", "seed"]].mean()
    final_four_df_modes = final_four_df[["w", "wl", "RPI_Rank", "sos", "seed"]].mode()
    final_four_df_stds = final_four_df[["w", "wl", "RPI_Rank", "sos", "seed"]].std()
    final_four_df_kurts = final_four_df[["w", "wl", "RPI_Rank", "sos", "seed"]].kurt()

    print(final_four_df_means)
    print(final_four_df_modes)
    print(final_four_df_stds)
    print(final_four_df_kurts)


# PMF of seeds according to which round in the tournament:
def pmf_seeds():
    seed_prob_16 = ncaa_final_df_16["seed"].value_counts(normalize=True)
    seed_prob_4 = ncaa_final_df_4["seed"].value_counts(normalize=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.bar(seed_prob_16.index, seed_prob_16.values, color='red', alpha=0.2, label="Sweet 16")
    ax1.set_title("PMF based on Round")
    ax1.set_xlim(0, 15)
    ax1.set_ylabel("Probability")
    ax1.set_ylim(top=0.4)
    ax1.legend()

    ax2.bar(seed_prob_4.index, seed_prob_4.values, color='black', alpha=0.2, label="Final 4")
    ax2.set_xlabel("Seed")
    ax2.set_xlim(0, 15)
    ax2.set_ylabel("Probability")
    ax2.set_ylim(top=0.4)
    ax2.legend()

    plt.show()


# CDF of seeds in that make the Final Four:
def cdf_seeds():
    seed_prob_4 = ncaa_final_df_4["seed"].value_counts(normalize=True)

    sorted_final_four = np.sort(seed_prob_4.index)
    cdf4 = np.cumsum(seed_prob_4)
    plt.plot(sorted_final_four, cdf4, label="cdf")
    plt.title("CDF of Final Four Seeds")
    plt.xlabel("Seed")
    plt.ylabel("CDF")
    plt.show()

    # CDF of model Pareto function
    # the next 3 lines allow the upper limit of the Pareto distribution to be capped at 11, to resemble my distribution.
    upper = 11
    pareto = np.random.pareto(1, int(500 * 5 / 4)) + 1
    pareto = pareto[pareto < upper][:500]

    # print(pareto)
    sort_pareto = np.sort(pareto)
    sort_pareto_cdf = np.cumsum(sort_pareto)
    plt.plot(sort_pareto, sort_pareto_cdf, label="Model CDF")
    plt.title("Model Pareto CDF, alpha = 1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# scatter plots of wins vs RPI, and Winning percentage (wl) vs. Strength of Schedule (sos)
def make_scatter():
    x = final_four_df['wl']
    y = final_four_df['RPI']
    plt.scatter(x, y)
    plt.title('Winning Percentage vs RPI')
    plt.xlabel('Winning Percentage')
    plt.ylabel('RPI')
    plt.show()

    a = final_four_df['RPI']
    b = final_four_df['sos']
    plt.scatter(a, b)
    plt.title('RPI vs Strength of Schedule')
    plt.xlabel('RPI')
    plt.ylabel('Strength of Schedule')
    plt.show()

    print("Pearson's correlation between Winning Percentage and RPI is",
          final_four_df['wl'].corr(final_four_df['RPI'], method='pearson'))

    print("Pearson's correlation between RPI and Strength of Schedule is",
          final_four_df['RPI'].corr(final_four_df['sos'], method='pearson'))

    print("The covariance between Winning Percentage and RPI is",
          final_four_df['wl'].cov(final_four_df['RPI']))

    print("The covariance between RPI and Strength of Schedule is",
          final_four_df['RPI'].cov(final_four_df['sos']))


# function to perform the Correlation hypothesis testing between seed and ncaa_numeric to prove/disprove randomness.
def correlation_test():
    x = ncaa_final_df['seed']
    y = ncaa_final_df['ncaa_numeric']
    r, pvalue = sp.pearsonr(x, y)
    print("Pearson R = ", r, "P Value = ", pvalue)

    correlation_list = []
    for iterations in range(1, 1001):
        shuffle_x = np.random.permutation(x)
        r_iteration, pvalue_iteration = sp.pearsonr(shuffle_x, y)
        correlation_list.append(r_iteration)

    print("Max observed Pearson R of shuffled samples:", max(correlation_list))

    plt.scatter(x, y)
    plt.xlabel("Seed")
    plt.ylabel("Ncaa Numeric (Larger means better)")
    plt.title("Seed vs. Tournament Success")
    plt.show()


# function to execute regression analysis between ncaa_numeric and seed, wins, and RPI
def regression_analysis():
    x = ncaa_final_df[['seed', 'w', 'RPI']]
    y = ncaa_final_df['ncaa_numeric']

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    model.predict(x)

    print(model.summary())


if __name__ == '__main__':
    make_hists()
    get_stats()
    pmf_seeds()
    cdf_seeds()
    make_scatter()
    correlation_test()
    regression_analysis()
