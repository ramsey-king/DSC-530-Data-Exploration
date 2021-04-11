'''
Ramsey King
Assignment 3.2
April 10, 2021
'''

import nsfg
import thinkstats2

'''
The PmfMean function accepts probability mass function as a parameter.  The mean variable is initialized at 0, and is
calculated according to the formula to calculate the mean from the probability mass, which is the summation of the 
discrete value times their probability.  The for loop executes the summation process to calculate the mean.
At the end of the loop, the mean is returned.  The "+1" for all loops in this program are used for indexing purposes.  
'''
def PmfMean(probmassfun):
    mean = 0
    for i in range(0, probmassfun.max() + 1):
        mean += i * probmassfun.value_counts(normalize=True)[i]
    return mean

'''
The PmfVar function accepts two parameters, the probability mass function, and mean.  The mean calculated from the
PmfMean function will be used in this function to prevent repeating code.  The formula for the variance given a 
probability mass function is defined as the summation of the probability of each discrete value multiplied by the
square of the difference of the discrete value and the mean.  The variance variable is initialized at 0 and returned
at the end of the function. 
'''
def PmfVar(probmassfun, mean):
    variance = 0
    for i in range(0, probmassfun.max() + 1):
        variance += probmassfun.value_counts(normalize=True)[i] * (i - mean) ** 2
    return variance

'''
The main function reads in the thinkstats information, calculates its mean and variance, and then compares the
functions that I have written to them. The printout illustrates that the values for both the mean and variance are 
identical.
'''
def main():

    resp = nsfg.ReadFemResp()
    pmf = thinkstats2.Pmf(resp.numkdhh)

    fun_mean = PmfMean(resp.numkdhh)
    fun_var = PmfVar(resp.numkdhh, fun_mean)

    print('pmf mean', pmf.Mean())
    print('My function mean', fun_mean, '\n')

    print('pmf variance', pmf.Var())
    print('My function variance', fun_var)


if __name__ == '__main__':
    main()
