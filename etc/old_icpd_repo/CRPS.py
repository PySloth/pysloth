"""
Function CRPS (formula in "Experimental results" section of the "Conformal predictive 
distributions based on kernel ridge regression" paper)
Created on Fri Nov 24 18:24:26 2017
@author: VM

# yRange - range over which Q was computed
# Q - distribution function (the prediction)
# y - y-value of the test point
# n - number of points in the training set
"""
import numpy as np


def CRPS(yRange, Q, y, n):
    step = (yRange[-1] - yRange[0])/(len(yRange)-1)
    ind = np.max(np.where(y >= yRange))
    Q1 = Q[0:(ind+1)]
    Q1 = (Q1 - 0.5/(n+1))
    Q1[Q1 <= 0] = 0

    sum1 = np.sum(Q1[0:ind] ** 2) * step
    sum1 = sum1 + (Q1[ind] ** 2) * (y-yRange[ind])

    Q2 = Q[ind:]
    Q2 = (1 - 0.5/(n+1) - Q2)
    Q2[Q2 <= 0] = 0

    sum2 = (Q2[1] ** 2) * (step - (y - yRange[ind]))
    sum2 = sum2 + np.sum(Q2[2:len(Q2)]**2) * step
    sum = sum1 + sum2
    return(sum)
