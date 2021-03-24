# Implement Split Conformal Predictive Distributions

import numpy as np
import statsmodels.api as sm


def SCPD_function(x_train, x_cal, y_train, y_cal, x_test, y_test, y_grid, K, n_delta, shuffle_ind=True):
    n_test = len(x_test)
    if shuffle_ind == True:
        x_train_cal = np.concatenate((x_train, x_cal), axis=0)
        y_train_cal = np.concatenate((y_train, y_cal), axis=0)
        if (len(x_train_cal.shape) == 1):
            x_train_cal = np.reshape(x_train_cal, (-1, 1))

        y_train_cal = np.reshape(y_train_cal, (-1, 1))
        xy_train_cal = np.concatenate((x_train_cal, y_train_cal), axis=1)
        np.random.shuffle(xy_train_cal)
        n = len(xy_train_cal)
        n_cal = int(n/K)
        n_train = n - n_cal

        train_data = xy_train_cal[0:n_train]
        cal_data = xy_train_cal[n_train:]

        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_cal = cal_data[:, :-1]
        y_cal = cal_data[:, -1]

    else:
        n_train = len(x_train)
        n_cal = len(x_cal)
        n = n_train + n_cal

    # Note the difference in argument order
    model = sm.OLS(y_train, x_train).fit()

    # Split (inductive) Conformal Predictive Distributions
    y_hat_cal = model.predict(x_cal)  # make predictions on the calibration set
    alpha_cal = y_cal - y_hat_cal  # compute alphas (conformity scores) on the calibrations set

    # initialize vector for storing CRPS for the test set
    CRPS = np.empty(n_test)

    # initialize array to store Q (the output of split conformal trancducer as per formua (1) in the paper)
    # Q_min corresponds to the value of tau = 0, Q_max corresponds to the value of tau = 1
    Q = np.empty((n_test, n_delta))
    Q[:] = np.NaN
    # initialize array for storing alphas for the test set
    alpha_y = np.empty((n_test, n_delta))
    alpha_y[:] = np.NaN

    # initialize array to store predictions of the underlying algorithm on the test set
    y_hat = np.empty(n_test)
    y_hat[:] = np.NaN

    # compute y_hat (predictions of the underlyin algorithm) on the test and calibration sets
    y_hat = model.predict(x_test)  # make predictions by the model for the test set
    # delta = 6 * np.std(y_hat) # +/- delta around y_hat (predicted value of y_test, length of range around y_hat is 2 * delta
    #y_grid = np.linspace(y_hat.min() - delta, y_hat.max() + delta, n_delta)

    # compute Q & CRPS for the test set
    for l in range(n_test):
        # create the grid around y_hat
        alpha_y[l] = y_grid - y_hat[l]

        for i in range(len(y_grid)):
            Q[l, i] = 1 / (n_cal) * np.sum(alpha_cal < alpha_y[l, i])

        CRPS[l] = CRPS_function(y_grid, Q[l], y_test[l])

    return (Q, CRPS)


def CRPS_function(yRange, Q, y):
    step = (yRange[-1] - yRange[0])/(len(yRange)-1)
    ind = np.max(np.where(y >= yRange))
    Q1 = Q[0:(ind+1)]

    #Q1[Q1<=0] = 0
    sum1 = np.sum(Q1[0:ind] ** 2) * step
    sum1 = sum1 + (Q1[ind] ** 2) * (y-yRange[ind])

    Q2 = Q[ind:]
    Q2 = (1 - Q2)
    #Q2[Q2<=0] = 0

    sum2 = 0

    if len(Q2) > 1:
        sum2 = (Q2[1] ** 2) * (step - (y - yRange[ind]))

    sum2 = sum2 + np.sum(Q2[2:len(Q2)]**2) * step
    sum = sum1 + sum2

    return(sum)
