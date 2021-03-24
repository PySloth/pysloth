# Implement Cross Conformal Predictive Distributions

import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm


def CCPD_function(x_train_cal, y_train_cal, x_test, y_test, y_grid, K, n_delta):
    xy_train_cal = np.concatenate((x_train_cal, np.reshape(y_train_cal, (len(x_train_cal), 1))), axis=1)

    n = len(x_train_cal)
    n_test = len(x_test)
    S = np.empty(K)
    S[:] = np.NaN

    alpha_y_fold = np.empty((n_test, n_delta))

    Q_folds = np.empty((n_test, n_delta, K))
    Q_folds[:] = np.NaN

    p_folds = np.zeros((n_test, n_delta))
    p_folds = np.zeros((n_test, n_delta))

    kf = KFold(n_splits=K)


# compute Q

    alpha_y_fold = np.empty((n_test, n_delta))

    for l in range(len(x_test)):
        k = 0
        alpha_y_fold[:] = np.NaN
        for train_ind, cal_ind in kf.split(xy_train_cal):
            train_data = xy_train_cal[train_ind]
            cal_data = xy_train_cal[cal_ind]
            x_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            x_cal = cal_data[:, :-1]
            y_cal = cal_data[:, -1]
            S[k] = len(x_cal)
            # train the model on the k-th training set
            model = sm.OLS(y_train, x_train).fit()
            y_hat_cal = model.predict(x_cal)  # predictions on the calibration set
            y_hat = model.predict(x_test)  # make predictions on the test set

            alpha_y_fold[l] = y_grid - y_hat[l]  # compute alpha for test point
            alpha_cal_fold = y_cal - y_hat_cal

            # Compute Q
            for i in range(len(y_grid)):
                Q_folds[l, i, k] = 1 / (len(x_cal) + 1) * np.sum(alpha_cal_fold < alpha_y_fold[l, i])
            k = k + 1

    # compute p
    for l in range(len(x_test)):
        for i in range(len(y_grid)):
            for k in range(K):
                p_folds[l, i] = p_folds[l, i] + S[k] / n * Q_folds[l, i, k]

    return (Q_folds, p_folds)  # (Q_folds, p_folds)
