from typing import Tuple
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold


def scpd_function(
    x_train: np.ndarray,
    x_cal: np.ndarray,
    y_train: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_grid: np.ndarray,
    k: int,
    n_delta: int,
    shuffle_ind: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """ Split Conformal Predictive Distributions

    Parameters
    ----------
    x_train: np.ndarray
        Training features
    x_cal: np.ndarray
        Calibration features
    y_train: np.ndarray
        Training target
    y_cal: np.ndarray
        Calibration target
    x_test: np.ndarray
        Test features
    y_test: np.ndarray
        Test target
    y_grid: np.ndarray
        Target grid
    k: int
        Number of fold
    n_delta: int
        Number of discrete values in an interval
    shuffle_ind: bool
        Whether to shuffle indicators (default=True)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Output of split-conformal transducer (Q) and CRPS

    """

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
        n_cal = int(n / k)
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

    model = sm.OLS(y_train, x_train).fit()

    y_hat_cal = model.predict(x_cal)
    alpha_cal = y_cal - y_hat_cal

    crps = np.empty(n_test)

    # Initialize array to store Q
    # Q_min corresponds to the value of tau = 0
    # Q_max corresponds to the value of tau = 1
    q = np.empty((n_test, n_delta))
    q[:] = np.NaN

    alpha_y = np.empty((n_test, n_delta))
    alpha_y[:] = np.NaN

    y_hat = np.empty(n_test)
    y_hat[:] = np.NaN
    y_hat = model.predict(x_test)

    # Compute Q & CRPS for the test set
    for l in range(n_test):
        alpha_y[l] = y_grid - y_hat[l]
        for i in range(len(y_grid)):
            q[l, i] = 1 / (n_cal) * np.sum(alpha_cal < alpha_y[l, i])
        crps[l] = crps_function(y_grid, q[l], y_test[l])

    return (q, crps)


def ccpd_function(
    x_train_cal: np.ndarray,
    y_train_cal: np.ndarray,
    x_test: np.ndarray,
    y_grid: np.ndarray,
    k: int,
    n_delta: int
) -> Tuple[np.ndarray, np.ndarray]:
    """ Cross Conformal Predictive Distributions

    Parameters
    ----------
    x_train_cal: np.ndarray
        Train and calibration features
    y_train_cal: np.ndarray
        Train and calibration target
    x_test: np.ndarray
        Test features
    y_grid: np.ndarray
        Target grid
    k: int
        Number of fold
    n_delta: int
        Number of discrete values in an interval

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Q folds and p folds

    """

    xy_train_cal = np.concatenate(
        (x_train_cal, np.reshape(y_train_cal, (len(x_train_cal), 1))), axis=1
    )

    n = len(x_train_cal)
    n_test = len(x_test)
    s = np.empty(k)
    s[:] = np.NaN

    alpha_y_fold = np.empty((n_test, n_delta))

    q_folds = np.empty((n_test, n_delta, k))
    q_folds[:] = np.NaN

    p_folds = np.zeros((n_test, n_delta))
    p_folds = np.zeros((n_test, n_delta))

    kf = KFold(n_splits=k)

    alpha_y_fold = np.empty((n_test, n_delta))

    for l in range(len(x_test)):
        k_ = 0
        alpha_y_fold[:] = np.NaN
        for train_ind, cal_ind in kf.split(xy_train_cal):
            train_data = xy_train_cal[train_ind]
            cal_data = xy_train_cal[cal_ind]
            x_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            x_cal = cal_data[:, :-1]
            y_cal = cal_data[:, -1]
            s[k_] = len(x_cal)

            model = sm.OLS(y_train, x_train).fit()
            y_hat_cal = model.predict(x_cal)
            y_hat = model.predict(x_test)

            alpha_y_fold[l] = y_grid - y_hat[l]
            alpha_cal_fold = y_cal - y_hat_cal

            for i in range(len(y_grid)):
                q_folds[l, i, k_] = (
                    1 / (len(x_cal) + 1) * np.sum(alpha_cal_fold < alpha_y_fold[l, i])
                )
            k_ = k_ + 1

    for l in range(len(x_test)):
        for i in range(len(y_grid)):
            for k_ in range(k):
                p_folds[l, i] = p_folds[l, i] + s[k_] / n * q_folds[l, i, k_]

    return (q_folds, p_folds)


def crps_function(y_range: np.ndarray, q: np.ndarray, y: float) -> float:
    """ Continuously Ranked Probabilistic System

    Parameters
    ----------
    y_range: np.ndarray
        Range over which Q was computed
    q: np.ndarray
        Distribution function
    y: float
        Target

    Returns
    -------
    float
        Sum

    """
    step = (y_range[-1] - y_range[0]) / (len(y_range) - 1)
    ind = np.max(np.where(y >= y_range))
    q1 = q[0:(ind + 1)]

    sum1 = np.sum(q1[0:ind] ** 2) * step
    sum1 = sum1 + (q1[ind] ** 2) * (y - y_range[ind])

    q2 = q[ind:]
    q2 = (1 - q2)

    sum2 = 0

    if len(q2) > 1:
        sum2 = (q2[1] ** 2) * (step - (y - y_range[ind]))

    sum2 = sum2 + np.sum(q2[2:len(q2)]**2) * step
    total_sum = sum1 + sum2

    return total_sum
