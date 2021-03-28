from typing import Tuple
import numpy as np


def scpd_function(
    x_train: np.ndarray,
    x_cal: np.ndarray,
    y_train: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_grid: np.ndarray,
    K: int,
    n_delta: int,
    shuffle_ind: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """ SCPD

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
    K: int
        Number of fold
    n_delta: int
        Number of discrete values in an interval
    shuffle_ind: bool = True
        Whether to shuffle indicators

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Q and CRPS

    """

    # NOTE: To implement
    ...


def ccpd_function(
    x_train_cal: np.ndarray,
    y_train_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_grid: np.ndarray,
    K: int,
    n_delta: int
) -> Tuple[np.ndarray, np.ndarray]:
    """ CCPD

    Parameters
    ----------
    x_train_cal: np.ndarray
        Train and calibration features
    y_train_cal: np.ndarray
        Train and calibration target
    x_test: np.ndarray
        Test features
    y_test: np.ndarray
        Test target
    y_grid: np.ndarray
        Target grid
    K: int
        Number of fold
    n_delta: int
        Numbe of discrete values in an interval

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Q folds and p folds

    """

    # NOTE: To implement
    ...


def crps_function(yRange, Q, y) -> float:
    """ CRPS

    Parameters
    ----------
    yRange: np.ndarray
        Target range
    Q: np.ndarray
        Q
    y: float
        Target

    Returns
    -------
    float
        Sum

    """

    # NOTE: To implement
    ...
