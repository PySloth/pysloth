.. image:: https://img.shields.io/pypi/v/pysloth
   :target: https://pypi.org/project/pysloth/

.. image:: https://readthedocs.org/projects/pysloth/badge/?version=latest
   :target: https://pysloth.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/PySloth/pysloth/actions/workflows/build-and-tests.yml/badge.svg
   :target: https://github.com/PySloth/pysloth

.. image:: https://codecov.io/gh/PySloth/pysloth/branch/main/graph/badge.svg?token=gAMTe66DIg
   :target: https://codecov.io/gh/PySloth/pysloth

pysloth
=======
A Python package for Probabilistic Prediction

v0.0.3

Installation
------------
This package supports Python 3.6, 3.7, 3.8, and 3.9

Install via PyPI
~~~~~~~~~~~~~~~~
Run ``pip install pysloth``

Install from repository
~~~~~~~~~~~~~~~~~~~~~~~
* Clone repo with SSH ``git clone git@github.com:PySloth/pysloth.git``
* Change directory to where ``README.md`` (this file) is located and run ``pip install .``

Quickstart
----------
The following is a code sample showing ``scpd`` and ``ccpd`` in action

.. code-block:: python

    from pysloth import scpd_function, ccpd_function

    import numpy as np
    import statsmodels.api as sm

    np.random.seed(142)
    n = 1000  # training set
    m = int(0.8 * n)  # proper training set
    n_cal = n - m  # Calibration = training - proper training

    n_test = 100
    sd_noise = 1

    n_delta = 1000  # discretization for y values in interval y_hat +/- 3 * delta
    w = 2  # the weights

    x_train = w * np.random.random(m) - 1
    x_cal = w * np.random.random(n_cal) - 1
    x_test = w * np.random.random(n_test) - 1

    y_train = w * x_train + np.random.randn(m) * sd_noise
    y_cal = w * x_cal + np.random.randn(n_cal) * sd_noise
    y_test = w * x_test + np.random.randn(n_test) * sd_noise

    x_train_cal = np.reshape(np.hstack((x_train, x_cal)), (n, 1))
    y_train_cal = np.reshape(np.hstack((y_train, y_cal)), (n, 1))
    xy_train_cal = np.hstack((x_train_cal, y_train_cal))

    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_train)
    model.summary()
    y_hat = model.predict(x_test)

    delta = 3 * np.std(y_hat)
    y_grid = np.linspace(y_hat.min() - delta, y_hat.max() + delta, n_delta)

    print(ccpd_function(x_train_cal, y_train_cal, x_test, y_grid, 5, n_delta))
    print(scpd_function(x_train, x_cal, y_train, y_cal, x_test, y_test, y_grid, 5, n_delta))

