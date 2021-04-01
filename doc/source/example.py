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
