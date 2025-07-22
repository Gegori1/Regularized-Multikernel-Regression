# %% libraries
import os
import sys
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.gaussian_process.kernels import (
    Matern, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel
)
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed  # Import joblib
import time

sys.path.append(os.path.join("..", ".."))
from kernel_builder import KernelBuilder
# from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1MuExplicit, SvrSocpMultiKernelL1TraceExplicit, SvrQcqpMultiKernelL1Mu, SvrQcqpMultiKernelL1Trace
from svr_qcqp_multikernel_l1_socp import (
    SvrSocpMultiKernelL1MuExplicit, 
    SvrQcqpMultiKernelL1Mu, 
    SvrQcqpMultiKernelL1Trace, 
    SvrSocpMultiKernelL1TraceExplicit
)

import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle

# %%
print("multikernel_l1")
parameter_path = os.path.join("average_multikernel.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

# %% Obtain data
data = pd.read_csv("../../Sunspots.csv", index_col=0)
X, y = (
    pd.to_datetime(data["Date"]).to_frame().assign(Date=lambda k: k.Date.dt.year + k.Date.dt.month / 12).values,
    data["Monthly Mean Total Sunspot Number"].values
)

# data = pd.read_csv("../../SN_m_tot_V2.0.csv", sep=";")
# X, y = (
#     data.Joint.values.reshape(-1, 1), 
#     data['Monthly Mean Total Sunspot Number'].values
# )


X_, y_ = X[1811:-120], y[1811:-120]

scaler = StandardScaler()
X_ = scaler.fit_transform(X_)
# scaler.scale_ = [1]

# %% load parameters
mu_stds = {}
mu_mean = {}

# Define the worker function for parallel processing
def process_split(train_index, test_index, X_, y_, scaler, k_params, parameter):
    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]


    model = SvrSocpMultiKernelL1TraceExplicit(
        kernel_params=k_params,
        C=parameter["C"],
        epsilon=parameter["epsilon"],
        tau=parameter["tau"],
        kronecker_kernel=True,
    )
    
    # model = SvrSocpMultiKernelL1TraceExplicit(
    #     kernel_params=k_params,
    #     C=13.452909,
    #     epsilon=1.058241,
    #     tau=166.431719,
    #     kronecker_kernel=True,
    # )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)  # Keep for potential later use if needed
    metric = mean_absolute_error(y_test.flatten(), y_pred.flatten())


    return metric, y_pred.flatten(), X_[test_index], model.mu_, y_pred_train.flatten(), X_train, y_train, model.beta_


def run_forecast_analysis():
    global X_, y_, scaler, parameters, mu_stds, mu_mean, mu_all, k_params # Allow modification of global variables if needed by the logic within
    for parameter in parameters:

        # test_size = int(parameter['test_size'])
        test_size = 12

        n_splits = int(120 / test_size)

        k_params = [
            ("linear", {}),
            ("rbf", {"gamma": 1e-2}),
            ("rbf", {"gamma": 1e-1}),
            ("rbf", {"gamma": 1e0}),
            ("rbf", {"gamma": 1e2}),
            ("poly", {"degree": 2}),
            ("poly", {"degree": 3}),
            ("sigmoid", {}),
            (ExpSineSquared(periodicity=132 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=132 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=132 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=11 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=11 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=11 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=6 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=6 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=6 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=3 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=3 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=3 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=1 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=1 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=1 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=0.25 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=0.25 / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=0.25 / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=0.33 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=0.33  / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=0.33  / scaler.scale_[0], length_scale=1e1), {}),
            (ExpSineSquared(periodicity=0.5 / scaler.scale_[0]), {}),
            (ExpSineSquared(periodicity=0.5  / scaler.scale_[0], length_scale=1e-1), {}),
            (ExpSineSquared(periodicity=0.5  / scaler.scale_[0], length_scale=1e1), {}),
            (RationalQuadratic(), {}),
            (RationalQuadratic(length_scale=1e-1), {}),
            (RationalQuadratic(length_scale=1e1), {}),
            (Matern(nu=0.5), {}),
            (Matern(nu=0.5, length_scale=1e-1), {}),
            (Matern(nu=0.5, length_scale=1e1), {}),
            (Matern(nu=1.5), {}),
            (Matern(nu=1.5, length_scale=1e-1), {}),
            (Matern(nu=1.5, length_scale=1e1), {}),
            (Matern(nu=2.5), {}),
            (Matern(nu=2.5, length_scale=1e-1), {}),
            (Matern(nu=2.5, length_scale=1e1), {}),
            (ConstantKernel(), {})
        ]

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        # Use joblib to parallelize the loop
        # n_jobs=-1 uses all available CPU cores
        results = Parallel(n_jobs=6)(
            delayed(process_split)(train_index, test_index, X_, y_, scaler, k_params, parameter)
            for train_index, test_index in tscv.split(X_)
        )

        # Unpack results from the parallel execution
        metric_all = [result[0] for result in results]
        prediction_all_splits = [result[1] for result in results]
        indices_all_splits = [result[2] for result in results]
        mu_all = [result[3] for result in results]
        beta_all = [result[-1] for result in results]

        # Store last split's train predictions and data for plotting example
        last_y_pred_train = results[-1][4]
        last_X_train = results[-1][5]
        last_y_train = results[-1][6]

        # Flatten the lists for plotting
        prediction_all = [item for sublist in prediction_all_splits for item in sublist]
        # Use original X_ values for plotting indices
        indices_all = np.concatenate(indices_all_splits)


        plt.title(f"MAPE test_size: {test_size}. mape: ({np.mean(metric_all):.6f})")
        plt.plot(X_, y_, alpha=0.5, color="k", linestyle=':', label='true')
        # Ensure indices_all an
        # d prediction_all align correctly for plotting
        # Sort based on the time index if necessary, though TimeSeriesSplit should maintain order
        sort_order = np.argsort(indices_all[:, 0])  # Sort based on the time feature
        plt.plot(indices_all[sort_order], prediction_all, label='predict', color='tab:red', alpha=0.5)  # Use markers if points are sparse
        # Plot the training prediction from the *last* split as an example
        # plt.plot(last_X_train, last_y_pred_train, label="train_set (last split)", color="yellow", alpha=0.5, linestyle='', marker='.')
        plt.legend()
        plt.show()

        print(f"average mape ({test_size}): {np.mean(metric_all)}")
        # Calculate MAE for the last training set as an example
        print(f"mape train (last split): {mean_absolute_error(last_y_train, last_y_pred_train)}")

        # Convert mu_all to a NumPy array
        mu_all_array = np.array(mu_all)

        # Calculate the standard deviation of mu (column-wise)
        std_mu = np.std(mu_all_array, axis=0)

        mu_stds[f"{test_size}"] = std_mu
        mu_mean[f"{test_size}"] = np.mean(mu_all_array, axis=0)

if __name__ == "__main__":
    start_time = time.time()
    run_forecast_analysis()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# %%
print("multikernel")
kronecker_kernel = True

kernels_list = {}
for k, mu in zip(k_params, np.mean(mu_all, axis=0)):
    if isinstance(k[0], str):
        kernels_list[f"{k[0]}_{k[1]}"] = float(mu)
    else:
        kernels_list[f"{k[0].__str__()}_{k[1]}"] = float(mu)

if kronecker_kernel:
    kernels_list["kronecker_kernel"] = float(np.mean(mu_all, axis=0)[-1])

kernels_list

# %% Obtain kernel
kernel_builder = KernelBuilder(k_params, model=None, weights=np.mean(mu_all, axis=0))
kernel = kernel_builder.build_kernel()
kernel
# Save the kernel to a file
filename = 'custom_kernel_qcqp_l1.pkl'
pickle.dump(kernel, open(filename, 'wb'))
print(f"Kernel saved to {filename}")

# %% Use predicted kernel

data = pd.read_csv("../../SN_m_tot_V2.0.csv", sep=";")
X, y = (
    data.Joint.values.reshape(-1, 1), 
    data['Monthly Mean Total Sunspot Number'].values
)


X_, y_ = X[1811:], y[1811:]

scaler = StandardScaler()
X_ = scaler.fit_transform(X_)
# scaler.scale_ = [1]


kernel_builder = KernelBuilder(k_params, model=None, weights=np.mean(mu_all, axis=0))
kernel = kernel_builder.build_kernel()
kernel

# Fit SVR model
test_size = 12
n_splits = 171 // test_size

predictions = []
true_values = []
X_values = []

tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

for train_index, test_index in tscv.split(X_):
    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]
    
    svr = SVR(
        C = parameters[0]["C"],
        epsilon=parameters[0]["epsilon"],
        kernel=kernel, 
    )
    svr.fit(X_train, y_train)

    # Make predictions
    y_pred = svr.predict(X_test)
    
    predictions.extend(y_pred)
    true_values.extend(y_test)
    X_values.extend(X_test)

# Calculate metric
metric = mean_absolute_error(true_values, predictions)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_, y_, alpha=0.5, color="k", linestyle=':', label='true')
plt.plot(X_[-168:], predictions, label='predict', color='tab:red', alpha=0.5)
plt.title(f"SVR with custom kernel, MAE: {metric:.6f}")
plt.legend()
plt.show()

print(f"MAE: {metric}")

# %%
