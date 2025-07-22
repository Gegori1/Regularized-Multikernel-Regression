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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from joblib import Parallel, delayed  # Import joblib
import time

# sys.path.append(os.path.join(".."))
# from svr_qcqp_multi_kernel_l1 import SvrQcqpMultiKernelL1Mu, SvrQcqpMultiKernelL1EpigraphTrace
# from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1MuExplicit
import seaborn as sns
from xgboost import XGBRegressor

# %%
print("sklearn")
parameter_path = os.path.join("average_xgboost.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

# %% Obtain data
data = pd.read_csv("../Sunspots.csv", index_col=0)
# X, y = (
#     pd.to_datetime(data["Date"]).to_frame().assign(Date=lambda k: k.Date.dt.year + k.Date.dt.month / 12).values,
#     data["Monthly Mean Total Sunspot Number"].values
# )
data = pd.read_csv("../SN_m_tot_V2.0.csv", sep=";")
X, y = (
    data.Joint.values.reshape(-1, 1), 
    data['Monthly Mean Total Sunspot Number'].values
)

X_, y_ = X[1811:], y[1811:]

scaler = StandardScaler()
X_ = scaler.fit_transform(X_)

# %% load parameters
mu_stds = {}
mu_mean = {}

# Define the worker function for parallel processing
def process_split(train_index, test_index, X_, y_, scaler, parameter):
    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]

    model = XGBRegressor(
        random_state=32,
        max_depth=parameter["max_depth"],
        n_estimators=parameter["n_estimators"],
        learning_rate=parameter["learning_rate"],
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)  # Keep for potential later use if needed
    metric = mean_absolute_error(y_test.flatten(), y_pred.flatten())

    # Return necessary results from this split
    # Return X_test (original scaled values) for plotting indices
    return metric, y_pred.flatten(), X_[test_index], y_pred_train.flatten(), X_train, y_train


def run_forecast_analysis():
    global X_, y_, scaler, parameters, mu_stds, mu_mean, mu_all # Allow modification of global variables if needed by the logic within
    for parameter in parameters:

        # test_size = int(parameter['test_size'])
        test_size = 12

        n_splits = int(171 / test_size)

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        # Use joblib to parallelize the loop
        # n_jobs=-1 uses all available CPU cores
        results = Parallel(n_jobs=6)(
            delayed(process_split)(train_index, test_index, X_, y_, scaler, parameter)
            for train_index, test_index in tscv.split(X_)
        )

        # Unpack results from the parallel execution
        metric_all = [result[0] for result in results]
        prediction_all_splits = [result[1] for result in results]
        indices_all_splits = [result[2] for result in results]

        # Store last split's train predictions and data for plotting example
        last_y_pred_train = results[-1][3]
        last_X_train = results[-1][4]
        last_y_train = results[-1][5]

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

if __name__ == "__main__":
    start_time = time.time()
    run_forecast_analysis()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# %%
plt.plot(X[-100:], y[-100:])
# %%
