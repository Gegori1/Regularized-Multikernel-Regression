# %%
import sys
import os
import optuna
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, Matern, ConstantKernel
)
from gcp_library.optimization_utils.optimizer_ts_allatonce import TimeSeriesOpt
# from gcp_library.optimization_utils.optimizer_grid_search_ts import TimeSeriesOpt

from sklearn.svm import SVR

# %% Obtain data
data = pd.read_csv("../Sunspots.csv", index_col=0)
X, y = (
    pd.to_datetime(data["Date"]).to_frame().assign(Date=lambda k: k.Date.dt.year + k.Date.dt.month / 12).values,
    data["Monthly Mean Total Sunspot Number"].values
)

X_, y_ = X[1811:], y[1811:]

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=120, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# %% parameters
logs_name = "results/local_sunspot_qcqp_svr"
load_previous = False
log_file = f"{logs_name}.jsonl"

params = {
    'C': (1e-2, 6000, "float"),
    'epsilon': (1e-2, 120, "float"),
    'gamma': (1e-2, 5e2, "float")
}

# %% Optimization bayesian opt

optimizer = TimeSeriesOpt(
    X_train, y_train,
    SVR,
    mean_absolute_error,
    save_path=log_file,
    cloud_name="default",
    cloud_bucket="default'",
    cloud_key="default",
    n_splits=10,
    test_size=12,
    n_jobs=6,
    upload_cloud_rate=15,
    upload_to_cloud=False,
    **params
)

study = optimizer.optimize(n_trials=3_000)
# %%