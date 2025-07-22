# %%
import sys
import os
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, ConstantKernel
from gcp_library.optimization_utils.optimizer_crossvalidation import CrossValidationOptimizer


sys.path.append(os.path.join("../.."))
from svr_qcqp_multi_kernel_l2 import SvrQcqpMultiKernelL2EpigraphTrace

# %% Obtain data
diabetes = pd.read_csv("../../diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values

# %% parameters
logs_name = "results/trace_constrained_stdscaler_trace_opt_5folds_false"
load_previous = False
log_file = f"{logs_name}.jsonl"

k_params = [
    ("linear", {}),
    ("rbf", {"gamma": 1e-2}),
    ("rbf", {"gamma": 1.1e-2}),
    ("rbf", {"gamma": 1.2e-2}),
    ("rbf", {"gamma": 1.3e-2}),
    ("rbf", {"gamma": 1.4e-2}),
    ("rbf", {"gamma": 1.5e-3}),
    ("rbf", {"gamma": 1.6e-2}),
    ("rbf", {"gamma": 1.7e-2}),
    ("rbf", {"gamma": 1.8e-2}),
    ("rbf", {"gamma": 1.9e-2}),
    ("rbf", {"gamma": 1e-1}),
    ("rbf", {"gamma": 0.2}),
    ("rbf", {"gamma": 0.3}),
    ("rbf", {"gamma": 0.4}),
    ("rbf", {"gamma": 0.5}),
    ("rbf", {"gamma": 0.6}),
    ("rbf", {"gamma": 0.7}),
    ("rbf", {"gamma": 0.8}),
    ("rbf", {"gamma": 0.9}),
    ("rbf", {"gamma": 1e0}),
    ("rbf", {"gamma": 10}),
    ("rbf", {"gamma": 20}),
    ("rbf", {"gamma": 30}),
    ("rbf", {"gamma": 40}),
    ("rbf", {"gamma": 50}),
    ("rbf", {"gamma": 60}),
    ("rbf", {"gamma": 70}),
    ("rbf", {"gamma": 80}),
    ("rbf", {"gamma": 90}),
    ("rbf", {"gamma": 1e2}),
    ("rbf", {"gamma": 2e2}),
    ("rbf", {"gamma": 3e2}),
    ("rbf", {"gamma": 4e2}),
    ("rbf", {"gamma": 5e2}),
    ("rbf", {"gamma": 6e2}),
    ("rbf", {"gamma": 7e2}),
    ("rbf", {"gamma": 8e2}),
    ("rbf", {"gamma": 9e2}),
    ("rbf", {"gamma": 1e3}),
    ("poly", {"degree": 2}),
    ("poly", {"degree": 3}),
    ("sigmoid", {}),
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

params = {
    'C': (1e-5, 100, "float"),
    'epsilon': (1e-3, 30, "float"),
    'tau': (1e-5, 1e3, "float"),
    'kernel_params': (k_params, None, "stationary"),
}

# %% Opimization cross validation
optimizer = CrossValidationOptimizer(
    X_train, y_train,
    SvrQcqpMultiKernelL2EpigraphTrace,
    mean_absolute_percentage_error,
    param_distributions=params,
    save_path=log_file,
    cloud_name="default",
    cloud_bucket="default",
    cloud_key="default",
    n_folds=5,
    n_jobs=4,
    random_state=42,
    standardize=True,
    upload_to_cloud=False
)

study = optimizer.optimize(n_trials=3_000)

