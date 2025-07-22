# %%
import sys
import os
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, Matern, ConstantKernel
)
# from gcp_library.optimization_utils.optimizer_ts_allatonce import TimeSeriesOpt
# from gcp_library.optimization_utils.optimizer_1_partition import TimeSeriesOpt
# from gcp_library.optimization_utils.optimizer_ts_parallel_splits import TimeSeriesOptParallelSplits
# from gcp_library.optimization_utils.optimizer_grid_search_ts import TimeSeriesOpt
# from gcp_library.optimization_utils.optimizer_grid_search_1_partition import GridSearchOpt1Partition
# from gcp_library.optimization_utils.sequential_simulator_ts import SequentialSimulatorTimeSeries
from gcp_library.optimization_utils.optimizer_crossvalidation import CrossValidationOptimizer

sys.path.append(os.path.join("..", ".."))
from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1TraceExplicit, SvrQcqpMultiKernelL1Trace
# from svr_qcqp_multi_kernel_l1 import SvrQcqpMultiKernelL1Trace
# from svr_qcqp_multi_kernel_l1_ import svr_qcqp_multi_kernel_l1

# %% Obtain data
diabetes = pd.read_csv("../../diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values

# %% parameters
logs_name = "results/stdscaler_trace_opt_5folds"
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
    'C': (1e-5, 1e3, "float"),
    'epsilon': (1e-3, 30, "float"),
    'tau': (2e3, 5e3, "float"),
    'kernel_params': (k_params, None, "stationary"),
    'kronecker_kernel': (True, None, "stationary")
}

# C_list = [ 9.        ,  9.53846154, 10.07692308, 10.61538462, 11.15384615,
#        11.69230769, 12.23076923, 12.76923077, 13.30769231, 13.84615385,
#        14.38461538, 14.92307692, 15.46153846, 16.        , 16.53846154,
#        17.07692308, 17.61538462, 18.15384615, 18.69230769, 19.23076923,
#        19.76923077, 20.30769231, 20.84615385, 21.38461538, 21.92307692,
#        22.46153846, 23.        , 23.53846154, 24.07692308, 24.61538462,
#        25.15384615, 25.69230769, 26.23076923, 26.76923077, 27.30769231,
#        27.84615385, 28.38461538, 28.92307692, 29.46153846, 30.        ]


# tau_list = [ 16.00561638,  25.8192828 ,  35.63294922,  45.44661564,
#         55.26028206,  65.07394848,  74.8876149 ,  84.70128133,
#         94.51494775, 104.32861417, 114.14228059, 123.95594701,
#        133.76961343, 143.58327985, 153.39694627, 163.21061269,
#        173.02427912, 182.83794554, 192.65161196, 202.46527838,
#        212.2789448 , 222.09261122, 231.90627764, 241.71994406,
#        251.53361048, 261.34727691, 271.16094333, 280.97460975,
#        290.78827617, 300.60194259, 310.41560901, 320.22927543,
#        330.04294185, 339.85660827, 349.6702747 , 359.48394112,
#        369.29760754, 379.11127396, 388.92494038, 398.7386068 ]

# params = {
#     'C': (C_list, "sequential"),
#     'epsilon': (0.06956321, "stationary"),
#     'tau': (tau_list, "sequential"),
#     'kernel_params': (k_params, "stationary"),
#     'kronecker_kernel': (True, "stationary")
# }


# %% Opimization cross validation

optimizer = CrossValidationOptimizer(
    X_train, y_train,
    SvrSocpMultiKernelL1TraceExplicit,
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


# %% Optimization grid search

# optimizer = TimeSeriesOpt(
#     X_train, y_train,
#     SvrSocpMultiKernelL1MuExplicit,
#     mean_absolute_error,
#     param_config=params,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default",
#     cloud_key="default",
#     n_splits=10,
#     test_size=12,
#     n_jobs=4,
#     upload_to_cloud=False
# )

# study = optimizer.optimize_splits()

# %% Optimization time series

# optimizer = TimeSeriesOpt(
#     X_train, y_train,
#     svr_qcqp_multi_kernel_l1,
#     mean_absolute_error,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default'",
#     cloud_key="default",
#     n_splits=10,
#     test_size=12,
#     n_jobs=4,
#     upload_cloud_rate=15,
#     upload_to_cloud=False,
#     **params
# )

# study = optimizer.optimize(n_trials=600)

# %% Optimization time series one by one in parallel

# optimizer = TimeSeriesOptParallelSplits(
#     X_train, y_train,
#     SvrQcqpMultiKernelL1EpigraphTrace,
#     mean_absolute_error,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default'",
#     cloud_key="default",
#     hyperparams_config=params,
#     n_splits=10,
#     test_size=12,
#     n_jobs_splits=4,
#     upload_cloud_rate=15,
#     upload_to_cloud=False,
# )

# study = optimizer.optimize(n_trials=600)
# %% Grid search one partition

# optimizer_instance = GridSearchOpt1Partition(
#     X_train=X_train,
#     y_train=y_train,
#     X_val=X_test,
#     y_val=y_test,
#     model=SvrQcqpMultiKernelL1Trace,
#     metric=mean_absolute_error,
#     param_config=params,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default'",
#     cloud_key="default",
#     upload_cloud_rate=100,
#     n_jobs=4,
#     upload_to_cloud=False
# )

# optimizer_instance.optimize_single_split()

# %% Optimization one time
# optimizer = TimeSeriesOpt(
#     X_train=X_train,
#     y_train=y_train,
#     X_val=X_test,
#     y_val=y_test,
#     model=SvrQcqpMultiKernelL1Trace,
#     metric=mean_absolute_error,
#     param_config=params,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default",
#     cloud_key="default",
#     upload_cloud_rate=100,
#     n_jobs=4,
#     upload_to_cloud=False, 
# )

# optimizer.optimize(n_trials=600)

# %% Sequential simulations time series

# optimizer = SequentialSimulatorTimeSeries(
#     X_train, y_train,
#     SvrSocpMultiKernelL1TraceExplicit,
#     mean_absolute_error,
#     save_path=log_file,
#     cloud_name="default",
#     cloud_bucket="default'",
#     cloud_key="default",
#     n_splits=10,
#     test_size=12,
#     upload_cloud_rate=15,
#     upload_to_cloud=False,
#     param_config=params
# )

# study = optimizer.run_simulations()