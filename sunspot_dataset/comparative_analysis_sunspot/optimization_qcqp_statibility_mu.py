# %%
import os
import sys
import pandas as pd



from svr_qcqp_multi_kernel_l1_oldnew import (
    SvrQcqpMultiKernelL1Trace, SvrQcqpMultiKernelL1EpigraphTrace
)

from gcp_library.optimization_utils.optimizer_grid_search_ts import TimeSeriesOpt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process.kernels import (
    ExpSineSquared, WhiteKernel, RationalQuadratic, Matern, ConstantKernel
)

model_name = "trace"


# %% Data preprocessing
data = pd.read_csv("../../Sunspots.csv", index_col=0)
X, y = (
    pd.to_datetime(data["Date"]).to_frame().assign(Date = lambda k: k.Date.dt.year + k.Date.dt.month / 12).values, 
    data["Monthly Mean Total Sunspot Number"].values
)

X, y = X[1811:-120], y[1811:-120]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% parameters
logs_name = f"results_{model_name}/results_svr_qcqp_l1"
log_file = f"{logs_name}.jsonl"


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

params = {
    'C': (1e-2, 1e3, 3, "float"),
    'epsilon': (1e-2, 1e2, 3, "float"),
    'tau': (1e-4, 1e4, 20, "float"),
    'kernel_params': (k_params, None, None, "stationary"),
    'kronecker_kernel': (False, None, None, "stationary"),
}

if model_name == "trace":
    model_alg = SvrQcqpMultiKernelL1EpigraphTrace
elif model_name == "mu":
    model_alg = SvrQcqpMultiKernelL1Trace
else:
    raise NameError(f"There is no model {model_name}. Possible names are 'trace' or 'mu'")


# %%
optimizer = TimeSeriesOpt(
    X=X,
    y=y,
    model=model_alg,
    metric=mean_absolute_error,
    param_config=params,
    save_path=log_file,
    cloud_name="default",
    cloud_bucket="default",
    cloud_key="default",
    n_splits=3,
    n_jobs=4,
    test_size=60,
    upload_cloud_rate=1000,
    upload_to_cloud=False,
)

optimizer.optimize_splits()

print("\n--- Grid Search Optimization Complete ---")

# %%


