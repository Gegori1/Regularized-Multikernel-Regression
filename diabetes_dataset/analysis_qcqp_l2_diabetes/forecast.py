# %%
import os
import sys
import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, Matern, ConstantKernel
)


sys.path.append(os.path.join("..", ".."))
# from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1TraceExplicit, SvrQcqpMultiKernelL1Trace
from svr_qcqp_multi_kernel_l2 import SvrQcqpMultiKernelL2Trace
from sklearn.utils import resample
import numpy as np

# %%
print("multikernel")
parameter_path = os.path.join("average_multikernel.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

# %%
diabetes = pd.read_csv("../../diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values
X_test, y_test = X_test.values, y_test.values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# %%

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

# %% predict
errors = []
for params in parameters:
    model = SvrQcqpMultiKernelL2Trace(
        kernel_params=k_params,
        C=params["C"],
        epsilon=params["epsilon"],
        tau=params["tau"],
    )
    model.fit(X_train, y_train)

    n_iterations = 1000
    n_size = int(len(X_test) * 0.8)
    scores = []
    for i in range(n_iterations):
        # print(f"iteration {i} ...")

        idx = resample(np.arange(len(X_test)), n_samples=n_size, replace=False, random_state=i+1)
        X, y = X_test[idx], y_test[idx]

        yhat = model.predict(X).flatten()
        score = mape(y, yhat)
        scores.append(score)

    error = np.array(scores).mean()
    print('MAPE: %.5f' % error)
    errors.append(error)

print('MAPE mean: %.5f' % np.mean(errors))
    

# %%
