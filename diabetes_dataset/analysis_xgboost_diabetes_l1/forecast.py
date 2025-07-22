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


from xgboost import XGBRegressor
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


# %% predict
errors = []
for params in parameters:
    model = XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                random_state=params["random_state"]
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
