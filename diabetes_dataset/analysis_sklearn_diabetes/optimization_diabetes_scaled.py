# %%
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split




# from gcp_library.optimization_utils.optimizer_ts_allatonce import TimeSeriesOpt
# from gcp_library.optimization_utils.optimizer_grid_search_ts import TimeSeriesOpt
from gcp_library.optimization_utils.optimizer_crossvalidation import CrossValidationOptimizer

from sklearn.svm import SVR

# %% Obtain data
diabetes = pd.read_csv("../../diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values

# %% parameters
logs_name = "results/local_diabetes_sklearn_svr"
load_previous = False
log_file = f"{logs_name}.jsonl"

params = {
    'C': (700, 1500, "float"),
    'epsilon': (8, 30, "float"),
    'gamma': (1e-4, 0.5, "float")
}

# %% Opimization cross validation

optimizer = CrossValidationOptimizer(
    X_train, y_train,
    SVR,
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

study = optimizer.optimize(n_trials=5_000)
# %%