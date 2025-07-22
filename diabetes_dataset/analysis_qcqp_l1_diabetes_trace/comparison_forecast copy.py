# %%
import os
import sys
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit # Added
from sklearn.gaussian_process.kernels import (
    Matern, ExpSineSquared, RationalQuadratic, ConstantKernel
)
# from sklearn.metrics.pairwise import laplacian_kernel

# Adjust path to import custom SVR modules
# Assumes this script is in analysis_qcqp_sunspot_final
# %%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1MuExplicit, SvrSocpMultiKernelL1TraceExplicit
# from svr_qcqp_multikernel_l1_socp_ import SvrQcqpMultiKernelL1Mu

def load_params_from_jsonl(file_path):
    """Loads the first parameter object from a JSONL file."""
    try:
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                return obj  # Return the first object found
    except FileNotFoundError:
        print(f"Error: Parameter file not found at {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def main():
    # --- 1. Load Data ---
    data_path = os.path.join(os.path.dirname(__file__), "SN_m_tot_V2.0.csv")
    try:
        data = pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
        
    X_raw = data.Joint.values.reshape(-1, 1)
    y_raw = data['Monthly Mean Total Sunspot Number'].values

    start_index = 1811
    if len(X_raw) <= start_index:
        print(f"Error: start_index {start_index} is out of bounds for data of length {len(X_raw)}.")
        return
        
    X_full_orig = X_raw[start_index:] # Original X values from start_index
    y_full = y_raw[start_index:] # Corresponding y values
    X_full_scaled = X_raw[start_index:]

    # --- 2. Scale Data (Overall for k_params and splitting) ---
    scaler_overall = StandardScaler()
    X_full_scaled = scaler_overall.fit_transform(X_full_orig)
    current_scaler_scale = scaler_overall.scale_[0]
    # current_scaler_scale = 1

    # --- 3. Load Hyperparameters ---
    params_svr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis_sklearn_sunspot", "average_sklearn.jsonl"))
    params_svr = load_params_from_jsonl(params_svr_path)
    if not params_svr:
        print(f"Could not load SVR params from {params_svr_path}. Exiting.")
        return

    params_socp_path = os.path.join(os.path.dirname(__file__), "average_multikernel.jsonl")
    params_socp = load_params_from_jsonl(params_socp_path)
    if not params_socp:
        print(f"Could not load SOCP SVR params from {params_socp_path}. Exiting.")
        return

    # --- 4. Define k_params for SvrSocpMultiKernelL1MuExplicit ---
    # current_scaler_scale is already defined from scaler_overall
    k_params_socp = [
        ("linear", {}),
        ("rbf", {"gamma": 1e-2}), ("rbf", {"gamma": 1e-1}), ("rbf", {"gamma": 1e0}), ("rbf", {"gamma": 1e2}),
        ("poly", {"degree": 2}), ("poly", {"degree": 3}),
        # ("sigmoid", {}),
        # (laplacian_kernel, {"gamma": 1e-1}), (laplacian_kernel, {"gamma": 1}), (laplacian_kernel, {"gamma": 1e1}),
        (ExpSineSquared(periodicity=132 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=132 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=132 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=11 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=11 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=11 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=6 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=6 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=6 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=3 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=3 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=3 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=1 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=1 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=1 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.25 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=0.25 / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.25 / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.33 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=0.33  / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.33  / current_scaler_scale, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.5 / current_scaler_scale), {}),
        (ExpSineSquared(periodicity=0.5  / current_scaler_scale, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.5  / current_scaler_scale, length_scale=1e1), {}),
        (RationalQuadratic(), {}),
        (RationalQuadratic(length_scale=1e-1), {}),
        (RationalQuadratic(length_scale=1e1), {}),
        (Matern(nu=0.5), {}), (Matern(nu=0.5, length_scale=1e-1), {}), (Matern(nu=0.5, length_scale=1e1), {}),
        (Matern(nu=1.5), {}), (Matern(nu=1.5, length_scale=1e-1), {}), (Matern(nu=1.5, length_scale=1e1), {}),
        (Matern(nu=2.5), {}), (Matern(nu=2.5, length_scale=1e-1), {}), (Matern(nu=2.5, length_scale=1e1), {}),
        (ConstantKernel(), {})
    ]

    # --- 5. TimeSeriesSplit Setup ---
    test_size_fold = 12
    num_total_samples_in_X_full = len(X_full_scaled)
    
    # Determine n_splits to cover approximately the last 171 samples,
    # consistent with original forecast.py logic.
    # The TimeSeriesSplit will generate test sets from the end of X_full_scaled.
    # We want the *concatenation* of these test sets to be the target forecast window.
    
    # If X_full_scaled itself is the data including the 171 points to be forecasted:
    # n_splits should be such that n_splits * test_size_fold covers the desired forecast horizon.
    # The original forecast files use int(171 / test_size)
    n_splits = int(171 / test_size_fold) 
    if n_splits == 0:
        print(f"Error: Not enough data in X_full (length {num_total_samples_in_X_full}) to create even one split for a forecast horizon of ~171 with test_size_fold {test_size_fold}")
        return

    # Adjust n_splits if X_full_scaled is too short for the desired number of splits and test_size
    # Max possible n_splits: (len(data) - test_size) // (train_increment_usually_test_size_for_rolling) + 1
    # Or more simply, ensure n_splits * test_size_fold + (n_splits-1)*some_train_increment + initial_train_size <= num_total_samples_in_X_full
    # sklearn's TimeSeriesSplit handles this by adjusting the train set size.
    # We just need to ensure that X_full_scaled is long enough for n_splits.
    # Minimum length for TimeSeriesSplit is n_splits * test_size (if train set can be empty for later splits, which is not typical)
    # or more realistically initial_train + (n_splits-1)*test_size + test_size
    
    min_samples_needed = test_size_fold * (n_splits + 1) # A rough lower bound for typical usage
    if num_total_samples_in_X_full < min_samples_needed : # test_size_fold * n_splits:
         print(f"Warning: X_full_scaled (length {num_total_samples_in_X_full}) might be too short for {n_splits} splits with test_size {test_size_fold}.")
         # Potentially reduce n_splits or handle error
         # For now, proceed as per original logic, sklearn will error if not possible.

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_fold)
    
    print(f"Using TimeSeriesSplit with n_splits={n_splits}, test_size_fold={test_size_fold}")
    print(f"Forecasting approximately {n_splits * test_size_fold} samples.")

    all_X_test_orig_folds = []
    all_y_test_folds = []
    all_y_pred_svr_folds = []
    all_y_pred_socp_folds = []
    all_mae_svr_folds = []
    all_mae_socp_folds = []

    # --- 6. Loop through TimeSeriesSplits ---
    fold_count = 0
    for train_index, test_index in tscv.split(X_full_scaled):
        fold_count += 1
        print(f"Processing fold {fold_count}/{n_splits}...")

        X_train_fold_scaled, X_test_fold_scaled = X_full_scaled[train_index], X_full_scaled[test_index]
        y_train_fold, y_test_fold = y_full[train_index], y_full[test_index]
        X_test_fold_orig = X_full_orig[test_index] # For plotting x-axis

        # Store for final concatenation
        all_X_test_orig_folds.append(X_test_fold_orig)
        all_y_test_folds.append(y_test_fold)

        # --- 6a. Sklearn SVR ---
        c_svr = params_svr.get("C")
        epsilon_svr = params_svr.get("epsilon")
        gamma_svr = params_svr.get("gamma")
        if None in [c_svr, epsilon_svr, gamma_svr]:
            print("Error: Missing SVR hyperparameters in params_svr for a fold. Skipping SVR for this fold.")
            all_y_pred_svr_folds.append(np.full_like(y_test_fold, np.nan)) # Placeholder
            all_mae_svr_folds.append(np.nan)
        else:
            model_svr = SVR(C=c_svr, epsilon=epsilon_svr, gamma=gamma_svr)
            model_svr.fit(X_train_fold_scaled, y_train_fold)
            y_pred_svr_fold = model_svr.predict(X_test_fold_scaled)
            all_y_pred_svr_folds.append(y_pred_svr_fold)
            all_mae_svr_folds.append(mean_absolute_error(y_test_fold, y_pred_svr_fold))

        # --- 6b. SvrSocpMultiKernelL1MuExplicit ---
        c_socp = params_socp.get("C")
        epsilon_socp = params_socp.get("epsilon")
        tau_socp = params_socp.get("tau")
        if None in [c_socp, epsilon_socp, tau_socp]:
            print("Error: Missing SOCP SVR hyperparameters in params_socp for a fold. Skipping SOCP for this fold.")
            all_y_pred_socp_folds.append(np.full_like(y_test_fold, np.nan)) # Placeholder
            all_mae_socp_folds.append(np.nan)
        else:
            model_socp = SvrSocpMultiKernelL1TraceExplicit(
                kernel_params=k_params_socp,
                C=c_socp,
                epsilon=epsilon_socp,
                tau=tau_socp,
                kronecker_kernel=True
            )
            model_socp.fit(X_train_fold_scaled, y_train_fold)
            y_pred_socp_fold = model_socp.predict(X_test_fold_scaled)
            all_y_pred_socp_folds.append(y_pred_socp_fold)
            all_mae_socp_folds.append(mean_absolute_error(y_test_fold, y_pred_socp_fold))
    
    if not all_X_test_orig_folds:
        print("Error: No splits were processed. Cannot generate plot or metrics.")
        return

    # --- 7. Concatenate results and Calculate Average MAE ---
    X_test_final_orig = np.concatenate(all_X_test_orig_folds)
    y_test_final = np.concatenate(all_y_test_folds)
    y_pred_svr_final = np.concatenate(all_y_pred_svr_folds)
    y_pred_socp_final = np.concatenate(all_y_pred_socp_folds)

    avg_mae_svr = np.nanmean(all_mae_svr_folds) # Use nanmean in case of skipped folds
    avg_mae_socp = np.nanmean(all_mae_socp_folds)

    print(f"\\n--- Final Results ---")
    print(f"Sklearn SVR Average MAE: {avg_mae_svr:.4f}")
    print(f"SvrSocpMultiKernelL1MuExplicit Average MAE: {avg_mae_socp:.4f}")
    
    num_forecast_points = len(y_test_final) # Actual number of points forecasted

    # --- 8. Plot Results ---
    plt.figure(figsize=(14, 7))
    time_index_plot = X_test_final_orig[:,0]

    plt.plot(time_index_plot, y_test_final, label='Actual Values', color='black', linestyle='-', marker='o', markersize=3)
    plt.plot(time_index_plot, y_pred_svr_final, label=f'Sklearn SVR (Avg MAE: {avg_mae_svr:.2f})', color='blue', alpha=0.8, linestyle='--')
    plt.plot(time_index_plot, y_pred_socp_final, label=f'QCQP SVR (Avg MAE: {avg_mae_socp:.2f})', color='red', alpha=0.8, linestyle=':')

    plt.title(f'Sunspot Forecast Comparison (Rolling Forecast: {num_forecast_points} Samples in {n_splits} splits of {test_size_fold})', fontsize=18)
    plt.xlabel('Time (Original Data Joint Value)', fontsize=16)
    plt.ylabel('Monthly Mean Total Sunspot Number', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(os.path.dirname(__file__), "comparison_forecast_plot_rolling.png")
    try:
        plt.savefig("comparison_forecast_plot_rolling.svg", format="svg")
        plt.savefig(plot_filename)
        print(f"\\nPlot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show()
    
# %%

if __name__ == "__main__":
    main()

# %%
