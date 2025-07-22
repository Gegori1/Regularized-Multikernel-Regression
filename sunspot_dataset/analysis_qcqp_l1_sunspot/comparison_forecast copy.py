# %%
import os
import sys
import pickle
import pandas as pd
import numpy as np
import jsonlines
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.gaussian_process.kernels import (
    Matern, ExpSineSquared, RationalQuadratic, ConstantKernel
)

# Adjust path to import custom SVR modules
# Assumes this script is in analysis_qcqp_sunspot_final
# %%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from svr_qcqp_multikernel_l1_socp import SvrSocpMultiKernelL1TraceExplicit
from svr_qcqp_multi_kernel_l1_ import svr_qcqp_multi_kernel_l1

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
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "SN_m_tot_V2.0.csv")
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

    params_xgb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis_xgboost_sunspot", "average_xgboost.jsonl"))
    params_xgb = load_params_from_jsonl(params_xgb_path)
    if not params_xgb:
        print(f"Could not load XGB params from {params_xgb_path}. Exiting.")
        return
        
    params_qcqp_trace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis_qcqp_l1_sunspot_trace", "average_multikernel.jsonl"))
    params_qcqp_trace = load_params_from_jsonl(params_qcqp_trace_path)
    if not params_qcqp_trace:
        print(f"Could not load trace constraned QCQP params from {params_qcqp_trace_path}. Exiting.")
        return

    params_socp_path = os.path.join(os.path.dirname(__file__), "average_multikernel.jsonl")
    params_socp = load_params_from_jsonl(params_socp_path)
    if not params_socp:
        print(f"Could not load SOCP SVR params from {params_socp_path}. Exiting.")
        return
    
    # --- Load Custom Kernels ---
    kernel_qcqp_l1_path = os.path.join(os.path.dirname(__file__), "custom_kernel_qcqp_l1.pkl")
    try:
        with open(kernel_qcqp_l1_path, 'rb') as f:
            custom_kernel_qcqp_l1 = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Kernel file not found at {kernel_qcqp_l1_path}")
        return

    kernel_qcqp_l1_trace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis_qcqp_l1_sunspot_trace", "custom_kernel_qcqp_l1_trace.pkl"))
    try:
        with open(kernel_qcqp_l1_trace_path, 'rb') as f:
            custom_kernel_qcqp_l1_trace = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Kernel file not found at {kernel_qcqp_l1_trace_path}")
        return


    # --- 4. Define k_params for SvrSocpMultiKernelL1MuExplicit ---
    # current_scaler_scale is already defined from scaler_overall
    k_params_socp = [
        ("linear", {}),
        ("rbf", {"gamma": 1e-2}), ("rbf", {"gamma": 1e-1}), ("rbf", {"gamma": 1e0}), ("rbf", {"gamma": 1e2}),
        ("poly", {"degree": 2}), ("poly", {"degree": 3}),
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
    all_y_pred_qcqp_folds = []
    all_y_pred_xgb_folds = []
    all_y_pred_custom_qcqp_l1_folds = []
    all_y_pred_custom_qcqp_l1_trace_folds = []
    
    all_mae_svr_folds = []
    all_mae_socp_folds = []
    all_mae_qcqp_folds = []
    all_mae_xgb_folds = []
    all_mae_custom_qcqp_l1_folds = []
    all_mae_custom_qcqp_l1_trace_folds = []
        

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

            # --- 6b. XGBoost ---
            n_estimators_xgb = params_xgb.get("n_estimators")
            learning_rate_xgb = params_xgb.get("learning_rate")
            max_depth_xgb = params_xgb.get("max_depth")
            if None in [n_estimators_xgb, learning_rate_xgb, max_depth_xgb]:
                print("Error: Missing XGBoost hyperparameters in params_xgb for a fold. Skipping XGBoost for this fold.")
                all_y_pred_xgb_folds.append(np.full_like(y_test_fold, np.nan)) # Placeholder
                all_mae_xgb_folds.append(np.nan)
            else:
                model_xgb = xgb.XGBRegressor(
                    n_estimators=n_estimators_xgb,
                    learning_rate=learning_rate_xgb,
                    max_depth=max_depth_xgb,
                    random_state=32  # For reproducibility
                )
                model_xgb.fit(X_train_fold_scaled, y_train_fold)
                y_pred_xgb_fold = model_xgb.predict(X_test_fold_scaled)
                all_y_pred_xgb_folds.append(y_pred_xgb_fold)
                all_mae_xgb_folds.append(mean_absolute_error(y_test_fold, y_pred_xgb_fold))

            # --- 6c. QCQP Trace Constrained ---
            c_qcqp = params_qcqp_trace.get("C")
            epsilon_qcqp = params_qcqp_trace.get("epsilon")
            tau_qcqp = params_qcqp_trace.get("tau")
            
            if None in [c_qcqp, epsilon_qcqp, tau_qcqp]:
                print("Error: Missing QCQP hyperparameters in params_qcqp_trace for a fold. Skipping QCQP for this fold.")
                all_y_pred_qcqp_folds.append(np.full_like(y_test_fold, np.nan)) # Placeholder
                all_mae_qcqp_folds.append(np.nan)
            else:
                model_qcqp = svr_qcqp_multi_kernel_l1(
                    C=c_qcqp,
                    epsilon=epsilon_qcqp,
                    tau=tau_qcqp,
                    kernel_params=k_params_socp,
                    kronecker_kernel=True,
                    verbose=False
                )
                model_qcqp.fit(X_train_fold_scaled, y_train_fold)
                y_pred_qcqp_fold = model_qcqp.predict(X_test_fold_scaled).flatten()
                all_y_pred_qcqp_folds.append(y_pred_qcqp_fold)
                all_mae_qcqp_folds.append(mean_absolute_error(y_test_fold, y_pred_qcqp_fold))

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
            
        # --- 6d. Custom Kernel QCQP L1 SVR ---
        c_custom_qcqp_l1 = params_socp.get("C")
        epsilon_custom_qcqp_l1 = params_socp.get("epsilon")
        if None in [c_custom_qcqp_l1, epsilon_custom_qcqp_l1]:
            print("Error: Missing Custom Kernel QCQP L1 SVR hyperparameters. Skipping for this fold.")
            all_y_pred_custom_qcqp_l1_folds.append(np.full_like(y_test_fold, np.nan))
            all_mae_custom_qcqp_l1_folds.append(np.nan)
        else:
            model_custom_qcqp_l1 = SVR(C=c_custom_qcqp_l1, epsilon=epsilon_custom_qcqp_l1, kernel=custom_kernel_qcqp_l1)
            model_custom_qcqp_l1.fit(X_train_fold_scaled, y_train_fold)
            y_pred_custom_qcqp_l1_fold = model_custom_qcqp_l1.predict(X_test_fold_scaled)
            all_y_pred_custom_qcqp_l1_folds.append(y_pred_custom_qcqp_l1_fold)
            all_mae_custom_qcqp_l1_folds.append(mean_absolute_error(y_test_fold, y_pred_custom_qcqp_l1_fold))

        # --- 6e. Custom Kernel QCQP L1 Trace SVR ---
        c_custom_qcqp_l1_trace = params_qcqp_trace.get("C")
        epsilon_custom_qcqp_l1_trace = params_qcqp_trace.get("epsilon")
        if None in [c_custom_qcqp_l1_trace, epsilon_custom_qcqp_l1_trace]:
            print("Error: Missing Custom Kernel QCQP L1 Trace SVR hyperparameters. Skipping for this fold.")
            all_y_pred_custom_qcqp_l1_trace_folds.append(np.full_like(y_test_fold, np.nan))
            all_mae_custom_qcqp_l1_trace_folds.append(np.nan)
        else:
            model_custom_qcqp_l1_trace = SVR(C=c_custom_qcqp_l1_trace, epsilon=epsilon_custom_qcqp_l1_trace, kernel=custom_kernel_qcqp_l1_trace)
            model_custom_qcqp_l1_trace.fit(X_train_fold_scaled, y_train_fold)
            y_pred_custom_qcqp_l1_trace_fold = model_custom_qcqp_l1_trace.predict(X_test_fold_scaled)
            all_y_pred_custom_qcqp_l1_trace_folds.append(y_pred_custom_qcqp_l1_trace_fold)
            all_mae_custom_qcqp_l1_trace_folds.append(mean_absolute_error(y_test_fold, y_pred_custom_qcqp_l1_trace_fold))
    
    if not all_X_test_orig_folds:
        print("Error: No splits were processed. Cannot generate plot or metrics.")
        return

    # --- 7. Concatenate results and Calculate Average MAE ---
    X_test_final_orig = np.concatenate(all_X_test_orig_folds)
    y_test_final = np.concatenate(all_y_test_folds)
    y_pred_svr_final = np.concatenate(all_y_pred_svr_folds)
    y_pred_socp_final = np.concatenate(all_y_pred_socp_folds)
    y_pred_qcqp_final = np.concatenate(all_y_pred_qcqp_folds)
    y_pred_xgb_final = np.concatenate(all_y_pred_xgb_folds)
    y_pred_custom_qcqp_l1_final = np.concatenate(all_y_pred_custom_qcqp_l1_folds)
    y_pred_custom_qcqp_l1_trace_final = np.concatenate(all_y_pred_custom_qcqp_l1_trace_folds)

    avg_mae_svr = np.nanmean(all_mae_svr_folds) # Use nanmean in case of skipped folds
    avg_mae_socp = np.nanmean(all_mae_socp_folds)
    avg_mae_qcqp = np.nanmean(all_mae_qcqp_folds)
    avg_mae_xgb = np.nanmean(all_mae_xgb_folds)
    avg_mae_custom_qcqp_l1 = np.nanmean(all_mae_custom_qcqp_l1_folds)
    avg_mae_custom_qcqp_l1_trace = np.nanmean(all_mae_custom_qcqp_l1_trace_folds)

    print(f"\\n--- Final Results ---")
    print(f"Sklearn SVR Average MAE: {avg_mae_svr:.4f}")
    print(f"SvrSocpMultiKernelL1MuExplicit Average MAE: {avg_mae_socp:.4f}")
    print(f"QCQP Trace Constrained Average MAE: {avg_mae_qcqp:.4f}")
    print(f"XGBoost Average MAE: {avg_mae_xgb:.4f}")
    print(f"Custom Kernel QCQP L1 SVR Average MAE: {avg_mae_custom_qcqp_l1:.4f}")
    print(f"Custom Kernel QCQP L1 Trace SVR Average MAE: {avg_mae_custom_qcqp_l1_trace:.4f}")
    
    num_forecast_points = len(y_test_final) # Actual number of points forecasted

    # --- 8. Plot Results ---
    plt.figure(figsize=(14, 7))
    time_index_plot = X_test_final_orig[:,0]

    plt.plot(time_index_plot, y_test_final, label='Actual Values', color='black', linestyle='-', marker='o', markersize=3)
    plt.plot(time_index_plot, y_pred_svr_final, label=f'Sklearn SVR (Avg MAE: {avg_mae_svr:.2f})', color='blue', alpha=0.8, linestyle='--')
    plt.plot(time_index_plot, y_pred_socp_final, label=f'QCQP (New) SVR (Avg MAE: {avg_mae_socp:.2f})', color='red', alpha=0.8, linestyle=':')
    plt.plot(time_index_plot, y_pred_qcqp_final, label=f'QCQP (Previous) SVR (Avg MAE: {avg_mae_qcqp:.2f})', color='green', alpha=0.8, linestyle='-.')
    plt.plot(time_index_plot, y_pred_xgb_final, label=f'XGBoost (Avg MAE: {avg_mae_xgb:.2f})', color='purple', alpha=0.8, linestyle='-')
    plt.plot(time_index_plot, y_pred_custom_qcqp_l1_final, label=f'Custom Kernel QCQP (New) SVR (Avg MAE: {avg_mae_custom_qcqp_l1:.2f})', color='orange', alpha=0.8, linestyle='--')
    plt.plot(time_index_plot, y_pred_custom_qcqp_l1_trace_final, label=f'Custom Kernel QCQP (Previous) SVR (Avg MAE: {avg_mae_custom_qcqp_l1_trace:.2f})', color='cyan', alpha=0.8, linestyle=':')

    plt.title(f'Sunspot Forecast Comparison (Rolling Forecast: {num_forecast_points} Samples in {n_splits} splits of {test_size_fold})', fontsize=18)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Monthly Mean Total Sunspot Number', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    # plt.ylim(-20, 240)
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
