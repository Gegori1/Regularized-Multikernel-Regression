# Multi-Kernel Support Vector Regression with Regularized Kernel Weights

This repository contains the implementation and analysis of various Multi-Kernel Support Vector Regression (MSVR) models for the paper `Multikernel Support Vector Regression with Regularized Kernel Weights`. It explores different optimization formulations, including QCQP (Quadratically Constrained Quadratic Program) with L1 and L2 regularization, and applies them to the diabetes and sunspot datasets.

## Project Structure

The repository is organized as follows:

-   **Root Directory**: Contains the core Python scripts for the SVR model implementations (e.g., `svr_qcqp_multi_kernel_l1.py`, `svr_qcqp_multi_kernel_l2.py`).
-   `diabetes.csv` / `Sunspots.csv`: The raw datasets used for the experiments.
-   `kernel_builder.py`: A script for constructing kernel matrices.
-   `diabetes_dataset/`: Contains all scripts and results related to the diabetes dataset.
    -   `analysis_*`: Subdirectories for different model analyses (e.g., `analysis_qcqp_l1_diabetes`), each containing scripts for optimization, analysis, and visualization.
-   `sunspot_dataset/`: Contains all scripts and results related to the sunspot dataset.
    -   `analysis_*`: Subdirectories similar to the diabetes dataset for model-specific experiments.

## Installation

This project requires Python and several dependencies. You can install the necessary packages using pip. The analysis scripts in particular require `tabulate` for displaying results.

```sh
pip install pandas jsonlines tabulate numpy scikit-learn
```

You may also need an optimization library like `cvxpy` depending on the model implementation.

## How to Use

1.  **Run Experiments**: Navigate to one of the analysis directories, for example, `diabetes_dataset/analysis_qcqp_l2_diabetes_trace/`. You can run the optimization script to perform hyperparameter tuning.
    ```sh
    python diabetes_dataset/analysis_qcqp_l2_diabetes_trace/optimization_diabetes_scaled.py
    ```
    This will generate `.jsonl` log files containing the results of each trial.

2.  **Analyze Results**: Use the provided analysis and visualization scripts to process the output logs. For example, the script in `sunspot_dataset/comparative_analysis_sunspot/analyze_model_results.py` can be used to aggregate and compare results from different models.
    ```sh
    python sunspot_dataset/comparative_analysis_sunspot/analyze_model_results.py
    ```
// filepath: README.md
# Multi-Kernel Support Vector Regression

This repository contains the implementation and analysis of various Multi-Kernel Support Vector Regression (SVR) models. It explores different optimization formulations, including QCQP (Quadratically Constrained Quadratic Program) with L1 and L2 regularization, and applies them to the diabetes and sunspot datasets.

## Project Structure

The repository is organized as follows:

-   **Root Directory**: Contains the core Python scripts for the SVR model implementations (e.g., `svr_qcqp_multi_kernel_l1.py`, `svr_qcqp_multi_kernel_l2.py`).
-   `diabetes.csv` / `Sunspots.csv`: The raw datasets used for the experiments.
-   `kernel_builder.py`: A script for constructing kernel matrices.
-   `diabetes_dataset/`: Contains all scripts and results related to the diabetes dataset.
    -   `analysis_*`: Subdirectories for different model analyses (e.g., `analysis_qcqp_l1_diabetes`), each containing scripts for optimization, analysis, and visualization.
-   `sunspot_dataset/`: Contains all scripts and results related to the sunspot dataset.
    -   `analysis_*`: Subdirectories similar to the diabetes dataset for model-specific experiments.

## Installation

This project requires Python and several dependencies. You can install the necessary packages using pip. The analysis scripts in particular require `tabulate` for displaying results.

```sh
pip install pandas jsonlines tabulate numpy scikit-learn
```

In case you need to modify some predictions, you can install the hyperparameter optimization library

```sh
pip3 install -e git+https://github.com/Gegori1/gcloud_library#egg=gcp_library --config-settings editable_mode=strict
```

You may also need an optimization library like `cvxpy` depending on the model implementation.

## How to Use

1.  **Run Experiments**: Navigate to one of the analysis directories, for example, `diabetes_dataset/analysis_qcqp_l2_diabetes_trace/`. You can run the optimization script to perform hyperparameter tuning.
    ```sh
    python diabetes_dataset/analysis_qcqp_l2_diabetes_trace/optimization_diabetes_scaled.py
    ```
    This will generate `.jsonl` log files containing the results of each trial.

2.  **Analyze Results**: Use the provided analysis and visualization scripts to process the output logs. For example, the script in `sunspot_dataset/comparative_analysis_sunspot/analyze_model_results.py` can be used to aggregate and compare results from