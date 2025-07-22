import os
import json
import glob
from typing import List, Dict, Any, Optional

# If you don't have tabulate, install it: pip install tabulate
# Or, you can modify the print_results function to use basic string formatting.
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Define the base path to the results directories
# Get the directory where the script is located
BASE_RESULTS_PATH = os.path.dirname(os.path.abspath(__file__))

MODELS_CONFIG = [
    {"name": "mu", "path_suffix": "results_mu"},
    {"name": "trace", "path_suffix": "results_trace"}
]

def process_jsonl_file(file_path: str) -> Dict[str, List[Any]]:
    """Reads a single JSONL file and extracts relevant data."""
    targets = []
    comp_times = []
    non_optimal_fits = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    targets.append(data.get('target'))
                    comp_times.append(data.get('computation_time_seconds'))
                    non_optimal_fits += data.get('non_optimal_fits', 0)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line in {file_path}: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Missing key {e} in an entry in {file_path}")
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return {"targets": [], "comp_times": [], "non_optimal_fits": 0}
    
    # Filter out None values if keys were missing or data was malformed
    targets = [t for t in targets if t is not None]
    comp_times = [ct for ct in comp_times if ct is not None]
    
    return {
        "targets": targets,
        "comp_times": comp_times,
        "non_optimal_fits": non_optimal_fits
    }

def calculate_stats(targets: List[float], comp_times: List[float], total_non_optimal_fits: int) -> Dict[str, Optional[float]]:
    """Calculates aggregated statistics."""
    stats = {
        "min_target": None,
        "max_target": None,
        "mean_target": None,
        "total_non_optimal_fits": total_non_optimal_fits,
        "avg_comp_time": None
    }

    if targets:
        stats["min_target"] = min(targets)
        stats["max_target"] = max(targets)
        stats["mean_target"] = sum(targets) / len(targets)
    
    if comp_times:
        stats["avg_comp_time"] = sum(comp_times) / len(comp_times)
        
    return stats

def process_directory(directory_path: str) -> Dict[str, Optional[float]]:
    """Processes all JSONL files in a directory and aggregates statistics."""
    all_targets: List[float] = []
    all_comp_times: List[float] = []
    cumulative_non_optimal_fits: int = 0

    file_pattern = os.path.join(directory_path, '*.jsonl')
    jsonl_files = glob.glob(file_pattern)

    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {directory_path}")
        return calculate_stats([], [], 0)

    for file_path in jsonl_files:
        file_data = process_jsonl_file(file_path)
        all_targets.extend(file_data["targets"])
        all_comp_times.extend(file_data["comp_times"])
        cumulative_non_optimal_fits += file_data["non_optimal_fits"]
        
    return calculate_stats(all_targets, all_comp_times, cumulative_non_optimal_fits)

def print_results(results_data: List[List[Any]]):
    """Prints the results in a table."""
    headers = ["Model", "Min Target", "Max Target", "Mean Target", "Total Non-Optimal Fits", "Avg Comp Time (s)"]
    
    if TABULATE_AVAILABLE:
        print(tabulate(results_data, headers=headers, tablefmt="grid", floatfmt=".4f"))
    else:
        print("Tabulate library not found. Printing basic table:")
        header_fmt = "{:<10} | {:<12} | {:<12} | {:<12} | {:<22} | {:<18}"
        row_fmt = "{:<10} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<22} | {:<18.4f}"
        none_fmt = "{:<10} | {:<12} | {:<12} | {:<12} | {:<22} | {:<18}"

        print(header_fmt.format(*headers))
        print("-" * (10 + 15*2 + 15 + 25 + 21)) # Adjust separator length
        for row in results_data:
            # Handle None values for printing
            formatted_row = []
            for i, item in enumerate(row):
                if item is None:
                    formatted_row.append("N/A")
                else:
                    formatted_row.append(item)
            
            try:
                # Check if all numeric fields are indeed numeric or N/A
                if all(isinstance(x, (int, float)) or x == "N/A" for x in formatted_row[1:]):
                     print(row_fmt.format(*formatted_row) if not any(x == "N/A" for x in formatted_row[1:]) else \
                           none_fmt.format(*formatted_row)) # crude fallback for N/A
                else:
                    print(none_fmt.format(*formatted_row))

            except TypeError: # Fallback for mixed types if formatting fails
                 print(none_fmt.format(*formatted_row))


def main():
    """Main function to orchestrate the analysis."""
    all_model_stats = []

    for model_config in MODELS_CONFIG:
        model_name = model_config["name"]
        model_dir_path = os.path.join(BASE_RESULTS_PATH, model_config["path_suffix"])
        
        print(f"\nProcessing model: {model_name} from {model_dir_path}...")
        
        if not os.path.isdir(model_dir_path):
            print(f"Warning: Directory not found: {model_dir_path}")
            stats = calculate_stats([], [], 0) # Empty stats
        else:
            stats = process_directory(model_dir_path)
        
        all_model_stats.append([
            model_name,
            stats["min_target"],
            stats["max_target"],
            stats["mean_target"],
            stats["total_non_optimal_fits"],
            stats["avg_comp_time"]
        ])

    print("\n--- Aggregated Results ---")
    print_results(all_model_stats)

if __name__ == "__main__":
    main()
