# %%
import os
import jsonlines
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import plotly.express as px

# %% Define path
def load_and_process_data(folder_name):
    """
    Loads and concatenates all jsonl files in a folder, then preprocesses the data.

    Args:
        folder_name (str): The name of the folder containing the jsonl files.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data.
    """
    jsonl_files = [f for f in os.listdir(folder_name) if f.endswith('.jsonl')]
    data_frames = []

    for file in jsonl_files:
        with jsonlines.open(os.path.join(folder_name, file)) as reader:
            data_frames.append(pd.DataFrame(reader))

    data = pd.concat(data_frames, ignore_index=True)

    params = pd.json_normalize(data['params'])

    data = (
        pd.concat(
            [
                data[["target"]],
                params,
            ], axis=1)
        .sort_values(by="target", ascending=False)
    )
    
    return data


data = load_and_process_data("results")


# %% min by group
# average = data.groupby("test_size").apply(lambda x: x.nsmallest(4, "target").iloc[0].to_dict()).to_dict()
# average = [{"test_size": k, **v} for k, v in average.items()]
average = data.tail(1).to_dict(orient="records")
with jsonlines.open("average_xgboost.jsonl", mode='w') as writer:
    writer.write_all(average)
  

# %% Visualize
test_size = 12
datar = (
    data
    # .query(f"test_size == {test_size}")
    .query("target <= 23.0")
    # .query(" C <= 5")
    # .query("tau <= 1")
    # .query("epsilon <= 200")
    # .query("gamma <= 100")
)

# %%
plt.scatter(
    datar["n_estimators"], datar.target,
    label="n_estimators", c=datar.target,
    cmap="viridis", alpha=0.5
)
plt.title(f"C (test size {test_size})")
plt.xlabel("n_estimators")
plt.show()

plt.scatter(
    datar["max_depth"], datar.target,
    label="max_depth", c=datar.target,
    cmap="viridis", alpha=0.2
)
plt.title(f"max_depth (test size {test_size})")
plt.xlabel("max_depth")
plt.show()

plt.scatter(
    datar["learning_rate"], datar.target,
    label="learning_rate", c=datar.target,
    cmap="viridis", alpha=0.2
)
plt.title(f"learning_rate (test size {test_size})")
plt.xlabel("learning_rate")
plt.show()

# %% 3d Scatter plot
fig = px.scatter_3d(
    datar,
    x='max_depth',
    y='n_estimators',
    z='target',
    color='target',
    labels={'max_depth': 'max_depth', 'learning_rate': 'learning_rate', 'n_estimators': 'n_estimators', 'target': 'target'},
    title='3D Scatter Plot',
    color_continuous_scale='viridis',
    hover_data={'learning_rate': True, 'max_depth': True}  # Include epsilon in hover data
)
fig.show()

# %%
