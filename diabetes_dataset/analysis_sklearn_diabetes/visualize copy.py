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
average = data.tail(5).to_dict(orient="records")
with jsonlines.open("average_sklearn.jsonl", mode='w') as writer:
    writer.write_all(average)
  

# %% Visualize
test_size = 12
datar = (
    data
    # .query(f"test_size == {test_size}")
    .query("target <= 55")
    # .query(" C <= 5")
    # .query("tau <= 1")
    # .query("epsilon <= 200")
    .query("gamma <= 100")
)

# %%
plt.scatter(
    datar["C"], datar.target,
    label="C", c=datar.target,
    cmap="viridis", alpha=0.5
)
plt.title(f"C (test size {test_size})")
plt.xlabel("C")
plt.show()

plt.scatter(
    datar["epsilon"], datar.target,
    label="epsilon", c=datar.target,
    cmap="viridis", alpha=0.2
)
plt.title(f"epsilon (test size {test_size})")
plt.xlabel("epsilon")
plt.show()

plt.scatter(
    datar["gamma"], datar.target,
    label="gamma", c=datar.target,
    cmap="viridis", alpha=0.2
)
plt.title(f"gamma (test size {test_size})")
plt.xlabel("gamma")
plt.show()

# %% 3d Scatter plot
fig = px.scatter_3d(
    datar,
    x='C',
    y='epsilon',
    z='target',
    color='target',
    labels={'C': 'C', 'epsilon': 'epsilon', 'gamma': 'gamma', 'target': 'target'},
    title='3D Scatter Plot',
    color_continuous_scale='viridis',
    hover_data={'epsilon': True, 'C': True}  # Include epsilon in hover data
)
fig.show()


# %%
