import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOSS_DIR = "model_loss"
METHOD_NAME = "deepnash_mp"
NAME = "mini_cnn_t_v11"

PATH = f"{LOSS_DIR}/{METHOD_NAME}/{NAME}"

WINDOW_SIZE = 1000

# Set display options

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv(f'{PATH}/loss.csv')

# Calculate the mean of the loss for each window
loss_means = df['loss'].groupby(np.arange(len(df)) // WINDOW_SIZE).mean()

# Create a figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot Losses
ax.plot(loss_means.index, loss_means, label='ウィンドウ毎の平均ロス')
ax.set_title('ステップ毎の平均ロス')
ax.set_xlabel(f'ステップ (x{WINDOW_SIZE})')
ax.set_ylabel('平均ロス')
ax.legend()
ax.grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{PATH}/area_ave_loss_plot.png')