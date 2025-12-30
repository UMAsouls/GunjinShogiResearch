import pandas as pd
import matplotlib.pyplot as plt

LOSS_DIR = "model_loss"
METHOD_NAME = "deepnash_mp"
NAME = "mini_cnn_v12"

PATH = f"{LOSS_DIR}/{METHOD_NAME}/{NAME}"

# Set display options

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv(f'{PATH}/loss.csv')

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot Losses
axes[0].plot(df['loss'], label='Total Loss')
axes[0].plot(df['value_loss'], label='Value Loss')
axes[0].plot(df['policy_loss'], label='Policy Loss')
axes[0].set_title('Losses over Steps')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot Entropies
axes[1].plot(df['entropy'], label='Current Entropy')
axes[1].plot(df['target_entropy'], label='Target Entropy')
axes[1].set_title('Entropy over Steps')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Entropy')
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{PATH}/loss_entropy_plot.png')