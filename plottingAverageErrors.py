import numpy as np
import matplotlib.pyplot as plt

# Load data
headers = "filename,RMSE mf,RMSE nisf,RMSE sf,RMSE sfTP,T,WRMSE mf,WRMSE nisf,WRMSE sf,WRMSE sfTP,fieldNum,velVariance".split(",")
d = np.genfromtxt("Data/TrajectoriesAndEstimates/GPResults/results.csv", 
                  delimiter=",", names=True, dtype=None, encoding="utf-8")

# Define column groups
rmse_cols = d.dtype.names[1:4]   # RMSE mf, RMSE nisf, RMSE sf, RMSE sfTP
wrmse_cols = d.dtype.names[6:9] # WRMSE mf, WRMSE nisf, WRMSE sf, WRMSE sfTP

# Define noise levels
noise_levels = []#[0, 0.1, 0.2]

# Compute means per noise level
results = {lvl: {} for lvl in noise_levels}
for lvl in noise_levels:
    mask = d[d.dtype.names[11]] == lvl
    for col in rmse_cols + wrmse_cols:
        results[lvl][col] = np.mean(d[col][mask])

# Compute overall means (all data, ignoring velVariance)
overall_means = {}
for col in rmse_cols + wrmse_cols:
    overall_means[col] = np.mean(d[col])

# --- Plotting function for grouped bar plots ---
def plot_grouped_bars(columns, title):
    x = np.arange(len(columns))  # positions for metrics
    width = 0.2  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot per noise level
    for i, lvl in enumerate(noise_levels):
        values = [results[lvl][col] for col in columns]
        ax.bar(x + i*width, values, width, label=f"velVariance={lvl}")

    # Plot overall mean as last group
    overall_values = [overall_means[col] for col in columns]
    ax.bar(x + len(noise_levels)*width, overall_values, width, 
           label="Mean WMSE", color="black", alpha=0.7)

    ax.set_xticks(x + width*(len(noise_levels)/2))
    ax.set_xticklabels(columns)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

# Plot RMSEs
plot_grouped_bars(rmse_cols, "RMSE Averages by Noise Level and Overall")

# Plot WRMSEs
plot_grouped_bars(wrmse_cols, "WMSE Averages by Noise Level and Overall")
plot_grouped_bars(wrmse_cols, "WMSE Averages")

# --- Print overall means in console ---
print("\nOverall Means (All Data):")
for col, val in overall_means.items():
    print(f"{col}: {val:.4f}")
