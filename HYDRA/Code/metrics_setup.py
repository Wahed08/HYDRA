import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

datasets = list(test_data.keys())
metrics = ['Silhouette', 'CH', 'DB']
x_locs = np.arange(len(metrics))
bar_width = 0.2
colors = ['tab:blue', 'tab:orange', 'tab:green']

raw_scores = []
for name in datasets:
    sil = silhouette_score(test_data[name]['latent_z'], test_data[name]['clusters'])
    ch = calinski_harabasz_score(test_data[name]['latent_z'], test_data[name]['clusters']) / 1000
    db = davies_bouldin_score(test_data[name]['latent_z'], test_data[name]['clusters'])
    raw_scores.append([sil, ch, db])

raw_scores = np.array(raw_scores)  

scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(raw_scores)

normalized_scores += 0.05
normalized_scores = np.clip(normalized_scores, 0, 1)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (name, color) in enumerate(zip(datasets, colors)):
    offset = (i - 1) * bar_width
    bars = ax.bar(x_locs + offset, normalized_scores[i], width=bar_width, label=name, color=color)

    for j, bar in enumerate(bars):
        height = bar.get_height()
        actual_score = raw_scores[i][j]
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{actual_score:.2f}", ha='center', va='bottom', fontsize=9)

ax.set_ylabel("Normalized Metric Score (0–1)\n Actual Metric Values Printed Above Each Bar", fontsize=12)
ax.set_xticks(x_locs)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_title("Normalized Clustering Metrics Across Test Datasets (Android, Chrome, IMageMagick) for HYDRA", fontsize=14)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.legend(title="Datasets", fontsize=10)

plt.tight_layout()
plt.savefig("clustering_metrics_improved.png", dpi=300)
plt.show()
