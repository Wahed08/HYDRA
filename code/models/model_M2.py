import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import os
from collections import Counter
from scipy.spatial import ConvexHull

np.random.seed(42)  
torch.manual_seed(42)  

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

def get_code_features(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

train_path = '/home/phoenix/Desktop/TA/Re_ New Code/Linux-Data-Updated.csv'
test_paths = {
    'Android': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/Android-Data.csv',
    'Chrome': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/Chrome-Data.csv',
    'ImageMagick': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/ImageMagick-Data.csv'
}

def load_or_generate_embeddings(df, path, filename):
    embed_file = f"{filename}_embeddings.npy"
    commit_file = f"{filename}_commit_ids.npy"

    if os.path.exists(embed_file) and os.path.exists(commit_file):
        embeddings = np.load(embed_file)
        commit_ids = np.load(commit_file)
    else:
        embeddings = []
        commit_ids = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing GraphCodeBERT embeddings ({filename})"):
            code = row['PCode']
            embedding = get_code_features(code)
            embeddings.append(embedding)
            commit_ids.append(row['Commit_Id'])
        embeddings = np.array(embeddings)
        np.save(embed_file, embeddings)
        np.save(commit_file, commit_ids)
    return embeddings, commit_ids

df_train = pd.read_csv(train_path)
df_train['PCode'] = df_train['PCode'].astype(str)
embeddings_train, commit_ids_train = load_or_generate_embeddings(df_train, train_path, "train")

train_data, val_data = train_test_split(embeddings_train, test_size=0.2, random_state=42)

param_grid = {
    'n_clusters': range(2, 11), 
    'n_init': [10, 20, 50],    
    'tol': [1e-4, 1e-5]         
}
results = []
for params in tqdm(product(*param_grid.values()), desc="Tuning K-means"):
    n_clusters, n_init, tol = params
    clustering = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, tol=tol, random_state=42)
    train_clusters = clustering.fit_predict(train_data)
    if len(np.unique(train_clusters)) > 1:
        silhouette = silhouette_score(train_data, train_clusters)
        ch_score = calinski_harabasz_score(train_data, train_clusters)
        db_score = davies_bouldin_score(train_data, train_clusters)
        results.append({
            'n_clusters': n_clusters,
            'n_init': n_init,
            'tol': tol,
            'silhouette': silhouette,
            'ch_score': ch_score,
            'db_score': db_score
        })
        print(f"n_clusters={n_clusters}, n_init={n_init}, tol={tol}, Silhouette: {silhouette:.4f}, CH: {ch_score:.2f}, DB: {db_score:.4f}")

best_result = max(results, key=lambda x: x['silhouette'])
print(f"Best Configuration: n_clusters={best_result['n_clusters']}, n_init={best_result['n_init']}, tol={best_result['tol']}, "
      f"Silhouette: {best_result['silhouette']:.4f}, CH: {best_result['ch_score']:.2f}, DB: {best_result['db_score']:.4f}")
best_params = {
    'n_clusters': best_result['n_clusters'],
    'n_init': best_result['n_init'],
    'tol': best_result['tol']
}

test_data = {}
for name, path in test_paths.items():
    df_test = pd.read_csv(path)
    df_test['PCode'] = df_test['PCode'].astype(str)
    embeddings_test, commit_ids_test = load_or_generate_embeddings(df_test, path, name)
    test_data[name] = {
        'features': embeddings_test,
        'commit_ids': np.array(commit_ids_test),
        'codes': df_test['PCode'].values
    }

for name in test_data:
    features = test_data[name]['features']
    commit_ids = test_data[name]['commit_ids']
    codes = test_data[name]['codes']

    clustering = KMeans(n_clusters=best_params['n_clusters'], init='k-means++', n_init=best_params['n_init'],
                        tol=best_params['tol'], random_state=42)
    clusters = clustering.fit_predict(features)

    test_data[name].update({
        'clusters': clusters
    })

    silhouette = silhouette_score(features, clusters)
    ch_score = calinski_harabasz_score(features, clusters)
    db_score = davies_bouldin_score(features, clusters)
    print(f"\n[{name}] Clustering Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}, CH: {ch_score:.2f}, DB: {db_score:.4f}")

    print(f"Top 5 Largest Clusters in {name}:")
    unique, counts = np.unique(clusters, return_counts=True)
    top_clusters = np.argsort(-counts)[:5]
    for cluster_idx in top_clusters:
        cluster_size = counts[cluster_idx]
        cluster_indices = np.where(clusters == unique[cluster_idx])[0]
        sample_commit = commit_ids[cluster_indices[0]] if len(cluster_indices) > 0 else "N/A"
        print(f"Cluster {unique[cluster_idx]}: Size = {cluster_size}, Sample Commit ID = {sample_commit}")

    df_out = pd.DataFrame({
        'Commit_ID': commit_ids,
        'Cluster': clusters,
        'Code': codes
    })
    df_out.to_csv(f"{name}_cluster_results.csv", index=False)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_coords, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster'] = clusters

    plt.figure(figsize=(14, 8))
    scatter = sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Cluster', style='Cluster',
                              palette='tab10', marker='o', s=100)
    plt.title(f"t-SNE of {name} - GraphCodeBERT + Tuned K-means")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    for cluster in range(best_params['n_clusters']):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 2:
            cluster_points = tsne_coords[cluster_indices]
            if np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0)) > 1e-10:
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', alpha=0.5)
                except QhullError as e:
                    print(f"Warning: ConvexHull failed for Cluster {cluster} in {name}: {e}")
            else:
                print(f"Warning: Insufficient variance in Cluster {cluster} in {name}, skipping hull.")

    for cluster in range(best_params['n_clusters']):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_points = tsne_coords[cluster_indices]
            cluster_center = cluster_points.mean(axis=0)
            x_range = tsne_coords[:, 0].max() - tsne_coords[:, 0].min()
            y_range = tsne_coords[:, 1].max() - tsne_coords[:, 1].min()
            offset = 0.1 * max(x_range, y_range)
            outward_x = cluster_center[0] + offset * np.sign(x_range)
            outward_y = cluster_center[1] + offset * np.sign(y_range)
            plt.text(outward_x, outward_y, f'Cluster {cluster}',
                     fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{name}_tsne_plot.png")
    plt.show()

datasets = list(test_data.keys())
metrics = ['Silhouette', 'CH', 'DB']
x_locs = np.arange(len(metrics))
bar_width = 0.2
colors = ['tab:blue', 'tab:orange', 'tab:green']

raw_scores = []
for name in datasets:
    sil = silhouette_score(test_data[name]['features'], test_data[name]['clusters'])
    ch = calinski_harabasz_score(test_data[name]['features'], test_data[name]['clusters']) / 1000
    db = davies_bouldin_score(test_data[name]['features'], test_data[name]['clusters'])
    raw_scores.append([sil, ch, db])

raw_scores = np.array(raw_scores)

scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(raw_scores)

normalized_scores += 0.05
normalized_scores = np.clip(normalized_scores, 0, 1)

fig, ax = plt.subplots(figsize=(12, 6))  

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
ax.set_title("Normalized Clustering Metrics Across Test Datasets (Android, Chrome, ImageMagick) for GraphCodeBERT + Tuned K-means", fontsize=14)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.legend(title="Datasets", fontsize=10, bbox_to_anchor=(1.15, 1), loc='upper left') 

plt.tight_layout()
plt.savefig("clustering_metrics_improved.png", dpi=300, bbox_inches='tight')
plt.show()