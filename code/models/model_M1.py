import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from itertools import product
import os
from collections import Counter
from scipy.spatial import ConvexHull, QhullError  
from sklearn.preprocessing import MinMaxScaler
np.random.seed(42) 

def detect_missing_malloc_check(code: str) -> int:
    alloc_pattern = r'\b(\w+)\s*\*\s*(\w+)\s*=\s*\([^\)]*\)\s*(malloc|calloc|realloc)\s*\([^;]+?\);|\b(\w+)\s*\*\s*(\w+)\s*=\s*(malloc|calloc|realloc)\s*\([^;]+?\);'
    matches = re.findall(alloc_pattern, code)
    ptrs = [match[1] if match[1] else match[4] for match in matches if match[1] or match[4]]
    for ptr in ptrs:
        is_checked = False
        is_used = False
        null_check_patterns = [rf'if\s*\(\s*{ptr}\s*\)', rf'if\s*\(\s*{ptr}\s*!=\s*NULL\s*\)', rf'if\s*\(\s*!{ptr}\s*\)', rf'if\s*\(\s*NULL\s*!=\s*{ptr}\s*\)']
        usage_patterns = [rf'{ptr}\s*\[', rf'\*\s*{ptr}', rf'{ptr}\s*->', rf'{ptr}\s*\(', rf'{ptr}\s*=']
        for pat in null_check_patterns:
            if re.search(pat, code):
                is_checked = True
                break
        for pat in usage_patterns:
            if re.search(pat, code):
                is_used = True
                break
        if is_used and not is_checked:
            return 1
    return 0

def missing_null_check_func(code):
    functions = re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*\{[^{}]*\}', code, re.DOTALL)
    for func in functions:
        ptr_decl = re.findall(r'\b\w+\s*\*\s*(\w+)\s*=\s*[^;]+;', func)
        ptr_func_decl = re.findall(r'\b\w+\s*\*\s*(\w+)\s*;', func)
        ptr_vars = list(set(ptr_decl + ptr_func_decl))
        for ptr in ptr_vars:
            unsafe_use = re.search(rf'[^a-zA-Z0-9_]({ptr})(->|\[\d*]|[^\w]?\*)|\b{ptr}\s*\(', func)
            if not unsafe_use:
                continue
            null_check = re.search(rf'if\s*\(\s*(!\s*{ptr}|{ptr}\s*==\s*NULL|{ptr}\s*!=\s*NULL)\s*\)', func)
            if not null_check:
                return 1
    return 0

def detect_race_condition(code: str) -> bool:
    field_assignment_pattern = r'\b\w+\s*(->|\.)\s*\w+\s*=.*?;'
    control_block_pattern = r'\b(if|while|for|switch)\s*\([^\)]+\)\s*{[^}]*' + field_assignment_pattern + r'[^}]*}'
    matches_control_blocks = re.findall(control_block_pattern, code, re.DOTALL)
    locking_primitive_pattern = r'\b(mutex_lock|pthread_mutex_lock|spin_lock)\b'
    unlocking_primitive_pattern = r'\b(mutex_unlock|pthread_mutex_unlock|spin_unlock)\b'
    has_locking = re.search(locking_primitive_pattern, code)
    has_unlocking = re.search(unlocking_primitive_pattern, code)
    if matches_control_blocks and not (has_locking and has_unlocking):
        return True
    return False

def logging_but_no_blocking(code):
    log_lines = re.findall(r'(syslog\([^)]*\)|printk\([^)]*\))', code)
    for line in log_lines:
        log_index = code.find(line)
        snippet = code[log_index:log_index+150]
        if not re.search(r'(return|exit|break|continue)', snippet):
            return 1
    return 0

def split_into_functions(code):
    functions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_\*\s]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*\{.*?\})', code, re.DOTALL)
    return functions

def missing_bounds_check(code):
    functions = split_into_functions(code)
    risky_keywords = ['recv', 'read', 'strcpy', 'memcpy', 'gets', 'strcat', 'write']
    for func in functions:
        found_risky = any(kw in func for kw in risky_keywords)
        has_check = any('if' in line and ('<' in line or '>' in line or '<=' in line or '>=' in line) for line in func.splitlines())
        if found_risky and not has_check:
            return 1
    return 0

def analyze_risks(code):
    risk_flags = {
        "Missing Null Check": missing_null_check_func(code),
        "Race Condition": detect_race_condition(code),
        "Missing Bounds Check": missing_bounds_check(code),
        "Unsafe Memory Allocation": detect_missing_malloc_check(code),
        "Logging Without Halting": logging_but_no_blocking(code),
        "issue_detected": 0
    }
    if any(v != 0 for v in risk_flags.values() if v is not False):
        risk_flags["issue_detected"] = 1
    return risk_flags

train_path = '/home/phoenix/Desktop/TA/Re_ New Code/Linux-Data-Updated.csv'
test_paths = {
    'Android': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/Android-Data.csv',
    'Chrome': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/Chrome-Data.csv',
    'ImageMagick': '/home/phoenix/Desktop/TA/Re_ New Code/CSV Data of three testing projects/ImageMagick-Data.csv'
}

def load_or_generate_heuristics(df, path, filename):
    heur_file = f"{filename}_heuristics.npy"
    commit_file = f"{filename}_commit_ids.npy"

    if os.path.exists(heur_file) and os.path.exists(commit_file):
        heuristics = np.load(heur_file)
        commit_ids = np.load(commit_file)
    else:
        heuristics = []
        commit_ids = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing heuristics ({filename})"):
            code = row['PCode']
            risk_flags = analyze_risks(code)
            heuristics.append([risk_flags[h] for h in [
                "Missing Null Check", "Race Condition", "Missing Bounds Check",
                "Unsafe Memory Allocation", "Logging Without Halting", "issue_detected"
            ]])
            commit_ids.append(row['Commit_Id'])
        heuristics = np.array(heuristics)
        np.save(heur_file, heuristics)
        np.save(commit_file, commit_ids)
    return heuristics, commit_ids

df_train = pd.read_csv(train_path)
df_train['PCode'] = df_train['PCode'].astype(str)
heuristics_train, commit_ids_train = load_or_generate_heuristics(df_train, train_path, "train")

train_data, val_data = train_test_split(heuristics_train, test_size=0.2, random_state=42)

param_grid = {
    'n_clusters': range(2, 7), 
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
    heuristics_test, commit_ids_test = load_or_generate_heuristics(df_test, path, name)
    test_data[name] = {
        'features': heuristics_test,
        'heuristics': heuristics_test,
        'commit_ids': np.array(commit_ids_test),
        'codes': df_test['PCode'].values
    }

heuristic_names = [
    "Missing Null Check", "Race Condition", "Missing Bounds Check",
    "Unsafe Memory Allocation", "Logging Without Halting", "None"
]
for name in test_data:
    features = test_data[name]['features']
    heuristics = test_data[name]['heuristics']
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
        'Heuristic_Label': [heuristic_names[np.argmax(h) if np.max(h) > 0 else -1] for h in heuristics],
        'Code': codes
    })
    df_out.to_csv(f"{name}_cluster_results.csv", index=False)

    print(f"\nHeuristic distribution by cluster in {name}:")
    cluster_heuristic_counts = []
    for c in range(best_params['n_clusters']):
        idx = np.where(clusters == c)[0]
        hsum = np.sum(heuristics[idx], axis=0)
        hsum[-1] = np.sum(heuristics[idx, -1] == 0)  
        cluster_heuristic_counts.append(hsum)
        top = np.argmax(hsum[:-1])  
        dominant_heuristic = heuristic_names[top]
        dominant_count = hsum[top]
        print(f"\nCluster {c}:")
        print(f"Dominant Heuristic = {dominant_heuristic}, Count = {dominant_count}")
        for h_idx, h_name in enumerate(heuristic_names):
            print(f"{h_name}: {hsum[h_idx]}")

    print(f"\n{name} Dataset - False Positives for 'None' Heuristic:")
    false_positives = []
    for idx in range(len(heuristics)):
        if heuristics[idx, -1] == 1:  
            if np.any(heuristics[idx, :-1] == 1):
                false_positive_heuristics = [h for h_idx, h in enumerate(heuristic_names[:-1]) if heuristics[idx, h_idx] == 1]
                false_positives.append((idx, false_positive_heuristics))

    if false_positives:
        print(f"Found {len(false_positives)} false positive(s) where 'None' was incorrectly assigned:")
        for idx, conflicting_heuristics in false_positives:
            print(f"\nCommit ID: {commit_ids[idx]}")
            print(f"Conflicting Heuristics: {', '.join(conflicting_heuristics)}")
            print(f"Code Snippet: {codes[idx][:200]}...\n")
    else:
        print("No false positives found for 'None' heuristic.")

    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_coords, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster'] = clusters
    tsne_df['Heuristic'] = [heuristic_names[np.argmax(h) if np.max(h) > 0 else -1] for h in heuristics]

    plt.figure(figsize=(14, 8))
    scatter = sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Heuristic', style='Cluster',
                              palette={
                                  'Missing Null Check': 'green',
                                  'Race Condition': 'red',
                                  'Missing Bounds Check': 'blue',
                                  'Unsafe Memory Allocation': 'purple',
                                  'Logging Without Halting': 'orange',
                                  'None': 'gray'
                              }, marker='o', s=100)
    plt.title(f"t-SNE of {name} - Heuristics + Tuned K-means")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title=' ', bbox_to_anchor=(1.05, 1), loc='upper left')

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
                    print(f"Warning: ConvexHull failed for Cluster {cluster} in {name} due to coplanarity or low dimensionality: {e}")
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
ax.set_title("Normalized Clustering Metrics Across Test Datasets (Android, Chrome, ImageMagick) for Heuristics + Tuned K-means", fontsize=14)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.legend(title="Datasets", fontsize=10, bbox_to_anchor=(1.15, 1), loc='upper left')  # Legend on right

plt.tight_layout()
plt.savefig("clustering_metrics_improved.png", dpi=300, bbox_inches='tight')
plt.show()