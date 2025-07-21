vae = VAE(input_dim, output_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
criterion = loss_function
vae.train()
best_val_loss = float('inf')
patience = 20
trigger_times = 0
train_losses = []
val_losses = []
for epoch in range(200):
    vae.train()
    total_loss = 0
    total_recon_loss = 0
    for i in range(0, len(train_data), 128):
        batch = torch.FloatTensor(train_data[i:i+128]).to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(batch)
        recon_loss = criterion(recon_batch, batch, mu, log_var)
        recon_loss.backward()
        optimizer.step()
        total_loss += recon_loss.item()
        total_recon_loss += recon_loss.item()
    
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_data), 128):
            val_batch = torch.FloatTensor(val_data[i:i+128]).to(device)
            recon_val_batch, _, _ = vae(val_batch)
            val_loss += criterion(recon_val_batch, val_batch, mu, log_var).item()
    avg_val_loss = val_loss / (len(val_data) / 128)
    avg_train_loss = total_loss / (len(train_data) / 128)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/200], Train Recon Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
  
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
    else:
        trigger_times += 1
    if trigger_times >= patience:
        print(f"Early stopping at Epoch {epoch+1}")
        break
vae.eval()
with torch.no_grad():
    _, mu, log_var = vae(torch.FloatTensor(train_data).to(device))
    z_train = vae.reparameterize(mu, log_var)
z_train = z_train.cpu().numpy()


test_data = {}
for name, path in test_paths.items():
    df_test = pd.read_csv(path)
    df_test['PCode'] = df_test['PCode'].astype(str)
    embeddings_test, heuristics_test, commit_ids_test = load_or_generate_embeddings(df_test, path, name)
    features_test = np.concatenate([embeddings_test, heuristics_test], axis=1)
    test_data[name] = {
        'features': features_test,
        'heuristics': heuristics_test,
        'commit_ids': np.array(commit_ids_test),
        'codes': df_test['PCode'].values
    }


def compute_reconstruction_error(model, data, device, batch_size=128):
    model.eval()
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i:i+batch_size]).to(device)
            recon_batch, _, _ = model(batch)
            error = F.mse_loss(recon_batch, batch, reduction='none').mean(dim=1)
            recon_errors.append(error.cpu().numpy())
    return np.concatenate(recon_errors)

heuristic_names = [
    "Missing Null Check", "Race Condition", "Missing Bounds Check",
    "Unsafe Memory Allocation", "Logging Without Halting", "None"
]
for name in test_data:
    features = test_data[name]['features']
    heuristics = test_data[name]['heuristics']
    commit_ids = test_data[name]['commit_ids']
    codes = test_data[name]['codes']

    with torch.no_grad():
        _, mu, log_var = vae(torch.FloatTensor(features).to(device))
        z_test = vae.reparameterize(mu, log_var)
    z_test = z_test.cpu().numpy()

    clustering = AgglomerativeClustering(n_clusters=best_params['n_clusters'], linkage='ward')
    clusters = clustering.fit_predict(z_test)

    recon_error = compute_reconstruction_error(vae, features, device)

    
    test_data[name].update({
        'latent_z': z_test,
        'clusters': clusters,
        'recon_error': recon_error
    })

    
    silhouette = silhouette_score(z_test, clusters)
    ch_score = calinski_harabasz_score(z_test, clusters)
    db_score = davies_bouldin_score(z_test, clusters)
    print(f"\n[{name}] Clustering Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}, CH: {ch_score:.2f}, DB: {db_score:.4f}")

    
    print(f"Top 5 Anomalous Functions in {name}:")
    top_idx = np.argsort(-recon_error)[:5]
    for idx in top_idx:
        print(f"Commit ID: {commit_ids[idx]}, Error: {recon_error[idx]:.4f}\n{codes[idx][:200]}...\n")

    
    df_out = pd.DataFrame({
        'Commit_ID': commit_ids,
        'Cluster': clusters,
        'Recon_Error': recon_error,
        'Heuristic_Label': [heuristic_names[np.argmax(h) if np.max(h) > 0 else -1] for h in heuristics],
        'Code': codes
    })
    df_out.to_csv(f"{name}_cluster_results.csv", index=False)

    
    print(f"Heuristic distribution by cluster in {name}:")
    cluster_heuristic_counts = []
    for c in range(best_params['n_clusters']):
        idx = np.where(clusters == c)[0]
        hsum = np.sum(heuristics[idx], axis=0)
        hsum[-1] = np.sum(heuristics[idx, -1] == 0)  
        cluster_heuristic_counts.append(hsum)
        top = np.argmax(hsum[:-1]) 
        print(f"Cluster {c}: Dominant Heuristic = {heuristic_names[top]}, Count = {hsum[top]}")

    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(z_test)
    tsne_df = pd.DataFrame(tsne_coords, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster'] = clusters
    tsne_df['Heuristic'] = [heuristic_names[np.argmax(h) if np.max(h) > 0 else -1] for h in heuristics]
    tsne_df['ReconError'] = recon_error
    cluster_df = pd.DataFrame(cluster_heuristic_counts, columns=heuristic_names)

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
    plt.title(f"t-SNE of {name} - Regex + GraphCodeBERT + VAE + Kmeans (HYDRA)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title=' ', bbox_to_anchor=(1.05, 1), loc='upper left')

    
    from scipy.spatial import ConvexHull
    for cluster in range(best_params['n_clusters']):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 1: 
            cluster_points = tsne_coords[cluster_indices]
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', alpha=0.5)

    
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

    
    cluster_heuristics = {i: {h: 0 for h in heuristic_names} for i in range(best_params['n_clusters'])}
    for idx, cluster in enumerate(clusters):
        for h_idx, h_name in enumerate(heuristic_names):
            if heuristics[idx, h_idx] == 1:
                cluster_heuristics[cluster][h_name] += 1

    
    print(f"\n{name} Dataset - Cluster Heuristic Counts:")
    for cluster in range(best_params['n_clusters']):
        print(f"\nCluster {cluster}:")
        for h_name, count in cluster_heuristics[cluster].items():
            print(f"{h_name}: {count}")

    
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