import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from itertools import product
from transformers import RobertaTokenizer, RobertaModel
import os
from collections import Counter

np.random.seed(42)  
torch.manual_seed(42) 

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(128, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def encode(self, x):
        x = self.encoder_layer(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * 0.001
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    return MSE

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

def get_code_features(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

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

def load_or_generate_embeddings(df, path, filename):
    embed_file = f"{filename}_embeddings.npy"
    heur_file = f"{filename}_heuristics.npy"
    commit_file = f"{filename}_commit_ids.npy"

    if os.path.exists(embed_file) and os.path.exists(heur_file) and os.path.exists(commit_file):
        embeddings = np.load(embed_file)
        heuristics = np.load(heur_file)
        commit_ids = np.load(commit_file)
    else:
        embeddings = []
        heuristics = []
        commit_ids = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing GraphCodeBERT embeddings and heuristics ({filename})"):
            code = row['PCode']
            embedding = get_code_features(code)
            risk_flags = analyze_risks(code)
            embeddings.append(embedding)
            heuristics.append([risk_flags[h] for h in [
                "Missing Null Check", "Race Condition", "Missing Bounds Check",
                "Unsafe Memory Allocation", "Logging Without Halting", "issue_detected"
            ]])
            commit_ids.append(row['Commit_Id'])
        embeddings = np.array(embeddings)
        heuristics = np.array(heuristics)
        np.save(embed_file, embeddings)
        np.save(heur_file, heuristics)
        np.save(commit_file, commit_ids)
    return embeddings, heuristics, commit_ids

df_train = pd.read_csv(train_path)
df_train['PCode'] = df_train['PCode'].astype(str)
embeddings_train, heuristics_train, commit_ids_train = load_or_generate_embeddings(df_train, train_path, "train")

features_train = np.concatenate([embeddings_train, heuristics_train], axis=1)
train_data, val_data = train_test_split(features_train, test_size=0.2, random_state=42)


input_dim = features_train.shape[1]
output_dim = input_dim
param_grid = {
    'latent_dim': [5, 10, 20],
    'n_clusters': [2, 3, 5]
}
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for params in tqdm(product(*param_grid.values()), desc="Tuning VAE and K-means"):
    latent_dim, n_clusters = params
    vae = VAE(input_dim, output_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    criterion = loss_function
    vae.train()
    best_val_loss = float('inf')
    patience = 20
    trigger_times = 0
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
        if (epoch + 1) % 10 == 0:
            avg_recon_loss = total_recon_loss / (len(train_data) / 128)
            print(f"Epoch [{epoch+1}/200], Latent Dim={latent_dim}, Train Recon Loss: {avg_recon_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
       
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at Epoch {epoch+1} for latent_dim={latent_dim}, n_clusters={n_clusters}")
            break
    vae.eval()
    with torch.no_grad():
        _, mu, log_var = vae(torch.FloatTensor(train_data).to(device))
        z_train = vae.reparameterize(mu, log_var)
    z_train = z_train.cpu().numpy()
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    train_clusters = clustering.fit_predict(z_train)
    if len(np.unique(train_clusters)) > 1:
        silhouette = silhouette_score(z_train, train_clusters)
        results.append({'latent_dim': latent_dim, 'n_clusters': n_clusters, 'silhouette': silhouette})
        print(f"latent_dim={latent_dim}, n_clusters={n_clusters}, Silhouette Score: {silhouette:.4f}")
best_result = max(results, key=lambda x: x['silhouette'])
print(f"Best Configuration: latent_dim={best_result['latent_dim']}, n_clusters={best_result['n_clusters']}, Silhouette Score: {best_result['silhouette']:.4f}")
best_params = {'latent_dim': best_result['latent_dim'], 'n_clusters': best_result['n_clusters']}

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