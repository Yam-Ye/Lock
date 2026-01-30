import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_FEAT = os.path.join(CURRENT_DIR, "../Dataset/Mirai_dataset.csv")
PATH_LBL = os.path.join(CURRENT_DIR, "../Dataset/Mirai_labels.csv")

BATCH_SIZE = 2048       
EPOCHS = 50             
LATENT_DIM = 128        
NUM_PROTOTYPES = 5      
LR = 0.0005             
MOMENTUM = 0.99         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# ================= 2. 数据加载 =================
def smart_feature_slice(df_values):
    TARGET_DIM = 115
    if df_values.shape[1] > TARGET_DIM: return df_values[:, -TARGET_DIM:]
    elif df_values.shape[1] < TARGET_DIM:
        pad = np.zeros((df_values.shape[0], TARGET_DIM - df_values.shape[1]))
        return np.hstack([df_values, pad])
    return df_values

def load_raw_data():
    if not os.path.exists(PATH_FEAT): return None
    try:
        df_feat = pd.read_csv(PATH_FEAT, header=None)
        features = smart_feature_slice(df_feat.values)
        labels = pd.read_csv(PATH_LBL, header=None).iloc[:, -1].values.flatten().astype(int) if os.path.exists(PATH_LBL) else np.zeros(len(features))
        min_len = min(len(labels), len(features))
        features, labels = features[:min_len], labels[:min_len]

        benign_idx = np.where(labels == 0)[0]
        attack_idx = np.where(labels == 1)[0]
        train_idx, test_benign_idx = train_test_split(benign_idx, test_size=0.2, random_state=42)
        test_idx = np.concatenate([test_benign_idx, attack_idx])

        return features[train_idx], features[test_idx], labels[test_idx], 115
    except: return None

# ================= 3. 模型定义 =================
class TGP_Gap(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, LATENT_DIM) 
        )
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        self.register_buffer("prototypes", torch.randn(NUM_PROTOTYPES, LATENT_DIM))

    def init_prototypes(self, x_train):
        self.eval()
        with torch.no_grad():
            idx = np.random.choice(len(x_train), min(2000, len(x_train)), replace=False)
            z = self.enc(torch.from_numpy(x_train[idx]).float().to(DEVICE)).cpu().numpy()
        kmeans = KMeans(n_clusters=NUM_PROTOTYPES, n_init=10).fit(z)
        self.prototypes.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(DEVICE)
        self.train()

    def update_prototypes_momentum(self, z_batch):
        with torch.no_grad():
            dists = torch.cdist(z_batch, self.prototypes)
            min_dist_idx = torch.argmin(dists, dim=1)
            for i in range(NUM_PROTOTYPES):
                mask = (min_dist_idx == i)
                if mask.sum() > 0:
                    z_avg = z_batch[mask].mean(dim=0)
                    self.prototypes[i] = MOMENTUM * self.prototypes[i] + (1 - MOMENTUM) * z_avg

    def forward(self, x):
        z = self.enc(x)
        rec = self.dec(z)
        return rec, z

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(DEVICE)
            rec, z = self.forward(x)
            
            diff = (x - rec) ** 2
            mse = torch.mean(diff, dim=1)
            dists = torch.cdist(z, self.prototypes)
            min_dist, _ = torch.min(dists, dim=1)
            
            score = mse + min_dist 
        return score.cpu().numpy()

class SSID_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))
    def forward(self, x): return self.net(x)
    def compute_score(self, x):
        with torch.no_grad(): return torch.mean((x - self.forward(x.to(DEVICE)))**2, dim=1).cpu().numpy()

class SSID_ADDRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, input_dim)
    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(1))
        return self.fc(out.squeeze(1))
    def compute_score(self, x):
        with torch.no_grad(): return torch.mean((x - self.forward(x.to(DEVICE)))**2, dim=1).cpu().numpy()

# ================= 4. 训练逻辑 =================
def train_model(model, loader, model_name):
    print(f"    Training {model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    
    for epoch in range(EPOCHS):
        for batch in loader:
            x_clean = batch[0].to(DEVICE)
            optimizer.zero_grad()
            
            if 'TGP' in model_name:
                noise = torch.randn_like(x_clean) * 0.2
                x_input = x_clean + noise
            else:
                x_input = x_clean 

            if 'TGP' in model_name:
                rec, z = model(x_input)
                rec_loss = F.mse_loss(rec, x_clean)
                dists = torch.cdist(z, model.prototypes)
                min_dist, _ = torch.min(dists, dim=1)
                loss = rec_loss + 0.1 * torch.mean(min_dist)
                
                loss.backward()
                optimizer.step()
                model.update_prototypes_momentum(z.detach())
            else:
                rec = model(x_input)
                loss = F.mse_loss(rec, x_clean)
                loss.backward()
                optimizer.step()

# ================= 5. 测试逻辑 =================
def test_robustness(model, X_test_raw, y_test, noise_levels, scaler):
    aucs = []
    scale_vec = scaler.scale_.reshape(1, -1)
    mean_vec = scaler.mean_.reshape(1, -1)
    
    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, X_test_raw.shape) * scale_vec
        X_test_noisy = X_test_raw + noise
        X_test_normalized = (X_test_noisy - mean_vec) / scale_vec
        X_test_normalized = np.clip(X_test_normalized, -10.0, 10.0)
        
        scores = []
        for i in range(0, len(X_test_normalized), 5000):
            batch = torch.from_numpy(X_test_normalized[i:i+5000]).float()
            scores.append(model.compute_score(batch))
        scores = np.concatenate(scores)
        try: val = roc_auc_score(y_test, scores)
        except: val = 0.5
        aucs.append(val)
    return aucs

# ================= 6. 主流程 =================
if __name__ == "__main__":
    data = load_raw_data()
    if data is None: exit()
    X_train_raw, X_test_raw, y_test, input_dim = data
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_train_norm = np.clip(X_train_norm, -10.0, 10.0)
    
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_train_norm).float()), batch_size=BATCH_SIZE, shuffle=True)

    models = {
        'TGP (Ours)': TGP_Gap(input_dim).to(DEVICE),
        'SSID-ADDRNN': SSID_ADDRNN(input_dim).to(DEVICE),
        'SSID-MLP': SSID_MLP(input_dim).to(DEVICE)
    }

    print(f"\n[2/6] Training Models...")
    models['TGP (Ours)'].init_prototypes(X_train_norm)
    for name, model in models.items():
        train_model(model, loader, name)

    print(f"\n[3/6] Running High-Res Tests...")
    
    noise_levels = np.linspace(0, 5.0, 21) 
    
    results = {}
    for name, model in models.items():
        print(f"    -> Testing {name}...")
        results[name] = test_robustness(model, X_test_raw, y_test, noise_levels, scaler)

    print(f"\n[4/6] Plotting...")
    plt.figure(figsize=(10, 7))
    
    draw_order = ['SSID-MLP', 'SSID-ADDRNN', 'TGP (Ours)']
    
    # 【修改点】统一线宽 lw=2
    styles = {
        'TGP (Ours)': {'color': '#d62728', 'fmt': '-', 'lw': 2, 'label': 'TGP-SSID (Proposed)', 'zorder': 10}, 
        'SSID-ADDRNN': {'color': '#1f77b4', 'fmt': '--', 'lw': 2, 'label': 'SSID-ADDRNN', 'zorder': 5}, 
        'SSID-MLP': {'color': '#2ca02c', 'fmt': '-.', 'lw': 2, 'label': 'SSID-MLP', 'zorder': 1}       
    }

    for name in draw_order:
        aucs = results[name]
        s = styles[name]
        plt.plot(noise_levels, aucs, linestyle=s['fmt'].replace('o','').replace('s','').replace('^',''), 
                 color=s['color'], linewidth=s['lw'], label=s['label'], zorder=s['zorder'])
        
        marker_style = 'o' if name=='TGP (Ours)' else ('s' if 'ADDRNN' in name else '^')
        # 统一 Marker 大小为 30
        plt.scatter(noise_levels[::5], aucs[::5], marker=marker_style, color=s['color'], s=30, zorder=s['zorder']+1)

        if name == 'TGP (Ours)':
            plt.text(noise_levels[0], aucs[0]+0.01, f'{aucs[0]:.3f}', color=s['color'], fontweight='bold', ha='center', fontsize=11)
            plt.text(noise_levels[-1], aucs[-1]+0.01, f'{aucs[-1]:.3f}', color=s['color'], fontweight='bold', ha='center', fontsize=11)

    plt.title('Robustness Comparison (Denoising vs Standard)', fontsize=16)
    plt.xlabel('Noise Multiplier (Sigma)', fontsize=14)
    plt.ylabel('Detection AUC', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower left')
    
    save_path = os.path.join(CURRENT_DIR, '../Figure/Exp3_TGP_Equal_Width.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Saved to: {save_path}")