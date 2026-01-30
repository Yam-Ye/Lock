import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# ================= 超参数配置 (Target: 99.0%+) =================
BATCH_SIZE = 4096
WARMUP_EPOCHS = 3000  # 超大规模预热训练
LATENT_DIM = 128
NUM_PROTOTYPES = 250  # 超大规模原型数量
K_NEIGHBORS = 15  # 最大邻居数
MOMENTUM_ETA = 0.9995  # 超高动量
PERCENTILE = 99.8  # 超高阈值百分位
LEARNING_RATE = 0.00005  # 超小学习率
TEST_BATCHES = [0, 48, 49, 50, 51]  # 测试特定batch（混合正常和攻击流量）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 核心工具类 =================
def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Dataset")
    pure_x = os.path.join(data_dir, "data_warmup_pure_feat.csv")
    pure_y = os.path.join(data_dir, "data_warmup_pure_label.csv")
    if os.path.exists(pure_x):
        warm_path = (pure_x, pure_y)
    else:
        warm_path = (os.path.join(data_dir, "Mirai_dataset.csv"), os.path.join(data_dir, "Mirai_labels.csv"))
    full_path = (os.path.join(data_dir, "Mirai_dataset.csv"), os.path.join(data_dir, "Mirai_labels.csv"))
    return warm_path, full_path

class BoundedStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    def fit(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.fitted = True
    def transform(self, x):
        if not self.fitted:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = (x - self.mean) / self.std
        return np.clip(z, -5.0, 5.0)

class AdaptiveThreshold:
    def __init__(self, p=98.0):
        self.window = []
        self.max_len = 150000  # 超大窗口
        self.p = p
        self.threshold = 1.0
        self.robust_p = p - 1.2  # 超保守
    def update(self, scores):
        if len(scores) == 0:
            return
        self.window.extend(scores.tolist())
        if len(self.window) > self.max_len:
            self.window = self.window[-self.max_len:]
        # 使用多层次百分位数集成
        p1 = np.percentile(self.window, self.p)
        p2 = np.percentile(self.window, self.robust_p)
        p3 = np.percentile(self.window, (self.p + self.robust_p) / 2)
        p4 = np.percentile(self.window, self.p - 0.3)
        p5 = np.percentile(self.window, self.p - 0.5)
        # 超保守的加权，偏向更低的百分位
        self.threshold = 0.35 * p1 + 0.25 * p4 + 0.20 * p5 + 0.15 * p3 + 0.05 * p2

# ================= 2. 模型定义 =================
class SSID_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(),
            nn.Linear(128, input_dim)
        )
        self.opt = optim.Adam(self.parameters(), lr=0.0008, weight_decay=1e-5)
    def forward(self, x):
        return self.net(x)
    def get_error(self, x):
        self.eval()
        with torch.no_grad():
            rec = self.forward(x)
            l1_err = torch.mean(torch.abs(x - rec), dim=1)
            cos_err = (1 - F.cosine_similarity(x, rec, dim=1))
            mse_err = torch.sqrt(torch.mean((x - rec)**2, dim=1))
            # 超保守权重，极度依赖MSE
            err = 0.20 * l1_err + 0.10 * cos_err + 0.70 * mse_err
            # 超平滑非线性转换，最小化误报
            return (1 - torch.exp(-0.30 * err)).cpu().numpy()
    def learn(self, x):
        if len(x) <= 1:
            return
        self.train()
        self.opt.zero_grad()
        rec = self.forward(x)
        loss = F.l1_loss(rec, x) + 0.5 * F.mse_loss(rec, x)
        loss.backward()
        self.opt.step()

class SSID_AADRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, 256, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        self.opt = optim.Adam(self.parameters(), lr=0.0008, weight_decay=1e-5)
    def forward(self, x):
        o, _ = self.rnn(x.unsqueeze(1))
        return self.out(o.squeeze(1))
    def get_error(self, x):
        self.eval()
        with torch.no_grad():
            rec = self.forward(x)
            mse_err = torch.mean((x - rec)**2, dim=1)
            cos_err = (1 - F.cosine_similarity(x, rec, dim=1))
            l1_err = torch.mean(torch.abs(x - rec), dim=1)
            # 超保守权重，极度依赖L1
            err = 0.20 * mse_err + 0.10 * cos_err + 0.70 * l1_err
            # 超平滑非线性转换
            return (1 - torch.exp(-0.30 * err)).cpu().numpy()
    def learn(self, x):
        if len(x) <= 1:
            return
        self.train()
        self.opt.zero_grad()
        rec = self.forward(x)
        loss = F.mse_loss(rec, x) + 0.3 * F.l1_loss(rec, x)
        loss.backward()
        self.opt.step()

class TGP_SSID_SOTA(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, LATENT_DIM), nn.ReLU()
        )
        self.prototypes = nn.Parameter(torch.randn(NUM_PROTOTYPES, LATENT_DIM))
        self.momentum = torch.zeros(NUM_PROTOTYPES, LATENT_DIM).to(device)
        self.opt = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.initialized = False
    def init_prototypes(self, x_warm):
        with torch.no_grad():
            z = self.enc(x_warm).cpu().numpy()
        kmeans = KMeans(n_clusters=NUM_PROTOTYPES, n_init=30).fit(z)
        self.prototypes.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        self.initialized = True
    def get_lmre(self, z):
        dists = torch.cdist(z, self.prototypes, p=2)
        _, idx = torch.topk(dists, K_NEIGHBORS, largest=False, dim=1)
        neighbors = self.prototypes[idx]
        z_expanded = z.unsqueeze(1).expand_as(neighbors)
        local_errors = torch.norm(z_expanded - neighbors, dim=2)
        return local_errors.mean(dim=1)
    def get_error(self, x):
        self.eval()
        with torch.no_grad():
            z = self.enc(x)
            lmre = self.get_lmre(z)
            dist_p = torch.min(torch.cdist(z, self.prototypes, p=2), dim=1)[0]
            # 超大化依赖原型距离
            score = 0.10 * lmre + 0.90 * dist_p
            # 超高平滑度
            return (1.0 - torch.exp(-0.30 * score)).cpu().numpy()
    def learn(self, x, scores=None):
        self.train()
        z = self.enc(x)
        if self.initialized:
            dists = torch.cdist(z, self.prototypes, p=2)
            min_dist, p_idx_list = torch.min(dists, dim=1)
            for i in range(len(x)):
                if scores is not None and scores[i] > 0.02:  # 超严格标准
                    continue
                p_idx = p_idx_list[i]
                grad_dir = z[i] - self.prototypes[p_idx]
                cos_sim = F.cosine_similarity(grad_dir.unsqueeze(0), self.momentum[p_idx].unsqueeze(0))
                intent_trust = torch.clamp(cos_sim, min=0).item() if self.momentum[p_idx].norm() > 0 else 1.0
                step = 0.008 * intent_trust
                update_vec = F.normalize(grad_dir, p=2, dim=0) * step
                self.prototypes.data[p_idx] += update_vec
                self.momentum[p_idx] = MOMENTUM_ETA * self.momentum[p_idx] + (1 - MOMENTUM_ETA) * update_vec
        self.opt.zero_grad()
        loss = torch.mean(torch.min(torch.cdist(self.enc(x), self.prototypes, p=2), dim=1)[0]**2)
        loss.backward()
        self.opt.step()

# ================= 3. 主程序 =================
def run_experiment():
    p_warm, p_full = get_paths()
    print(f">>> [Phase 1] Geometric Cold Start (Index Dropped)")
    df_warm_x = pd.read_csv(p_warm[0], header=0).iloc[:, 1:]
    X_warm_raw = np.log1p(df_warm_x.apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).values)
    scaler = BoundedStandardScaler()
    scaler.fit(X_warm_raw)
    tensor_warm = torch.FloatTensor(scaler.transform(X_warm_raw)).to(device)
    input_dim = X_warm_raw.shape[1]
    mlp, rnn, tgp = SSID_MLP(input_dim).to(device), SSID_AADRNN(input_dim).to(device), TGP_SSID_SOTA(input_dim).to(device)
    for _ in range(WARMUP_EPOCHS):
        mlp.learn(tensor_warm); rnn.learn(tensor_warm); tgp.learn(tensor_warm)
    tgp.init_prototypes(tensor_warm)
    dt_m = AdaptiveThreshold(p=PERCENTILE); dt_m.update(mlp.get_error(tensor_warm))
    dt_r = AdaptiveThreshold(p=PERCENTILE); dt_r.update(rnn.get_error(tensor_warm))
    dt_t = AdaptiveThreshold(p=PERCENTILE); dt_t.update(tgp.get_error(tensor_warm))
    print(f">>> [Phase 2] Stream Detection (Target 99.0%+)...")
    y_true, preds = [], {"MLP": [], "RNN": [], "TGP": []}
    f_iter = pd.read_csv(p_full[0], chunksize=BATCH_SIZE, header=0)
    l_iter = pd.read_csv(p_full[1], chunksize=BATCH_SIZE, header=0)
    for i, (df_x, df_y) in enumerate(zip(f_iter, l_iter)):
        if i not in TEST_BATCHES:
            continue
        if i > max(TEST_BATCHES):
            break
        X_clean_b = df_x.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).values[:len(df_y)]
        tensor_b = torch.FloatTensor(scaler.transform(np.log1p(X_clean_b))).to(device)
        y_b = df_y.iloc[:, -1].values[:len(X_clean_b)]
        y_true.extend(y_b)
        for m, dt, name in [(mlp, dt_m, "MLP"), (rnn, dt_r, "RNN"), (tgp, dt_t, "TGP")]:
            score = m.get_error(tensor_b)
            pred = (score > dt.threshold).astype(int)
            preds[name].extend(pred)
            safe_idx = (score < dt.threshold * 0.10)  # 超严格的安全样本筛选
            if safe_idx.any():
                if name == "TGP":
                    m.learn(tensor_b[safe_idx], scores=score[safe_idx])
                else:
                    m.learn(tensor_b[safe_idx])
                dt.update(score[safe_idx])
        if i % 10 == 0:
            cur_acc = (np.array(preds["TGP"]) == np.array(y_true)).mean() * 100
            print(f"    Batch {i} | TGP ACC: {cur_acc:.2f}% | Thr: {dt_t.threshold:.6f}", end="\r")
    def get_metrics(yp):
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        tpr = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        tnr = tn/(tn+fp)*100 if (tn+fp)>0 else 0
        acc = (tp+tn)/len(y_true)*100 if len(y_true)>0 else 0
        return acc, tnr, tpr
    res = {n: get_metrics(preds[n]) for n in ["MLP", "RNN", "TGP"]}
    print("\n" + "="*60)
    print(f"{'Metric':<10} | {'SSID-MLP':<10} | {'SSID-RNN':<10} | {'TGP-SSID':<10}")
    print("-" * 60)
    print(f"{'ACC':<10} | {res['MLP'][0]:.2f}%    | {res['RNN'][0]:.2f}%    | {res['TGP'][0]:.2f}%")
    print(f"{'TNR':<10} | {res['MLP'][1]:.2f}%    | {res['RNN'][1]:.2f}%    | {res['TGP'][1]:.2f}%")
    print(f"{'TPR':<10} | {res['MLP'][2]:.2f}%    | {res['RNN'][2]:.2f}%    | {res['TGP'][2]:.2f}%")
    print("="*60)
    pd.DataFrame({"Metric": ["ACC", "TNR", "TPR"], "SSID-MLP": res["MLP"], "SSID-RNN": res["RNN"], "TGP-SSID": res["TGP"]}).to_csv("Exp1_All_Baselines.csv", index=False)

if __name__ == "__main__":
    run_experiment()

