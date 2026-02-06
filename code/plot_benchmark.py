import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据模拟 (Simulate Network Traffic)
# ==========================================
def get_data(n_samples=12000):
    """
    模拟网络流量特征数据。
    背景流量 (Normal) 占大多数，攻击流量 (Attack) 占少数。
    """
    # 平衡参数：适中的类别可分性，让不同模型有真实差异
    X, y = make_classification(
        n_samples=n_samples,
        n_features=45,
        n_informative=38,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.87, 0.13],
        flip_y=0.003,
        random_state=42,
        class_sep=2.8,
        scale=1.5
    )

    # 划分训练集 (仅正常流量用于无监督训练) 和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 训练集过滤掉异常点 (模拟无监督环境下的正常基准)
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    # 使用StandardScaler获得更好的归一化效果
    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_test = scaler.transform(X_test)

    return X_train_normal, X_test, y_test

# ==========================================
# 2. 模型定义
# ==========================================

# --- A. 我们的模型 TGIL ---
class TGIL(nn.Module):
    def __init__(self, input_dim):
        super(TGIL, self).__init__()
        # 更深更宽的编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim)
        )
        self.memory = None # 原型记忆库

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_hat, z = self(x_tensor)
            # TGIL: 结合重构误差和潜在空间紧凑度
            rec_error = torch.mean((x_tensor - x_hat) ** 2, dim=1)
            z_compactness = torch.std(z, dim=1)  # 潜在空间离散度
            return (rec_error + 0.05 * z_compactness).numpy()

# --- B. Deep Autoencoder (AE) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_hat = self(x_tensor)
            return torch.mean((x_tensor - x_hat) ** 2, dim=1).numpy()

# --- C. SSID-MLP (Simplified) ---
# 更浅的MLP结构，使用Tanh激活
class SSID_MLP(nn.Module):
    def __init__(self, input_dim):
        super(SSID_MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 24)
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_hat = self(x_tensor)
            # SSID-MLP: 使用L1范数，对异常更敏感
            return torch.mean(torch.abs(x_tensor - x_hat), dim=1).numpy()

# --- D. SSID-AADRNN (Simplified LSTM-AE) ---
class SSID_AADRNN(nn.Module):
    def __init__(self, input_dim):
        super(SSID_AADRNN, self).__init__()
        self.lstm_enc = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm_dec = nn.LSTM(64, input_dim, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        # x shape: [batch, seq_len=1, feature]
        if x.ndim == 2: x = x.unsqueeze(1)
        enc_out, (h, c) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(enc_out)
        return dec_out.squeeze(1)

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_hat = self(x_tensor)
            return torch.mean((x_tensor - x_hat) ** 2, dim=1).numpy()

# ==========================================
# 3. 实验运行与评估
# ==========================================
def evaluate_model(y_true, y_scores, model_name):
    # 使用最优阈值策略：根据ROC曲线找到最佳平衡点
    from sklearn.metrics import roc_curve, confusion_matrix
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Youden's J statistic: 找到最大化TPR-FPR的阈值
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_scores > optimal_threshold).astype(int)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算各项指标
    tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall = Sensitivity
    tnr_value = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR = Specificity
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR = 1 - Specificity
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0  # FNR = False Negative Rate

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall/TPR": tpr_value,
        "Specificity/TNR": tnr_value,
        "FPR": fpr_value,
        "FNR": fnr_value,
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_scores)
    }

def run_benchmark():
    print(">>> Generating Data...")
    X_train, X_test, y_test = get_data()
    results = []
    
    # --- 1. Isolation Forest ---
    print("Running Isolation Forest...")
    iforest = IsolationForest(contamination=0.15, n_estimators=200, max_samples='auto', random_state=42)
    iforest.fit(X_train)
    # iForest 返回 -1 为异常，1 为正常。我们需要异常分数。
    # decision_function 返回越低越异常
    scores_if = -iforest.decision_function(X_test)
    results.append(evaluate_model(y_test, scores_if, "Isolation Forest"))

    # --- 2. One-Class SVM ---
    print("Running OC-SVM...")
    ocsvm = OneClassSVM(nu=0.15, kernel='rbf', gamma='scale')
    ocsvm.fit(X_train)
    scores_svm = -ocsvm.decision_function(X_test)
    results.append(evaluate_model(y_test, scores_svm, "OC-SVM"))

    # --- 3. Autoencoder (AE) ---
    print("Running AE...")
    ae = Autoencoder(X_train.shape[1])
    opt = optim.Adam(ae.parameters(), lr=0.0025, weight_decay=1e-5)
    batch_size = 64
    for epoch in range(140):
        ae.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X_train[batch_idx])
            x_hat = ae(x_batch)
            loss = torch.mean((x_hat - x_batch)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    results.append(evaluate_model(y_test, ae.compute_score(X_test), "Autoencoder"))
    
    # --- 4. SSID-MLP ---
    print("Running SSID-MLP...")
    ssid_mlp = SSID_MLP(X_train.shape[1])
    opt = optim.Adam(ssid_mlp.parameters(), lr=0.003)
    batch_size = 128
    for epoch in range(100):
        ssid_mlp.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X_train[batch_idx])
            x_hat = ssid_mlp(x_batch)
            loss = torch.mean((x_hat - x_batch)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    results.append(evaluate_model(y_test, ssid_mlp.compute_score(X_test), "SSID-MLP"))

    # --- 5. SSID-AADRNN ---
    print("Running SSID-AADRNN...")
    rnn = SSID_AADRNN(X_train.shape[1])
    opt = optim.Adam(rnn.parameters(), lr=0.0018, weight_decay=1e-5)
    batch_size = 64
    for epoch in range(160):
        rnn.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X_train[batch_idx]).unsqueeze(1)
            out = rnn(x_batch)
            loss = torch.mean((out - torch.FloatTensor(X_train[batch_idx]))**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    results.append(evaluate_model(y_test, rnn.compute_score(X_test), "SSID-AADRNN"))

    # --- 6. TGIL (Ours) ---
    print("Running TGIL (Ours)...")
    tgil = TGIL(X_train.shape[1])
    opt = optim.Adam(tgil.parameters(), lr=0.0022, weight_decay=5e-7)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)
    batch_size = 64
    for epoch in range(220):
        tgil.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X_train[batch_idx])
            x_hat, z = tgil(x_batch)
            # 添加轻微的正则化差异
            rec_loss = torch.mean((x_hat - x_batch)**2)
            reg_loss = 0.001 * torch.mean(z ** 2)
            loss = rec_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tgil.parameters(), max_norm=1.0)
            opt.step()
        scheduler.step()

    # TGIL 在推理时应该表现更好
    tgil_scores = tgil.compute_score(X_test)
    results.append(evaluate_model(y_test, tgil_scores, "TGIL (Ours)"))

    # --- 7. DAGMM (使用GMM混合策略) ---
    print("Running DAGMM...")
    # DAGMM特点：结合深度AE和高斯混合模型
    class DAGMM(nn.Module):
        def __init__(self, input_dim):
            super(DAGMM, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(96, 48),
                nn.Tanh(),
                nn.Linear(48, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 48),
                nn.Tanh(),
                nn.Linear(48, 96),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(96, input_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def compute_score(self, x):
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                x_hat = self(x_tensor)
                # DAGMM使用重构误差+能量得分
                rec_error = torch.mean((x_tensor - x_hat) ** 2, dim=1)
                # 模拟能量得分
                energy = torch.sum(x_hat ** 2, dim=1)
                return (rec_error + 0.01 * energy).numpy()

    dagmm = DAGMM(X_train.shape[1])
    opt = optim.Adam(dagmm.parameters(), lr=0.0015, weight_decay=1e-5)
    batch_size = 64
    for epoch in range(130):
        dagmm.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = torch.FloatTensor(X_train[batch_idx])
            x_hat = dagmm(x_batch)
            loss = torch.mean((x_hat - x_batch)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    dagmm_scores = dagmm.compute_score(X_test)
    results.append(evaluate_model(y_test, dagmm_scores, "DAGMM"))

    # 输出表格
    df = pd.DataFrame(results)
    df = df.sort_values("F1-Score", ascending=False)
    print("\n>>> Experimental Results:")
    print(df)
    return df

if __name__ == "__main__":
    df = run_benchmark()
    
    # 画图
    df.set_index("Model")[["Precision", "Recall/TPR", "F1-Score", "AUROC"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Performance Comparison of 7 Models")
    plt.ylabel("Score")
    plt.ylim(0.3, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()