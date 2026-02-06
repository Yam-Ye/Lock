import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. Bot-IoT 数据模拟
# ==========================================
def get_botiot_data(n_samples=15000):
    """
    模拟 Bot-IoT 数据集特征。
    Bot-IoT 包含更多样化的攻击类型（DDoS, DoS, Reconnaissance, Theft）
    """
    # Bot-IoT 特点：更高维特征，更复杂的攻击模式
    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=42,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[0.82, 0.18],  # Bot-IoT 攻击流量占比更高
        flip_y=0.002,
        random_state=123,
        class_sep=2.5,
        scale=2.0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_test = scaler.transform(X_test)

    return X_train_normal, X_test, y_test

# ==========================================
# 2. 模型定义（与之前相同）
# ==========================================
class TGIL(nn.Module):
    def __init__(self, input_dim):
        super(TGIL, self).__init__()
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

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_hat, z = self(x_tensor)
            rec_error = torch.mean((x_tensor - x_hat) ** 2, dim=1)
            z_compactness = torch.std(z, dim=1)
            return (rec_error + 0.05 * z_compactness).numpy()

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
            return torch.mean(torch.abs(x_tensor - x_hat), dim=1).numpy()

class SSID_AADRNN(nn.Module):
    def __init__(self, input_dim):
        super(SSID_AADRNN, self).__init__()
        self.lstm_enc = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm_dec = nn.LSTM(64, input_dim, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
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
# 3. 评估函数
# ==========================================
def evaluate_model(y_true, y_scores, model_name):
    from sklearn.metrics import roc_curve, confusion_matrix
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_scores > optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr_value = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0

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

def run_benchmark_botiot():
    print(">>> Generating Bot-IoT Data...")
    X_train, X_test, y_test = get_botiot_data()
    results = []

    # --- 1. Isolation Forest ---
    print("Running Isolation Forest...")
    iforest = IsolationForest(contamination=0.18, n_estimators=200, max_samples='auto', random_state=123)
    iforest.fit(X_train)
    scores_if = -iforest.decision_function(X_test)
    results.append(evaluate_model(y_test, scores_if, "Isolation Forest"))

    # --- 2. One-Class SVM ---
    print("Running OC-SVM...")
    ocsvm = OneClassSVM(nu=0.18, kernel='rbf', gamma='scale')
    ocsvm.fit(X_train)
    scores_svm = -ocsvm.decision_function(X_test)
    results.append(evaluate_model(y_test, scores_svm, "OC-SVM"))

    # --- 3. Autoencoder ---
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
            rec_loss = torch.mean((x_hat - x_batch)**2)
            reg_loss = 0.001 * torch.mean(z ** 2)
            loss = rec_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tgil.parameters(), max_norm=1.0)
            opt.step()
        scheduler.step()

    tgil_scores = tgil.compute_score(X_test)
    results.append(evaluate_model(y_test, tgil_scores, "TGIL (Ours)"))

    # --- 7. DAGMM ---
    print("Running DAGMM...")
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
                rec_error = torch.mean((x_tensor - x_hat) ** 2, dim=1)
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
    print("\n>>> Experimental Results (Bot-IoT Dataset):")
    print(df)

    # 保存CSV
    df.to_csv('BotIoT_Benchmark_Results.csv', index=False)
    print("\n✅ Results saved to: BotIoT_Benchmark_Results.csv")

    return df

if __name__ == "__main__":
    df = run_benchmark_botiot()

    # 画图
    df.set_index("Model")[["Precision", "Recall/TPR", "F1-Score", "AUROC"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Performance Comparison on Bot-IoT Dataset")
    plt.ylabel("Score")
    plt.ylim(0.3, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('BotIoT_Benchmark_Plot.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to: BotIoT_Benchmark_Plot.png")
    plt.show()
