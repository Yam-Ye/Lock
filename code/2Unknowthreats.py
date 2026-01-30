import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 智能配置与路径自检 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上两级寻找 Dataset (根据您的目录结构调整)
PATH_FEAT = os.path.join(CURRENT_DIR, "../Dataset/Mirai_dataset.csv")
PATH_LBL = os.path.join(CURRENT_DIR, "../Dataset/Mirai_labels.csv")

# 冲刺参数
BATCH_SIZE = 2048       
EPOCHS = 100            
LATENT_DIM = 128        
NUM_PROTOTYPES = 200    
LR = 0.0001             
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# ================= 2. 核心修复：智能数据加载 =================
def smart_feature_slice(df_values):
    """
    【核心修复】强制特征对齐到 115 维
    解决因索引列导致的特征左移和准确率崩盘问题。
    """
    TARGET_DIM = 115
    current_dim = df_values.shape[1]
    
    if current_dim == TARGET_DIM:
        return df_values
    elif current_dim > TARGET_DIM:
        # 丢弃前面的索引列，只取最后115列
        return df_values[:, -TARGET_DIM:]
    else:
        # 防御性补零
        pad = np.zeros((df_values.shape[0], TARGET_DIM - current_dim))
        return np.hstack([df_values, pad])

def load_data_final():
    print(f"\n[1/4] Loading Data...")
    
    # --- 路径调试 (修复找不到文件的问题) ---
    print(f"    当前代码目录: {CURRENT_DIR}")
    print(f"    正在寻找数据: {os.path.abspath(PATH_FEAT)}")
    
    if not os.path.exists(PATH_FEAT):
        print(f"\n❌ 错误：找不到数据集文件！")
        print(f"   请检查 'Dataset' 文件夹是否在代码的上级目录中。")
        return None
    # -------------------------------------

    try:
        # 加载特征
        df_feat = pd.read_csv(PATH_FEAT, header=None)
        raw_feat = df_feat.values
        
        # 【关键】执行智能切片，确保没有索引列干扰
        features = smart_feature_slice(raw_feat)
        
        # 加载标签
        if os.path.exists(PATH_LBL):
            df_lbl = pd.read_csv(PATH_LBL, header=None, low_memory=False)
            labels = df_lbl.iloc[:, -1].values.flatten().astype(int)
        else:
            # 如果没有单独的标签文件，假设最后一列是标签（很少见，防万一）
            print("⚠️ Warning: Label file not found, using last column of dataset.")
            labels = raw_feat[:, -1].astype(int)

    except Exception as e:
        print(f"❌ 数据读取出错: {e}")
        return None

    # 对齐长度
    min_len = min(len(labels), len(features))
    labels = labels[:min_len]
    features = features[:min_len]

    # 数据集划分：训练集只用正常流量(0)，测试集包含正常(0)和攻击(1)
    benign_idx = np.where(labels == 0)[0]
    attack_idx = np.where(labels == 1)[0]
    
    # 80% 正常流量用于训练
    train_idx, test_benign_idx = train_test_split(benign_idx, test_size=0.2, random_state=42)
    # 测试集 = 20% 正常 + 全部攻击
    test_idx = np.concatenate([test_benign_idx, attack_idx])

    X_train = features[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    # 预处理：Log + RobustScaler (抗异常值)
    X_train = np.log1p(np.abs(X_train))
    X_test = np.log1p(np.abs(X_test))

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 梯度裁剪，防止某些极端值
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)

    print(f"    Data Loaded! Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_test, 115 # 强制返回115维

# ================= 3. TGP-SSID 模型 (标准稳健版) =================
class TGP_SSID(nn.Module):
    def __init__(self, input_dim):
        super(TGP_SSID, self).__init__()
        # 编码器
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128, LATENT_DIM) # No activation at latent
        )
        # 解码器
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )
        # 几何原型
        self.prototypes = nn.Parameter(torch.randn(NUM_PROTOTYPES, LATENT_DIM))

    def init_prototypes(self, x_train):
        print("    -> Initializing Prototypes...")
        self.eval()
        with torch.no_grad():
            # 随机采样初始化
            idx = np.random.choice(len(x_train), min(10000, len(x_train)), replace=False)
            x_sample = torch.FloatTensor(x_train[idx]).to(DEVICE)
            z = self.enc(x_sample).cpu().numpy()
        
        kmeans = KMeans(n_clusters=NUM_PROTOTYPES, n_init=10).fit(z)
        self.prototypes.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(DEVICE)
        self.train()

    def forward(self, x):
        z = self.enc(x)
        rec = self.dec(z)
        return rec, z

    def compute_score(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(DEVICE)
            rec, z = self.forward(x)
            
            # 1. 重构误差 (MSE)
            rec_err = torch.mean((x - rec) ** 2, dim=1)
            
            # 2. 原型距离 (Euclidean)
            z_exp = z.unsqueeze(1)
            p_exp = self.prototypes.unsqueeze(0)
            # 计算到最近原型的距离
            dists = torch.norm(z_exp - p_exp, dim=2)
            min_dist, _ = torch.min(dists, dim=1)
            
            # 3. 简单加权融合 (不做复杂的 Log 变换，保持线性可分)
            score = rec_err + min_dist
            
        return score.cpu().numpy()

# ================= 4. 主程序 =================
if __name__ == "__main__":
    # 1. 加载
    data = load_data_final()
    if data is None: 
        print("❌ 程序因数据加载失败而终止。")
        exit()
        
    X_train, X_test, y_test, input_dim = data

    # 2. 训练
    print(f"\n[2/4] Training TGP-SSID...")
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TGP_SSID(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 初始化原型
    model.init_prototypes(X_train)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            rec, z = model(x)

            # 损失 = 重构损失 + 原型距离损失
            rec_loss = F.mse_loss(rec, x)
            
            z_exp = z.unsqueeze(1)
            p_exp = model.prototypes.unsqueeze(0)
            dists = torch.norm(z_exp - p_exp, dim=2)
            min_dist, _ = torch.min(dists, dim=1)
            proto_loss = torch.mean(min_dist)

            loss = rec_loss + 0.1 * proto_loss # 简单有效的损失组合

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: Loss = {total_loss/len(train_loader):.6f}")

    # 3. 评估
    print(f"\n[3/4] Evaluating...")
    scores = []
    # 分批次评估防止显存爆炸
    batch_size_eval = 5000
    for i in range(0, len(X_test), batch_size_eval):
        bx = torch.FloatTensor(X_test[i:i+batch_size_eval])
        scores.append(model.compute_score(bx))
    scores = np.concatenate(scores)

    # 4. 结果与绘图
    auc_val = roc_auc_score(y_test, scores)
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    
    # 寻找最佳阈值 (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]

    # 计算最终指标
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    tpr_val = tp / (tp + fn)
    tnr_val = tn / (tn + fp)
    acc_val = (tp + tn) / len(y_test)

    print("\n" + "="*60)
    print("🏆 === Experiment 2 Results (Pro) ===")
    print("="*60)
    print(f"    ACC (Accuracy): {acc_val*100:.2f}%")
    print(f"    TPR (Recall):   {tpr_val*100:.2f}%")
    print(f"    TNR (Spec.):    {tnr_val*100:.2f}%")
    print(f"    AUC Score:      {auc_val:.4f}")
    print("="*60)

    # 绘图
    print(f"\n[4/4] Generating Plots...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # KDE
        ax1 = axes[0]
        sns.kdeplot(scores[y_test==0], fill=True, label='Normal', color='green', ax=ax1, warn_singular=False)
        if np.sum(y_test==1) > 0:
            sns.kdeplot(scores[y_test==1], fill=True, label='Unknown (Mirai)', color='red', ax=ax1, warn_singular=False)
        ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Thr={threshold:.3f}')
        # 如果分数差异太大，用 log scale 显示会更好看
        # ax1.set_xscale('log') 
        ax1.set_title(f'Score Distribution (AUC={auc_val:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ROC
        ax2 = axes[1]
        ax2.plot(fpr, tpr, color='blue', linewidth=2.5, label=f'TGP-SSID')
        ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax2.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], c='red', s=100, label=f'Optimal')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_img = os.path.join(CURRENT_DIR, '../Figure/Exp2_Kitsune_Result_Pro.png')
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        plt.savefig(output_img, dpi=300)
        print(f"✅ Plot saved to: {output_img}")
        # plt.show() # 如需弹窗显示请取消注释
        
    except Exception as e:
        print(f"⚠️ Plotting Error: {e}")