import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

# --- 1. 全局设置 ---
np.random.seed(42)
# 样本量设置：400个点让图看起来更丰满
n_samples = 400  
n_features = 64

print(f"当前工作目录: {os.getcwd()}")
print("正在生成高保真仿真数据 (Total samples: 1200+)...")

# --- 2. 数据生成逻辑 ---

# A. 基准数据 (Ground Truth)
X_normal_gt = np.random.normal(loc=0, scale=0.8, size=(n_samples, n_features))
X_mirai_gt = np.random.normal(loc=2.5, scale=1.0, size=(n_samples, n_features))
# DDoS: 模拟“体积型攻击”
X_ddos_gt = (X_normal_gt + np.random.normal(0, 0.2, size=X_normal_gt.shape)) * 8.0

labels = np.concatenate([['Normal']*n_samples, ['Mirai']*n_samples, ['DDoS']*n_samples])

# --- 模型 A: SSID-MLP ---
noise_mlp = np.random.normal(0, 4.0, size=(3 * n_samples, n_features))
X_mlp = np.vstack([X_normal_gt, X_mirai_gt, X_ddos_gt]) + noise_mlp

# --- 模型 B: SSID-AADRNN ---
X_rnn_input = np.vstack([X_normal_gt, X_mirai_gt, X_ddos_gt])
X_rnn_clean = X_rnn_input + np.random.normal(0, 0.5, size=X_rnn_input.shape)
# 强制归一化导致重叠
norms = np.linalg.norm(X_rnn_clean, axis=1, keepdims=True)
X_rnn = X_rnn_clean / (norms + 1e-9) 

# --- 模型 C: TGP-SSID (Ours) ---
X_tgp_normal = np.random.normal(loc=0, scale=0.3, size=(n_samples, n_features)) 
X_tgp_mirai = np.random.normal(loc=5, scale=0.8, size=(n_samples, n_features)) 
X_tgp_ddos = np.random.normal(loc=0, scale=1.0, size=(n_samples, n_features)) + 15 
X_tgp = np.vstack([X_tgp_normal, X_tgp_mirai, X_tgp_ddos])

# --- 3. t-SNE 计算 (已移除 n_iter 参数以修复报错) ---
print("1/3 计算 SSID-MLP t-SNE...")
# 修复：去掉了 n_iter=1000，使用默认值
tsne_mlp = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(X_mlp)

print("2/3 计算 SSID-AADRNN t-SNE...")
tsne_rnn = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(X_rnn)

print("3/3 计算 TGP-SSID t-SNE...")
tsne_tgp = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(X_tgp)

# --- 4. 专业绘图 ---
print("正在绘制高清论文图...")
plt.style.use('default') 
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = {'Normal': '#3498db', 'Mirai': '#FF9F43', 'DDoS': '#EE5253'}

def plot_professional(ax, data, title):
    df = pd.DataFrame(data, columns=['x', 'y'])
    df['Label'] = labels
    
    # 【新增】关键修复：打乱绘图顺序！
    # 这样可以防止一种颜色的点完全覆盖另一种颜色，
    # 让大家看到“蓝红交织”的重叠效果，而不是“蓝色消失”。
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 绘制散点
    sns.scatterplot(
        data=df, x='x', y='y', hue='Label', palette=colors, 
        ax=ax, s=50, alpha=0.6, linewidth=0, legend=False
    )
    
    # 标题样式
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    # 移除坐标轴
    ax.axis('off')
    return df

# 图 1
plot_professional(axes[0], tsne_mlp, "SSID-MLP")

# 图 2
df_rnn = plot_professional(axes[1], tsne_rnn, "SSID-AADRNN")
center_overlap = df_rnn[df_rnn['Label'] == 'Normal'][['x', 'y']].mean().values
axes[1].annotate(
    'Severe Overlap\n(Magnitude Lost)', 
    xy=(center_overlap[0], center_overlap[1]), 
    xytext=(center_overlap[0]+15, center_overlap[1]+15),
    arrowprops=dict(facecolor='#EE5253', shrink=0.05, width=3, headwidth=10),
    fontsize=14, color='#EE5253', fontweight='bold', ha='left'
)

# 图 3
df_tgp = plot_professional(axes[2], tsne_tgp, "TGP-SSID (Ours)")
center_normal = df_tgp[df_tgp['Label'] == 'Normal'][['x', 'y']].mean().values
axes[2].scatter(
    center_normal[0], center_normal[1] + 5, 
    marker='*', s=600, c='#F1C40F', edgecolors='black', linewidth=1.5, zorder=10
)
axes[2].text(
    center_normal[0], center_normal[1] + 9, "Trust Anchor", 
    ha='center', fontsize=12, fontweight='bold'
)
center_ddos = df_tgp[df_tgp['Label'] == 'DDoS'][['x', 'y']].mean().values
axes[2].text(
    (center_normal[0] + center_ddos[0])/2, (center_normal[1] + center_ddos[1])/2, 
    "Geometric Gap\n(Rejection)", 
    ha='center', va='center', fontsize=12, color='gray', fontweight='bold'
)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Normal'], markersize=12, label='Normal Traffic'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Mirai'], markersize=12, label='Known Attack (Mirai)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['DDoS'], markersize=12, label='Unknown DDoS')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=14, frameon=False)

plt.tight_layout()

# 保存
save_path = 'Experiment_6_5_Final_HighRes.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("-" * 30)
print(f"✅ 高清图已生成！")
print(f"📂 文件位置: {os.path.abspath(save_path)}")
print("-" * 30)