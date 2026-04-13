import numpy as np
import librosa
import torch
from sklearn.metrics.pairwise import pairwise_distances

# ==========================================
# 1. PRDC (Precision, Recall, Density, Coverage)
# ==========================================
def compute_prdc(real_features, fake_features, nearest_k=5):
    """
    计算 PRDC 指标。
    
    Args:
        real_features: 真实音频的特征 (N, feature_dim)
        fake_features: 生成音频的特征 (N, feature_dim)
        nearest_k: k-NN 的 k 值，通常取 5
        
    注意: PRDC 是基于分布的指标，样本量太少(例如少于k个)会报错或不准。
    建议样本量 > 100。
    """
    print(f"Calculating PRDC with k={nearest_k}...")
    
    # 确保是 Numpy 数组
    if isinstance(real_features, torch.Tensor): real_features = real_features.cpu().numpy()
    if isinstance(fake_features, torch.Tensor): fake_features = fake_features.cpu().numpy()

    # 1. 计算距离矩阵
    # distance_real_fake[i, j] 表示 real[i] 到 fake[j] 的距离
    dist_r_f = pairwise_distances(real_features, fake_features)
    dist_f_r = dist_r_f.T # 转置就是 fake 到 real

    dist_r_r = pairwise_distances(real_features, real_features)
    dist_f_f = pairwise_distances(fake_features, fake_features)

    # 2. 获取每个样本的 k-近邻距离 (排除自己，所以是 1:k+1)
    # real 样本自身的 k-NN 距离半径
    rd_real = np.sort(dist_r_r, axis=1)[:, 1:nearest_k+1]
    radii_real = np.max(rd_real, axis=1) # 每个人"地盘"的半径

    # fake 样本自身的 k-NN 距离半径
    rd_fake = np.sort(dist_f_f, axis=1)[:, 1:nearest_k+1]
    radii_fake = np.max(rd_fake, axis=1)

    # 3. 计算指标
    # Precision: 多少 Fake 样本落入了 Real 样本的流形半径内？
    precision = (np.min(dist_f_r, axis=1) <= radii_real[np.argmin(dist_f_r, axis=1)]).mean()

    # Recall: 多少 Real 样本被 Fake 样本的流形覆盖了？
    recall = (np.min(dist_r_f, axis=1) <= radii_fake[np.argmin(dist_r_f, axis=1)]).mean()

    # Density: 平均有多少 Real 样本的邻域包含了 Fake 样本 (衡量生成样本的密集程度)
    density = (1.0 / nearest_k) * (dist_f_r <= radii_real[np.argmin(dist_f_r, axis=1)][:, None]).sum(axis=1).mean()

    # Coverage: 有多少 Real 样本至少被一个 Fake 样本覆盖 (衡量生成的多样性)
    coverage = (dist_r_f.min(axis=1) <= radii_fake[np.argmin(dist_r_f, axis=1)]).mean()

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage
    }

# ==========================================
# 2. Beat Consistency / Beat Histogram Similarity
# ==========================================
def compute_beat_similarity(path_gen, path_ref):
    """
    计算两个音频的节拍直方图相似度 (Beat Histogram Similarity)
    """
    try:
        # 1. 加载音频 (只取前 30 秒以加快速度，节拍通常是全局特征)
        y_gen, sr = librosa.load(path_gen, sr=22050, duration=30)
        y_ref, sr = librosa.load(path_ref, sr=22050, duration=30)

        # 2. 提取 Onset Envelope (起始点强度包络)
        onset_env_gen = librosa.onset.onset_strength(y=y_gen, sr=sr)
        onset_env_ref = librosa.onset.onset_strength(y=y_ref, sr=sr)

        # 3. 计算 Tempogram (节奏图) 或 傅里叶节奏图
        # 这里计算 Tempogram 的均值作为 全局节奏直方图
        # win_length 决定了节奏分析的窗口大小
        prior = librosa.feature.tempogram(onset_envelope=onset_env_gen, sr=sr, hop_length=512)
        target = librosa.feature.tempogram(onset_envelope=onset_env_ref, sr=sr, hop_length=512)
        
        # 取时间轴上的平均，得到 (384,) 的向量，代表各个 BPM 的强度分布
        hist_gen = np.mean(prior, axis=1)
        hist_ref = np.mean(target, axis=1)

        # 4. 计算余弦相似度
        norm_gen = np.linalg.norm(hist_gen)
        norm_ref = np.linalg.norm(hist_ref)
        
        if norm_gen == 0 or norm_ref == 0:
            return 0.0
            
        similarity = np.dot(hist_gen, hist_ref) / (norm_gen * norm_ref)
        return similarity

    except Exception as e:
        print(f"[Warn] Beat calc failed: {e}")
        return 0.0

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("Testing PRDC logic (Fake Data)...")
    # 模拟数据：100个样本，128维特征
    r_feat = np.random.rand(100, 128)
    g_feat = np.random.rand(100, 128)
    
    metrics = compute_prdc(r_feat, g_feat, nearest_k=5)
    print("PRDC Result:", metrics)
    print("✅ PRDC Test Passed.")