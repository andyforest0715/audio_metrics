import os
import re
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path
import os

# 设置 Hugging Face 缓存路径（指向你的本地目录）
os.environ['HF_HOME'] = r'G:\env\huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'G:\env\huggingface_cache\hub'
os.environ['TRANSFORMERS_CACHE'] = r'G:\env\huggingface_cache\hub'

# 强制离线模式，禁止联网下载
os.environ['HF_HUB_OFFLINE'] = '1'

# ================= 配置区域 =================
# 1. 路径设置
BASE_ROOT = r"G:/0-thesis-project/empathybgm/experiment"
GT_FOLDER = "0_GroundTruth"
TARGET_FOLDERS = [
    "1_Swap_Match",
    "2_Swap_Mismatch",
    "3_Music_Match",
    "4_Music_Mismatch",
    "5_NoDrums",
]

# 2. 环境设置
# os.environ['HF_HOME'] = r"G:/T7shield/env/torch_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 核心评测类 =================
class AdditionalMetricsEngine:
    def __init__(self, device):
        self.device = device
        print(f"[{self.device}] 初始化评测引擎 (无 CLAP)...")

    # -----------------------------------
    # Metric 1: LSD & PC (频谱质量)
    # -----------------------------------
    def compute_spectral_metrics(self, y_gen, y_ref, sr=16000):
        """同时计算 LSD 和 PC"""
        try:
            # 计算 Mel 频谱
            def get_log_mel(y):
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=80)
                return np.log1p(S)

            S_gen = get_log_mel(y_gen)
            S_ref = get_log_mel(y_ref)

            # 对齐长度
            min_len = min(S_gen.shape[1], S_ref.shape[1])
            S_gen = S_gen[:, :min_len]
            S_ref = S_ref[:, :min_len]

            # 1. LSD (越低越好)
            diff = (S_ref - S_gen) ** 2
            lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))

            # 2. PC (越高越好)
            # 展平计算相关性
            pc, _ = pearsonr(S_gen.flatten(), S_ref.flatten())

            return lsd, pc
        except Exception as e:
            print(f"Spectral Error: {e}")
            return None, None

    # -----------------------------------
    # Metric 2: Beat Similarity (节拍一致性)
    # -----------------------------------
    def compute_beat_similarity(self, y_gen, y_ref, sr):
        try:
            # 提取 Onset Strength
            onset_gen = librosa.onset.onset_strength(y=y_gen, sr=sr)
            onset_ref = librosa.onset.onset_strength(y=y_ref, sr=sr)

            # 计算 Tempogram (节奏直方图)
            tempo_gen = np.mean(librosa.feature.tempogram(onset_envelope=onset_gen, sr=sr), axis=1)
            tempo_ref = np.mean(librosa.feature.tempogram(onset_envelope=onset_ref, sr=sr), axis=1)

            # 余弦相似度
            norm_g = np.linalg.norm(tempo_gen)
            norm_r = np.linalg.norm(tempo_ref)
            if norm_g == 0 or norm_r == 0: return 0.0
            
            return np.dot(tempo_gen, tempo_ref) / (norm_g * norm_r)
        except:
            return 0.0

    # -----------------------------------
    # 辅助: 加载音频
    # -----------------------------------
    def load_pair_audio(self, p_gen, p_ref, sr=16000):
        try:
            # Librosa 加载 (CPU numpy)
            yg, _ = librosa.load(p_gen, sr=sr)
            yr, _ = librosa.load(p_ref, sr=sr)
            return yg, yr
        except:
            return None, None

# ================= 2. 主流程 =================
def pair_files_by_group(gt_dir, gen_dir):
    """基于 G0XX 或 GroupXX 配对"""
    def _get_key(path):
        # 新命名: G001_8836dbb1.wav → key = "G001"
        m = re.search(r"(G\d{3})", path.name)
        if m:
            return m.group(1)
        # 旧命名 fallback
        m = re.search(r"(Group\d+)", path.name, re.IGNORECASE)
        return m.group(1).lower() if m else None
    
    gt_files = {_get_key(p): p for p in Path(gt_dir).glob("*") if p.suffix.lower() in {'.wav','.mp3','.flac'}}
    gen_files = {_get_key(p): p for p in Path(gen_dir).glob("*") if p.suffix.lower() in {'.wav','.mp3','.flac'}}
    
    if None in gt_files: del gt_files[None]
    if None in gen_files: del gen_files[None]
    
    keys = sorted(set(gt_files.keys()) & set(gen_files.keys()))
    return [(gt_files[k], gen_files[k]) for k in keys]

def main():
    engine = AdditionalMetricsEngine(DEVICE)
    final_results = []

    gt_full_path = Path(BASE_ROOT) / GT_FOLDER / "audio"
    if not gt_full_path.exists():
        gt_full_path = Path(BASE_ROOT) / GT_FOLDER # 兼容无audio子目录情况

    print(f"\nGT Path: {gt_full_path}")

    for folder_name in TARGET_FOLDERS:
        gen_full_path = Path(BASE_ROOT) / folder_name / "audio"
        if not gen_full_path.exists(): gen_full_path = Path(BASE_ROOT) / folder_name
        
        if not gen_full_path.exists():
            print(f"Skipping {folder_name} (Not found)")
            continue

        print(f"\nProcessing: {folder_name} ...")
        
        # 1. 配对
        pairs = pair_files_by_group(gt_full_path, gen_full_path)
        if len(pairs) == 0:
            print("  No pairs found.")
            continue

        # 2. 逐个计算
        metrics_accum = {
            "LSD": [], "PC": [], "Beat_Sim": []
        }

        for p_ref, p_gen in tqdm(pairs):
            # 信号级指标 (LSD, PC, Beat) - 需要加载波形
            y_gen, y_ref = engine.load_pair_audio(p_gen, p_ref)
            
            if y_gen is not None and y_ref is not None:
                # LSD & PC
                lsd, pc = engine.compute_spectral_metrics(y_gen, y_ref)
                if lsd is not None: 
                    metrics_accum["LSD"].append(lsd)
                    metrics_accum["PC"].append(pc)
                
                # Beat Consistency
                beat_sim = engine.compute_beat_similarity(y_gen, y_ref, sr=16000)
                metrics_accum["Beat_Sim"].append(beat_sim)

        # 3. 统计平均值
        row = {
            "Experiment": folder_name,
            "Count": len(pairs),
            "LSD": np.mean(metrics_accum["LSD"]) if metrics_accum["LSD"] else None,
            "PC": np.mean(metrics_accum["PC"]) if metrics_accum["PC"] else None,
            "Beat_Sim": np.mean(metrics_accum["Beat_Sim"]) if metrics_accum["Beat_Sim"] else None,
        }
        final_results.append(row)
        
        print(f"  -> LSD: {row['LSD']:.4f} | PC: {row['PC']:.4f}")

    # 4. 保存
    if final_results:
        df = pd.DataFrame(final_results)
        save_path = "additional_metrics_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\n✅ 所有指标计算完成！已保存至: {os.path.abspath(save_path)}")
        print(df)

if __name__ == "__main__":
    main()