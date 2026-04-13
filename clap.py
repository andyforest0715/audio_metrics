import os
import re
import torch
import numpy as np
import pandas as pd
import laion_clap
from tqdm import tqdm
from pathlib import Path

# ================= ⚙️ 云服务器配置 =================
BASE_ROOT = "/root/autodl-tmp/experiment"

# 模型缓存路径
HF_CACHE = "/root/.cache/huggingface" 
os.environ['HF_HOME'] = HF_CACHE
os.environ['TORCH_HOME'] = HF_CACHE
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
# 文件夹定义
GT_FOLDER = "0_GroundTruth"
TARGET_FOLDERS = [
    "0_GroundTruth",
    "1_Swap_Match",
    "2_Swap_Mismatch",
    "3_Music_Match",
    "4_Music_Mismatch",
    "5_NoDrums",
]

# CLAP 模型权重路径
CLAP_CKPT_PATH = "/autodl-fs/data/model/music_audioset_epoch_15_esc_90.14.pt"

# 结果文件名
RESULT_FILE = "aesthetic_detailed_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================


class AestheticScorer:
    def __init__(self, device):
        self.device = device
        print(f"[{self.device}] 正在初始化 CLAP 美学评分模型...")
        print(f"📂 缓存目录: {os.environ['HF_HOME']}")

        try:
            self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            self.clap.load_ckpt(ckpt=CLAP_CKPT_PATH)
            self.clap.to(self.device)
            self.clap.eval()
            print("✅ CLAP 模型加载成功。")
            self._init_anchors()
        except Exception as e:
            print(f"\n❌ CLAP 加载失败: {e}")
            print("提示: 请检查 CLAP_CKPT_PATH 路径是否正确。")
            self.clap = None

    def _init_anchors(self):
        """定义美学锚点"""
        positive_prompts = [
            "High quality music", "Beautiful background music", "Soothing melody",
            "Professional recording studio sound", "Cinematic soundtrack"
        ]
        negative_prompts = [
            "Low quality audio", "Noisy and distorted sound", "Annoying screeching sound",
            "Bad recording", "Random noise"
        ]

        with torch.no_grad():
            self.pos_emb = self.clap.get_text_embedding(positive_prompts, use_tensor=True).to(self.device)
            self.neg_emb = self.clap.get_text_embedding(negative_prompts, use_tensor=True).to(self.device)

    def get_score(self, wav_path):
        if self.clap is None: return 0.0
        try:
            with torch.no_grad():
                audio_emb = self.clap.get_audio_embedding_from_filelist(
                    x=[str(wav_path)], use_tensor=True
                ).to(self.device)

                pos_sim = torch.nn.functional.cosine_similarity(audio_emb, self.pos_emb).mean()
                neg_sim = torch.nn.functional.cosine_similarity(audio_emb, self.neg_emb).mean()

                raw_score = pos_sim - neg_sim
                score = (raw_score.item() + 0.1) * 50 + 50
                return max(0, min(100, score))
        except Exception as e:
            print(f"[Warn] 处理出错 {Path(wav_path).name}: {e}")
            return 0.0


# ================= 基于 G001 / GroupXX 配对 =================
def _get_key(path):
    """新命名 G001_xxx 优先；旧命名 GroupXX fallback"""
    m = re.search(r"(G\d{3})", path.name)
    if m:
        return m.group(1)
    m = re.search(r"(Group\d+)", path.name, re.IGNORECASE)
    return m.group(1).lower() if m else None


def list_audio_with_keys(audio_dir):
    """扫描音频目录，返回 [(key, path), ...]"""
    exts = {'.wav', '.mp3', '.flac'}
    results = []
    for p in Path(audio_dir).glob("*"):
        if p.suffix.lower() in exts:
            k = _get_key(p)
            if k: results.append((k, p))
    return sorted(results, key=lambda x: x[0])


def main():
    scorer = AestheticScorer(DEVICE)
    if scorer.clap is None:
        return

    detailed_results = []

    print(f"\n开始批量评测 (Root: {BASE_ROOT})")

    for folder_name in TARGET_FOLDERS:
        gen_path = Path(BASE_ROOT) / folder_name / "audio"
        if not gen_path.exists():
            gen_path = Path(BASE_ROOT) / folder_name

        if not gen_path.exists():
            print(f"⚠️ 跳过 {folder_name} (不存在)")
            continue

        files = list_audio_with_keys(gen_path)

        if len(files) == 0:
            print(f"⚠️ {folder_name}: 无音频文件")
            continue

        print(f"\nProcessing: {folder_name} (Count: {len(files)})")

        current_folder_scores = []
        for group_id, p_gen in tqdm(files, desc=f"Eval {folder_name}"):
            s = scorer.get_score(p_gen)

            detailed_results.append({
                "Experiment_Folder": folder_name,
                "Group_ID": group_id,
                "Audio_Filename": p_gen.name,
                "Aesthetic_Score": s
            })
            current_folder_scores.append(s)

        if current_folder_scores:
            avg_score = np.mean(current_folder_scores)
            print(f"  -> Avg Score: {avg_score:.2f}")

    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df = df.sort_values(by=["Experiment_Folder", "Group_ID"])
        df.to_csv(RESULT_FILE, index=False)
        print(f"\n✅ 详细结果已保存至: {os.path.abspath(RESULT_FILE)}")

        print("\n=== 数据预览 ===")
        print(df.head())

        print("\n=== 各组平均分汇总 ===")
        print(df.groupby("Experiment_Folder")["Aesthetic_Score"].mean())
    else:
        print("\n❌ 未生成结果，请检查路径。")


if __name__ == "__main__":
    main()