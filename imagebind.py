import os
import sys
import re
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from pathlib import Path

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional
# ========================================================

# ================= ⚙️ Windows 本地配置 =================
BASE_ROOT = r"/root/autodl-tmp/experiment"

# GT 视频目录（GroundTruth 下的 video 子文件夹）
GT_VIDEO_DIR = os.path.join(BASE_ROOT, "0_GroundTruth", "video")

# 待评测的实验组
TARGET_FOLDERS = [
    "0_GroundTruth",
    "1_Swap_Match",
    "2_Swap_Mismatch",
    "3_Music_Match",
    "4_Music_Mismatch",
    "5_NoDrums",
]

# 模型路径
MODEL_PATH = r"/root/autodl-fs/imagebind/imagebind_huge.pth"

# 结果文件名
RESULT_FILE = "new_imagebind_detailed_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================

# 导入 ImageBind
try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
except ImportError:
    print("❌ 错误: 未找到 'imagebind' 库。")
    sys.exit(1)

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
AUDIO_EXTS = {'.wav', '.mp3', '.flac'}


class ImageBindEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self._load_model()

    def _load_model(self):
        print(f"[{self.device}] ⏳ Loading ImageBind model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ 找不到模型文件: {MODEL_PATH}")

        self.model = imagebind_model.imagebind_huge(pretrained=False)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ ImageBind Model Loaded.")

    def evaluate_pair(self, video_path, audio_path):
        inputs = {}
        try:
            audio_data = data.load_and_transform_audio_data([str(audio_path)], self.device)
            video_data = data.load_and_transform_video_data([str(video_path)], self.device)

            inputs[ModalityType.AUDIO] = audio_data
            inputs[ModalityType.VISION] = video_data

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    embeddings = self.model(inputs)
                    audio_emb = embeddings[ModalityType.AUDIO]
                    video_emb = embeddings[ModalityType.VISION]
                    sim = torch.nn.functional.cosine_similarity(audio_emb, video_emb)
                    score = sim.item()

            del inputs, audio_data, video_data, embeddings
            return score

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"[OOM] 显存不足跳过: {Path(video_path).name}")
                torch.cuda.empty_cache()
                return None
            print(f"[Runtime Error] {e}")
            return None
        except Exception as e:
            print(f"\n❌ [Error] Processing {Path(video_path).name}: {e}")
            return None


# ================= 基于 G001 / GroupXX 配对 =================
def _get_key(path):
    """新命名 G001_xxx 优先；旧命名 GroupXX fallback"""
    m = re.search(r"(G\d{3})", path.name)
    if m:
        return m.group(1)
    m = re.search(r"(Group\d+)", path.name, re.IGNORECASE)
    return m.group(1).lower() if m else None


def pair_files_by_group(video_dir, audio_dir):
    vid_files = {}
    if os.path.exists(video_dir):
        for p in Path(video_dir).glob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                k = _get_key(p)
                if k: vid_files[k] = p
    else:
        print(f"❌ 视频目录不存在: {video_dir}")
        return []

    aud_files = {}
    if os.path.exists(audio_dir):
        for p in Path(audio_dir).glob("*"):
            if p.suffix.lower() in AUDIO_EXTS:
                k = _get_key(p)
                if k: aud_files[k] = p

    common_keys = sorted(set(vid_files.keys()) & set(aud_files.keys()))
    return [(k, vid_files[k], aud_files[k]) for k in common_keys]


def main():
    torch.cuda.empty_cache()
    gc.collect()

    evaluator = ImageBindEvaluator(DEVICE)
    detailed_results = []

    print(f"\n🎥 GT Video Directory: {GT_VIDEO_DIR}")

    for folder_name in TARGET_FOLDERS:
        audio_dir = Path(BASE_ROOT) / folder_name / "audio"
        if not audio_dir.exists():
            audio_dir = Path(BASE_ROOT) / folder_name

        if not audio_dir.exists():
            print(f"⚠️ 跳过 {folder_name}: 路径不存在")
            continue

        print(f"\nProcessing Experiment: {folder_name}")

        pairs = pair_files_by_group(GT_VIDEO_DIR, audio_dir)
        if len(pairs) == 0:
            print("  No matching pairs found.")
            continue

        current_folder_scores = []
        for group_id, vid_path, aud_path in tqdm(pairs, desc=f"Eval {folder_name}"):
            score = evaluator.evaluate_pair(vid_path, aud_path)

            if score is not None:
                detailed_results.append({
                    "Experiment_Folder": folder_name,
                    "Group_ID": group_id,
                    "Video_Filename": vid_path.name,
                    "Audio_Filename": aud_path.name,
                    "Score": score
                })
                current_folder_scores.append(score)

            torch.cuda.empty_cache()

        if current_folder_scores:
            print(f"  -> Avg: {np.mean(current_folder_scores):.4f} (Count: {len(current_folder_scores)})")

    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df = df.sort_values(by=["Experiment_Folder", "Group_ID"])
        df.to_csv(RESULT_FILE, index=False)
        print(f"\n✅ 评测完成！详细结果已保存至: {os.path.abspath(RESULT_FILE)}")

        print("\n数据预览:")
        print(df.head())

        print("\n=== 各组平均分汇总 ===")
        summary = df.groupby("Experiment_Folder")["Score"].mean()
        print(summary)
    else:
        print("\n❌ 未生成任何有效结果。")


if __name__ == "__main__":
    main()
