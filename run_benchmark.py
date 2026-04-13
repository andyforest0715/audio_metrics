import os
import pandas as pd
import torch
from audio_metric import AudioMetricsEngine  # <--- 导入刚才写的模块

# ================= 配置区域 =================
# 1. 基础路径
BASE_ROOT = r"G:/0-thesis-project/empathybgm/experiment"
HF_CACHE = r"G:/env/huggingface_cache"
TORCH_CACHE = r"G:/T7shield/env/torch_cache"

# 2. 文件夹定义
GT_FOLDER = "0_GroundTruth"
TARGET_FOLDERS = [
    "1_Swap_Match",
    "2_Swap_Mismatch",
    "3_Music_Match",
    "4_Music_Mismatch",
    "5_NoDrums",
]

# 3. 设置环境
os.environ['HF_HOME'] = HF_CACHE
os.environ['TORCH_HOME'] = TORCH_CACHE
# PANNs 模型路径
PANNS_PATH = r"G:\env\models\Cnn14_mAP=0.431.pth"
if os.path.exists(PANNS_PATH):
    os.environ['PANNS_PATH'] = PANNS_PATH

def run_evaluation():
    # 1. 初始化引擎 (只加载一次模型，节省时间)
    engine = AudioMetricsEngine(device="cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    # GT 的完整路径
    gt_full_path = os.path.join(BASE_ROOT, GT_FOLDER)
    if not os.path.exists(gt_full_path):
        print(f"❌ 错误: 找不到 Ground Truth 文件夹: {gt_full_path}")
        return

    # 2. 遍历所有实验组
    for folder_name in TARGET_FOLDERS:
        gen_full_path = os.path.join(BASE_ROOT, folder_name)
        
        print(f"\n" + "="*50)
        print(f"正在评测: {folder_name}")
        print(f"对比路径: {gt_full_path} vs {gen_full_path}")
        print("="*50)

        if not os.path.exists(gen_full_path):
            print(f"⚠️ 跳过: 文件夹不存在 {gen_full_path}")
            continue

        # === 核心调用：使用模块计算 ===
        metrics = engine.compute_metrics_for_folder(gt_full_path, gen_full_path)
        
        if metrics:
            # 整理结果
            row = {
                "Experiment_ID": folder_name.split("_")[0], # 提取前面的数字如 '1'
                "Experiment_Name": folder_name,
                "Sample_Count": metrics.get("count", 0),
                "FAD": metrics.get("FAD", None),
                "FD": metrics.get("FD", None),
                "KL": metrics.get("KL", None),
                # PRDC (如果有的话)
                "Precision": metrics.get("precision", None),
                "Recall": metrics.get("recall", None),
                "Density": metrics.get("density", None),
                "Coverage": metrics.get("coverage", None),
            }
            results.append(row)
            print(f"✅ 完成 {folder_name}: FAD = {row['FAD']:.4f}")
        
        # 显存清理
        torch.cuda.empty_cache()

    # 3. 保存结果到 CSV
    if results:
        df = pd.DataFrame(results)
        # 按 Experiment_ID 排序
        df = df.sort_values(by="Experiment_ID")
        
        output_csv = "final_benchmark_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 所有评测结束！结果已保存至: {os.path.abspath(output_csv)}")
        print(df[["Experiment_Name", "FAD", "KL", "Sample_Count"]]) # 打印简略表
    else:
        print("\n❌ 未生成任何有效结果。")

if __name__ == "__main__":
    run_evaluation()
