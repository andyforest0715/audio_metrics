import os
import re
import numpy as np
import torch
import torchaudio
from scipy import linalg
from tqdm import tqdm
from pathlib import Path

# 尝试导入可选库
try:
    from metric_plus import compute_prdc
    HAS_PRDC = True
except ImportError:
    HAS_PRDC = False

class AudioMetricsEngine:
    def __init__(self, device=None, window_sec=2.0, stride_sec=1.0):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.vgg = None
        self.vgg_pp = None
        self.panns = None
        self.debug_printed = False 
        
        self._init_models()

    def _init_models(self):
        print(f"[{self.device}] 正在初始化评测模型...")
        
        # --- 1. Load VGGish ---
        
        try:
            from torchvggish import vggish, vggish_input
            # 手动定义 VGGish 需要的官方权重下载链接字典

            vggish_urls = {

                'vggish': r'file:///G:\env\torch_cache\hub\checkpoints\vggish-10086976.pth',
                'pca': r'file:///G:\env\torch_cache\hub\checkpoints\vggish_pca_params-970ea276.pth'
                # 'vggish': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
                # 'pca': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth'
            }
            self.vgg = vggish.VGGish(urls=vggish_urls, pretrained=True, preprocess=False, postprocess=True).to(self.device).eval()
            self.vgg_pp = vggish_input
            self.vgg.postprocess = True 
            print("  [OK] VGGish Loaded.")
        except Exception as e:
            print(f"  [Error] VGGish Load Failed: {e}")

        # --- 2. Load PANNs ---
        try:
            from panns_inference import AudioTagging
            panns_path = os.environ.get("PANNS_PATH", "")
            if not panns_path:
                # 常见路径 fallback
                for p in [
                    r"G:\env\models\Cnn14_mAP=0.431.pth",
                    "/Volumes/T7shield/env/models/Cnn14_mAP=0.431.pth",
                    os.path.expanduser("~/models/Cnn14_mAP=0.431.pth"),
                ]:
                    if os.path.exists(p):
                        panns_path = p
                        break
            if os.path.exists(panns_path):
                self.panns = AudioTagging(checkpoint_path=panns_path, device=self.device)
                print("  [OK] PANNs Loaded.")
            else:
                print(f"  [Warn] PANNs model not found at {panns_path}")
        except Exception as e:
            print(f"  [Error] PANNs Load Failed: {e}")

    # ================= 数学计算 =================
    @staticmethod
    def _compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean): covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))

    @staticmethod
    def _compute_kl(probs_ref, probs_gen, eps=1e-10):
        p = np.mean(probs_ref, axis=0); q = np.mean(probs_gen, axis=0)
        p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
        return float(np.sum(p/p.sum() * (np.log(p/p.sum()) - np.log(q/q.sum()))))

    # ================= 特征提取 =================
    def _extract_sliding(self, y_full, sr, extract_func):
        win_len = int(self.window_sec * sr)
        stride_len = int(self.stride_sec * sr)
        
        if len(y_full) <= win_len:
            with torch.no_grad():
                t_seg = torch.from_numpy(y_full).float().to(self.device)
                res = extract_func(t_seg)
                return res.squeeze() if res is not None else None
        
        embeddings = []
        for start in range(0, len(y_full), stride_len):
            end = start + win_len
            if end > len(y_full): break
            seg = y_full[start:end]
            with torch.no_grad():
                t_seg = torch.from_numpy(seg).float().to(self.device)
                emb = extract_func(t_seg)
                if emb is not None: embeddings.append(emb)
        
        if not embeddings: return None
        return np.mean(np.stack(embeddings), axis=0).squeeze()

    def get_vggish_feat(self, y_16k_np):
        if self.vgg is None: return None
        
        def _func(wav_t):
            wav_np = wav_t.cpu().numpy()
            
            # 幅度保险
            max_val = np.max(np.abs(wav_np))
            if max_val > 1.0: wav_np = wav_np / max_val
            
            # 预处理
            ex = self.vgg_pp.waveform_to_examples(wav_np, 16000)
            if ex is None or len(ex) == 0: return None
            
            if isinstance(ex, torch.Tensor): ex_t = ex.to(self.device)
            else: ex_t = torch.from_numpy(ex).to(self.device)
            
            if ex_t.dim() == 3: ex_t = ex_t.unsqueeze(1)
            
            # === 前向传播 ===
            # 1. CNN
            features = self.vgg.features(ex_t)
            features = features.transpose(1, 2).transpose(2, 3)
            features = features.contiguous().view(features.size(0), -1)
            
            # 2. FC (12288 -> 128)
            if hasattr(self.vgg, 'embeddings'):
                features = self.vgg.embeddings(features)
            
            # 3. PCA + Quantization (Output: 0-255)
            if hasattr(self.vgg, 'pproc'):
                features = self.vgg.pproc(features)
            
            # === 核心修复: 反量化 (0~255 -> -2.0~2.0) ===
            # VGGish 的标准操作是将 float 映射到 uint8
            # 如果我们检测到数值很大(均值>50)，说明被量化了，需要还原
            if features.mean() > 50.0:
                features = (features.float() / 255.0) * 4.0 - 2.0

            # === DEBUG 打印 ===
            if not self.debug_printed:
                val_min = features.min().item()
                val_max = features.max().item()
                val_mean = features.mean().item()
                print(f"\n[DEBUG VGGish Final] Min: {val_min:.4f} | Max: {val_max:.4f} | Mean: {val_mean:.4f}")
                if val_mean > 10.0:
                    print("⚠️ 警告: 数值依然过大，反量化可能失败！")
                else:
                    print("✅ 成功: 数值已还原至标准分布 (-2.0 ~ 2.0)，FAD 将恢复正常。")
                self.debug_printed = True
                
            return features.cpu().numpy()
            
        return self._extract_sliding(y_16k_np, 16000, _func)

    def get_panns_feat(self, y_32k_np):
        if self.panns is None: return None, None
        win_len = int(self.window_sec * 32000)
        embs, probs = [], []
        
        for start in range(0, len(y_32k_np), int(self.stride_sec * 32000)):
            if start + win_len > len(y_32k_np): break
            seg = y_32k_np[start : start + win_len]
            m = np.max(np.abs(seg))
            if m > 1.0: seg = seg / m
                
            try:
                c, e = self.panns.inference(seg[None, :])
                embs.append(e[0]); probs.append(c[0])
            except: pass
            
        if not embs: return None, None
        return np.mean(embs, axis=0), np.mean(probs, axis=0)

    def load_audio(self, path):
        try:
            y, sr = torchaudio.load(str(path), normalize=True)
            if y.shape[0] > 1: y = torch.mean(y, dim=0, keepdim=True)
            y = y.squeeze()

            peak = torch.abs(y).max()
            if peak > 1.0: y = y / peak
            elif peak == 0: return None

            res = {}
            if sr != 16000: res[16000] = torchaudio.transforms.Resample(sr, 16000)(y).numpy()
            else: res[16000] = y.numpy()
            
            if sr != 32000: res[32000] = torchaudio.transforms.Resample(sr, 32000)(y).numpy()
            else: res[32000] = y.numpy()
            return res
        except Exception as e:
            return None

    def compute_metrics_for_folder(self, ref_dir, gen_dir, verbose=True):
        ref_audio = Path(ref_dir) / "audio" if (Path(ref_dir) / "audio").exists() else Path(ref_dir)
        gen_audio = Path(gen_dir) / "audio" if (Path(gen_dir) / "audio").exists() else Path(gen_dir)

        pairs = self._pair_files(ref_audio, gen_audio)
        if len(pairs) < 2: return None

        vgg_r, vgg_g = [], []
        panns_emb_r, panns_emb_g = [], []
        panns_prob_r, panns_prob_g = [], []

        iterator = tqdm(pairs, desc="Processing") if verbose else pairs
        for _, p_ref, p_gen in iterator:
            wavs_r = self.load_audio(p_ref)
            wavs_g = self.load_audio(p_gen)
            if not wavs_r or not wavs_g: continue

            # VGGish
            fr, fg = self.get_vggish_feat(wavs_r[16000]), self.get_vggish_feat(wavs_g[16000])
            if fr is not None and fg is not None:
                vgg_r.append(fr); vgg_g.append(fg)

            # PANNs
            er, pr = self.get_panns_feat(wavs_r[32000])
            eg, pg = self.get_panns_feat(wavs_g[32000])
            if er is not None and eg is not None:
                panns_emb_r.append(er); panns_emb_g.append(eg)
                panns_prob_r.append(pr); panns_prob_g.append(pg)

        metrics = {"count": len(vgg_r)}
        
        # FAD
        if len(vgg_r) > 2:
            try:
                fr_np = np.array(vgg_r).reshape(len(vgg_r), -1)
                fg_np = np.array(vgg_g).reshape(len(vgg_g), -1)
                mu_r, sig_r = np.mean(fr_np, axis=0), np.cov(fr_np, rowvar=False)
                mu_g, sig_g = np.mean(fg_np, axis=0), np.cov(fg_np, rowvar=False)
                metrics["FAD"] = self._compute_frechet_distance(mu_r, sig_r, mu_g, sig_g)
                if HAS_PRDC and len(vgg_r) > 5:
                    metrics.update(compute_prdc(fr_np, fg_np, nearest_k=5))
            except: pass

        # PANNs Metrics
        if len(panns_emb_r) > 2:
            try:
                er_np = np.array(panns_emb_r).reshape(len(panns_emb_r), -1)
                eg_np = np.array(panns_emb_g).reshape(len(panns_emb_g), -1)
                mu_r, sig_r = np.mean(er_np, axis=0), np.cov(er_np, rowvar=False)
                mu_g, sig_g = np.mean(eg_np, axis=0), np.cov(eg_np, rowvar=False)
                metrics["FD"] = self._compute_frechet_distance(mu_r, sig_r, mu_g, sig_g)
                metrics["KL"] = self._compute_kl(np.stack(panns_prob_r), np.stack(panns_prob_g))
            except: pass

        return metrics

    def _pair_files(self, dir1, dir2):
        def _get_key(path):
            # 新命名: G001_8836dbb1.wav → key = "G001"
            # 旧命名: Group01_V_0001.wav → key = "group01"
            m = re.search(r"(G\d{3})", path.name)
            if m:
                return m.group(1)
            m = re.search(r"(Group\d+)", path.name, re.IGNORECASE)
            return m.group(1).lower() if m else None
        f1 = {_get_key(p): p for p in Path(dir1).glob("*") if p.suffix.lower() in {'.wav','.mp3','.flac'}}
        f2 = {_get_key(p): p for p in Path(dir2).glob("*") if p.suffix.lower() in {'.wav','.mp3','.flac'}}
        if None in f1: del f1[None]
        if None in f2: del f2[None]
        keys = sorted(set(f1.keys()) & set(f2.keys()))
        return [(k, f1[k], f2[k]) for k in keys]