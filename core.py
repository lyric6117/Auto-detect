"""核心筛选逻辑（GPU预处理优化版 - 完整版）"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')


# ⚡ GPU预处理Dataset
class ImageDatasetGPU(Dataset):
    """GPU加速的图像加载（最小CPU预处理）"""
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, path, True
        except Exception as e:
            return torch.zeros(3, 224, 224), path, False


class ImageDataset(Dataset):
    """标准Dataset（CPU预处理）"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, path, True
        except Exception as e:
            return torch.zeros(3, 518, 518), path, False


class GPUPreprocessor(nn.Module):
    """⚡ 在GPU上做resize和normalize"""
    def __init__(self, image_size=518):
        super().__init__()
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                         mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return x


class PrefetchLoader:
    """异步预加载到GPU"""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True

        for next_data in self.loader:
            with torch.cuda.stream(self.stream):
                next_data = [d.cuda(non_blocking=True) if isinstance(d, torch.Tensor) else d
                            for d in next_data]

            if not first:
                yield current_data
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            current_data = next_data

        yield current_data

    def __len__(self):
        return len(self.loader)


class AnomalyScreener:
    """异常样本筛选器（GPU预处理优化版）"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print("="*70)
        print("🚀 异常样本筛选系统（GPU预处理优化版）")
        print("="*70)

        # ⚡ GPU预处理器
        self.use_gpu_preprocessing = getattr(config, 'USE_GPU_PREPROCESSING', True)

        if self.use_gpu_preprocessing:
            print("⚡ GPU预处理: 已启用")
            self.gpu_preprocessor = GPUPreprocessor(config.IMAGE_SIZE).to(self.device)
            self.gpu_preprocessor.eval()
            self.transform = None
        else:
            print("⚙️  CPU预处理")
            self.transform = self._get_dinov3_transform()
            self.gpu_preprocessor = None

        self.use_amp = getattr(config, 'USE_AMP', True) and self.device.type == 'cuda'

        # 缓存
        self.cache_dir = Path(getattr(config, 'CACHE_DIR', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)

        # 加载模型
        self.model = self._load_dinov3()

        # 数据
        self.all_features = None
        self.all_paths = None
        self.pca = None

    def _get_dinov3_transform(self):
        """CPU预处理"""
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_dinov3(self):
        """加载DINOv3"""
        print(f"\n📥 加载DINOv3模型: {self.config.MODEL_NAME}")

        repo_root = self.config.DINOV3_REPO_ROOT
        sys.path.insert(0, repo_root)

        import dinov3.distributed as distributed
        from dinov3.configs import setup_config, DinoV3SetupArgs, setup_job
        from dinov3.models import build_model_for_eval

        config_path = os.path.join(
            repo_root, "dinov3", "configs", "train", f"{self.config.MODEL_NAME}.yaml"
        )

        output_dir = "./outputs/dinov3_tmp"
        os.makedirs(output_dir, exist_ok=True)

        setup_job(
            output_dir=output_dir,
            distributed_enabled=False,
            seed=42,
            distributed_timeout=datetime.timedelta(minutes=30)
        )

        setup_args = DinoV3SetupArgs(
            config_file=config_path,
            pretrained_weights=self.config.DINOV3_WEIGHT_PATH,
            output_dir=output_dir,
            opts=[]
        )

        cfg = setup_config(setup_args, strict_cfg=False)
        model = build_model_for_eval(config=cfg, pretrained_weights=self.config.DINOV3_WEIGHT_PATH)

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False

        print(f"✅ DINOv3模型加载完成")
        return model

    def _get_image_paths(self, directory):
        """获取图像路径"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        paths = []
        directory = Path(directory)
        for ext in extensions:
            paths.extend(directory.rglob(f"*{ext}"))
        return sorted([str(p) for p in set(paths)])

    def _get_cache_key(self, image_dir):
        """缓存key"""
        key_str = f"{image_dir}_{self.config.MODEL_NAME}_{self.config.IMAGE_SIZE}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    @torch.no_grad()
    def extract_all_features(self, image_dir=None, use_cache=True):
        """⚡ GPU预处理版特征提取"""
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR

        print("\n" + "="*70)
        print("🔍 提取图像特征（GPU预处理优化）")
        print("="*70)

        # 缓存检查
        cache_key = self._get_cache_key(image_dir)
        cache_file = self.cache_dir / f"features_{cache_key}.pkl"

        if use_cache and cache_file.exists():
            print(f"📦 加载缓存: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.all_features = cache_data['features']
            self.all_paths = cache_data['paths']
            print(f"✅ 缓存加载完成: {self.all_features.shape}")
            return self.all_features, self.all_paths

        # 获取图像
        image_paths = self._get_image_paths(image_dir)
        print(f"📂 找到图像: {len(image_paths)} 张")

        # 选择Dataset
        if self.use_gpu_preprocessing:
            dataset = ImageDatasetGPU(image_paths)
            print("⚡ 使用GPU预处理Dataset")
        else:
            dataset = ImageDataset(image_paths, self.transform)
            print("⚙️  使用CPU预处理Dataset")

        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=getattr(self.config, 'PREFETCH_FACTOR', 4),
            persistent_workers=True if self.config.NUM_WORKERS > 0 else False,
            drop_last=False,
        )

        # 异步加载
        if self.use_gpu_preprocessing:
            dataloader = PrefetchLoader(dataloader, self.device)
            print("⚡ 使用异步GPU预加载")

        print(f"⚡ 配置: BS={self.config.BATCH_SIZE}, Workers={self.config.NUM_WORKERS}")

        # 提取特征
        features_list = []
        valid_paths = []
        use_cls_token = getattr(self.config, 'USE_CLS_TOKEN', True)

        import time
        total_images = 0
        start_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            for batch_tensors, batch_paths, batch_valid in tqdm(dataloader, desc="提取特征"):

                if isinstance(batch_valid, torch.Tensor):
                    valid_mask = batch_valid.cpu().numpy()
                else:
                    valid_mask = np.array(batch_valid)

                if not valid_mask.any():
                    continue

                # GPU预处理
                if self.use_gpu_preprocessing:
                    if not batch_tensors.is_cuda:
                        batch_tensors = batch_tensors.cuda(non_blocking=True)
                    batch_tensors = batch_tensors[valid_mask]
                    batch_tensors = self.gpu_preprocessor(batch_tensors)
                else:
                    batch_tensors = batch_tensors[valid_mask].to(self.device, non_blocking=True)

                batch_paths_valid = [p for p, v in zip(batch_paths, valid_mask) if v]

                # 特征提取
                feats_dict = self.model.forward_features(batch_tensors)

                if use_cls_token:
                    if 'x_norm_clstoken' in feats_dict:
                        feats = feats_dict['x_norm_clstoken']
                    elif 'x_prenorm' in feats_dict:
                        feats = feats_dict['x_prenorm'][:, 0]
                    else:
                        feats = list(feats_dict.values())[0]
                        if feats.dim() == 3:
                            feats = feats[:, 0]
                else:
                    if 'x_norm_patchtokens' in feats_dict:
                        feats = feats_dict['x_norm_patchtokens'].mean(dim=1)
                    elif 'x_prenorm' in feats_dict:
                        feats = feats_dict['x_prenorm'][:, 1:].mean(dim=1)
                    else:
                        feats = list(feats_dict.values())[0]
                        if feats.dim() == 3:
                            feats = feats.mean(dim=1)

                if getattr(self.config, 'NORMALIZE_FEATURES', True):
                    feats = F.normalize(feats, dim=-1)

                features_list.append(feats.cpu().half().numpy())
                valid_paths.extend(batch_paths_valid)
                total_images += len(batch_paths_valid)

        elapsed = time.time() - start_time
        print(f"\n⏱️  提取耗时: {elapsed:.1f}秒")
        print(f"📊 速度: {total_images/elapsed:.1f} 张/秒")
        print(f"💾 显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

        # 合并特征
        features = np.vstack(features_list).astype(np.float32)
        self.all_paths = valid_paths
        print(f"✅ 特征提取完成: {features.shape}")

        # PCA
        if getattr(self.config, 'USE_PCA', True):
            pca_dim = min(self.config.PCA_DIM, features.shape[1], len(features) - 1)
            print(f"📉 PCA降维: {features.shape[1]} → {pca_dim}")

            if len(features) > 10000:
                self.pca = IncrementalPCA(n_components=pca_dim, batch_size=1000)
            else:
                self.pca = PCA(n_components=pca_dim, random_state=42)

            features = self.pca.fit_transform(features)

        self.all_features = features.astype(np.float32)
        print(f"📊 最终特征: {self.all_features.shape}")

        # 保存缓存
        if use_cache:
            print(f"💾 保存缓存...")
            with open(cache_file, 'wb') as f:
                pickle.dump({'features': self.all_features, 'paths': self.all_paths}, f)

        return self.all_features, self.all_paths

    def compute_anomaly_scores(self):
        """计算异常分数"""
        print("\n" + "="*70)
        print("💯 计算异常分数")
        print("="*70)

        scores_dict = {}

        # KNN
        if 'knn' in self.config.METHODS:
            print("📊 KNN...")
            k = min(10, len(self.all_features) - 1)
            nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=-1)
            nbrs.fit(self.all_features)
            distances, _ = nbrs.kneighbors(self.all_features)
            scores_dict['knn'] = distances[:, 1:].mean(axis=1)

        # Isolation Forest
        if 'isolation' in self.config.METHODS:
            print("📊 Isolation Forest...")
            iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            iso.fit(self.all_features)
            scores_dict['isolation'] = -iso.score_samples(self.all_features)

        # 融合
        ensemble = np.mean([np.argsort(np.argsort(s))/(len(s)-1) for s in scores_dict.values()], axis=0)
        scores_dict['ensemble'] = ensemble

        return scores_dict

    def select_samples_for_annotation(self, scores):
        """筛选样本"""
        n_select = self.config.TOP_K if self.config.USE_TOP_K else int(len(scores) * self.config.TOP_PERCENT)
        return np.argsort(scores)[-n_select:][::-1]

    def save_results(self, scores_dict, selected_indices):
        """保存结果（完整版）"""
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("💾 保存结果")
        print("="*70)

        # 1. 完整CSV
        df_all = pd.DataFrame({
            'image_path': self.all_paths,
            'image_name': [Path(p).name for p in self.all_paths],
            'ensemble_score': scores_dict['ensemble'],
        })

        for name, scores in scores_dict.items():
            if name != 'ensemble':
                df_all[f'{name}_score'] = scores

        df_all['selected_for_annotation'] = 0
        df_all.loc[selected_indices, 'selected_for_annotation'] = 1
        df_all = df_all.sort_values('ensemble_score', ascending=False)

        csv_all = output_dir / 'all_images_with_scores.csv'
        df_all.to_csv(csv_all, index=False, encoding='utf-8-sig')
        print(f"📄 完整结果: {csv_all}")

        # 2. 筛选样本CSV
        df_selected = df_all[df_all['selected_for_annotation'] == 1].copy()
        df_selected['annotation_label'] = ''
        df_selected['annotation_notes'] = ''

        csv_selected = output_dir / 'samples_for_annotation.csv'
        df_selected.to_csv(csv_selected, index=False, encoding='utf-8-sig')
        print(f"📄 待标注: {csv_selected} ({len(df_selected)} 个)")

        # 3. 摘要
        summary = {
            '总图像数': len(self.all_paths),
            '筛选数量': len(selected_indices),
            '筛选比例': f"{len(selected_indices)/len(self.all_paths)*100:.2f}%",
            '分数均值': f"{scores_dict['ensemble'].mean():.4f}",
            '分数标准差': f"{scores_dict['ensemble'].std():.4f}",
            '选中样本最低分': f"{scores_dict['ensemble'][selected_indices].min():.4f}",
            '选中样本最高分': f"{scores_dict['ensemble'][selected_indices].max():.4f}",
        }

        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("异常样本筛选摘要\n")
            f.write("="*70 + "\n\n")
            for k, v in summary.items():
                f.write(f"{k:20s}: {v}\n")
        print(f"📄 摘要: {summary_file}")

        # 4. ✅ 可视化
        if getattr(self.config, 'SAVE_VISUALIZATIONS', True):
            self._visualize_results(scores_dict, selected_indices, output_dir)

        print(f"\n📁 所有结果已保存到: {output_dir.absolute()}")
        return csv_selected

    def _visualize_results(self, scores_dict, selected_indices, output_dir):
        """可视化结果"""
        print("\n📊 生成可视化...")

        scores = scores_dict['ensemble']

        # 1. 分数分布
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        selected_scores = scores[selected_indices]
        threshold = selected_scores.min()
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Score Distribution (All)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(selected_scores, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Score Distribution (Selected)', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 分数分布图")

        # 2. ✅ Top异常样本
        viz_k = min(getattr(self.config, 'VIZ_TOP_K', 100), len(selected_indices))
        print(f"   生成Top-{viz_k}异常样本图...")

        top_indices = selected_indices[:viz_k]
        n_cols = getattr(self.config, 'GRID_COLS', 5)
        n_rows = (viz_k + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            if i >= len(axes):
                break

            img_path = self.all_paths[idx]
            score = scores[idx]

            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
            except:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            axes[i].imshow(img)
            axes[i].set_title(f'#{i+1} | Score: {score:.4f}\n{Path(img_path).name}',
                            fontsize=8)
            axes[i].axis('off')

        for i in range(len(top_indices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        viz_file = output_dir / f'top_{viz_k}_anomalies.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Top-{viz_k}异常样本图: {viz_file}")
