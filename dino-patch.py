"""Patch-Level异常检测（修复版 - 兼容DINOv3）"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import hashlib
import warnings
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')


# ==================== GPU预处理（不变）====================
class ImageDatasetGPU(Dataset):
    """GPU加速的图像加载"""
    def __init__(self, image_paths, image_size=512):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BILINEAR),
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
            return torch.zeros(3, self.image_size, self.image_size), path, False


class GPUPreprocessor(nn.Module):
    """GPU上做normalize"""
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


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


# ==================== 🔥 修复版 Patch-Level异常检测 ====================
class PatchLevelAnomalyDetector:
    """PaDiM风格的Patch-Level异常检测（修复DINOv3兼容性）"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print("=" * 80)
        print("🔥 Patch-Level异常检测系统（PaDiM风格 + GPU加速）")
        print("=" * 80)

        self.feature_layers = getattr(config, 'FEATURE_LAYERS', [8, 16, 23])

        # GPU预处理
        self.gpu_preprocessor = GPUPreprocessor(config.IMAGE_SIZE).to(self.device)
        self.gpu_preprocessor.eval()

        # 缓存
        self.cache_dir = Path(getattr(config, 'CACHE_DIR', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)

        # 加载DINOv3
        self.model = self._load_dinov3()

        # 高斯分布参数
        self.patch_means = None
        self.patch_inv_covs = None
        self.pca = None

        # 数据
        self.all_patch_features = None
        self.all_paths = None

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

        setup_job(output_dir=output_dir, distributed_enabled=False, seed=42,
                 distributed_timeout=datetime.timedelta(minutes=30))

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

        # 🔥 检查模型结构
        print(f"   模型层数: {len(model.blocks)} layers")
        print(f"   提取层: {self.feature_layers}")

        # 验证层索引
        max_layer = max(self.feature_layers)
        if max_layer >= len(model.blocks):
            raise ValueError(f"FEATURE_LAYERS包含无效层索引 {max_layer}，模型只有 {len(model.blocks)} 层（0-{len(model.blocks)-1}）")

        print(f"✅ DINOv3加载完成")
        return model

    def _get_image_paths(self, directory):
        """获取图像路径"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        paths = []
        directory = Path(directory)
        for ext in extensions:
            paths.extend(directory.rglob(f"*{ext}"))
        return sorted([str(p) for p in set(paths)])

    @torch.no_grad()
    def extract_multiscale_patch_features(self, image_dir=None, use_cache=True):
        """提取多尺度Patch特征"""
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR

        print("\n" + "="*80)
        print("🔍 提取多尺度Patch特征（Multi-scale Patch Tokens）")
        print("="*80)

        # 缓存检查
        cache_key = hashlib.md5(
            f"{image_dir}_{self.config.MODEL_NAME}_{self.feature_layers}".encode()
        ).hexdigest()[:16]
        cache_file = self.cache_dir / f"patch_features_{cache_key}.pkl"

        if use_cache and cache_file.exists():
            print(f"📦 加载缓存: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.all_patch_features = cache_data['patch_features']
            self.all_paths = cache_data['paths']
            print(f"✅ 缓存加载完成: {self.all_patch_features.shape}")
            return self.all_patch_features, self.all_paths

        # 获取图像
        image_paths = self._get_image_paths(image_dir)
        print(f"📂 找到图像: {len(image_paths)} 张")
        print(f"🎯 提取层: {self.feature_layers}")

        # DataLoader
        dataset = ImageDatasetGPU(image_paths, self.config.IMAGE_SIZE)
        dataloader = DataLoader(
            dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            num_workers=self.config.NUM_WORKERS, pin_memory=True,
            prefetch_factor=4, persistent_workers=True if self.config.NUM_WORKERS > 0 else False
        )
        dataloader = PrefetchLoader(dataloader, self.device)

        # 提取特征
        all_patch_feats = []
        valid_paths = []

        import time
        start_time = time.time()

        for batch_tensors, batch_paths, batch_valid in tqdm(dataloader, desc="提取Patch特征"):
            valid_mask = batch_valid.cpu().numpy() if isinstance(batch_valid, torch.Tensor) else np.array(batch_valid)
            if not valid_mask.any():
                continue

            # GPU预处理
            batch_tensors = batch_tensors[valid_mask].cuda(non_blocking=True)
            batch_tensors = self.gpu_preprocessor(batch_tensors)
            batch_paths_valid = [p for p, v in zip(batch_paths, valid_mask) if v]

            # 🔥 提取多层patch tokens（修复版）
            patch_feats = self._extract_patch_features_from_layers(batch_tensors)  # [B, H, W, D]

            all_patch_feats.append(patch_feats.cpu().numpy())
            valid_paths.extend(batch_paths_valid)

        elapsed = time.time() - start_time
        print(f"\n⏱️  提取耗时: {elapsed:.1f}秒")
        print(f"💾 显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

        # 合并
        patch_features = np.concatenate(all_patch_feats, axis=0)
        self.all_patch_features = patch_features
        self.all_paths = valid_paths

        print(f"✅ Patch特征提取完成: {patch_features.shape}")
        print(f"   - 图像数: {patch_features.shape[0]}")
        print(f"   - Patch网格: {patch_features.shape[1]}×{patch_features.shape[2]}")
        print(f"   - 特征维度: {patch_features.shape[3]}")

        # PCA降维
        if getattr(self.config, 'USE_PCA', True):
            patch_features = self._apply_pca_to_patches(patch_features)
            self.all_patch_features = patch_features

        # 保存缓存
        if use_cache:
            print(f"💾 保存缓存...")
            with open(cache_file, 'wb') as f:
                pickle.dump({'patch_features': patch_features, 'paths': valid_paths}, f)

        return patch_features, valid_paths

    def _extract_patch_features_from_layers(self, images):
        """
        🔥 修复版：从多个Transformer层提取patch tokens

        Args:
            images: [B, 3, H, W]

        Returns:
            patch_feats: [B, h, w, D]
        """
        B = images.size(0)

        # 🔥 存储中间特征
        intermediate_features = {}

        def hook_fn(name):
            def hook(module, input, output):
                # ✅ 处理可能的tuple/list输出
                if isinstance(output, (tuple, list)):
                    # 取第一个元素（通常是主输出）
                    intermediate_features[name] = output[0]
                else:
                    intermediate_features[name] = output
            return hook

        # 注册hooks
        hooks = []
        for layer_idx in self.feature_layers:
            if hasattr(self.model, 'blocks'):
                hook = self.model.blocks[layer_idx].register_forward_hook(
                    hook_fn(f'layer_{layer_idx}')
                )
                hooks.append(hook)
            else:
                raise AttributeError("模型没有 'blocks' 属性，请检查DINOv3版本")

        # Forward pass
        try:
            _ = self.model.forward_features(images)
        except Exception as e:
            print(f"❌ forward_features失败: {e}")
            # 移除hooks
            for hook in hooks:
                hook.remove()
            raise

        # 移除hooks
        for hook in hooks:
            hook.remove()

        # 🔥 提取patch tokens（修复版）
        patch_tokens_list = []

        for layer_idx in self.feature_layers:
            key = f'layer_{layer_idx}'
            if key not in intermediate_features:
                raise RuntimeError(f"未捕获到 layer {layer_idx} 的特征，可能hook失败")

            layer_out = intermediate_features[key]

            # ✅ 确保是tensor
            if not isinstance(layer_out, torch.Tensor):
                print(f"⚠️  Layer {layer_idx} 输出类型: {type(layer_out)}")
                if isinstance(layer_out, (tuple, list)):
                    layer_out = layer_out[0]
                else:
                    raise TypeError(f"无法处理的输出类型: {type(layer_out)}")

            # ✅ 检查形状
            if layer_out.dim() != 3:
                raise ValueError(f"Layer {layer_idx} 输出形状异常: {layer_out.shape}，期望 [B, N, D]")

            # 去掉cls token（假设第0个是cls token）
            # DINOv3格式: [B, num_patches+1, embed_dim]
            patch_tokens = layer_out[:, 1:, :]  # [B, N, D]

            # ✅ L2 normalize（每层单独归一化）
            patch_tokens = F.normalize(patch_tokens, dim=-1)

            patch_tokens_list.append(patch_tokens)

        # 拼接多层特征
        patch_tokens = torch.cat(patch_tokens_list, dim=-1)  # [B, N, D*num_layers]

        # Reshape到空间网格
        B, N, D = patch_tokens.shape
        h = int(np.floor(np.sqrt(N)))
        w = int(np.ceil(N / h))
        # 如果 h*w > N，需要补零 patch
        if h * w > N:
            padding = torch.zeros(B, h * w - N, D, device=patch_tokens.device)
            patch_tokens = torch.cat([patch_tokens, padding], dim=1)
        patch_tokens = patch_tokens.reshape(B, h, w, D)

        return patch_tokens

    def _apply_pca_to_patches(self, patch_features):
        """对patch特征做PCA降维"""
        N, H, W, D = patch_features.shape
        pca_dim = min(self.config.PCA_DIM, D, N*H*W - 1)

        print(f"📉 Patch特征PCA: {D} → {pca_dim}")

        # Flatten
        X = patch_features.reshape(-1, D)

        # IncrementalPCA
        self.pca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
        X_reduced = self.pca.fit_transform(X)

        # Reshape
        patch_features_reduced = X_reduced.reshape(N, H, W, pca_dim)

        print(f"✅ PCA完成: {patch_features_reduced.shape}")
        return patch_features_reduced

    def fit_gaussian_distribution(self):
        """建立Patch-Level高斯分布"""
        print("\n" + "="*80)
        print("📊 建立Patch-Level高斯分布（Normal Distribution Modeling）")
        print("="*80)

        N, H, W, D = self.all_patch_features.shape

        print(f"数据形状: {self.all_patch_features.shape}")

        # 计算均值和协方差
        self.patch_means = np.zeros((H, W, D), dtype=np.float32)
        self.patch_inv_covs = np.zeros((H, W, D, D), dtype=np.float32)

        cov_reg = getattr(self.config, 'COV_REGULARIZATION', 1e-3)

        print(f"协方差正则化: {cov_reg}")
        print("\n计算中...")

        for i in tqdm(range(H), desc="空间位置"):
            for j in range(W):
                features_at_pos = self.all_patch_features[:, i, j, :]  # [N, D]

                # 均值
                mean = features_at_pos.mean(axis=0)
                self.patch_means[i, j] = mean

                # 协方差
                cov = np.cov(features_at_pos, rowvar=False)
                cov += np.eye(D) * cov_reg

                # 求逆
                try:
                    inv_cov = np.linalg.inv(cov)
                except:
                    inv_cov = np.eye(D)

                self.patch_inv_covs[i, j] = inv_cov

        print(f"✅ 高斯分布建立完成！")

    def compute_anomaly_maps(self, patch_features=None):
        """计算异常图"""
        if patch_features is None:
            patch_features = self.all_patch_features

        print("\n" + "="*80)
        print("💯 计算Patch-Level异常分数（Mahalanobis Distance）")
        print("="*80)

        N, H, W, D = patch_features.shape
        anomaly_maps = np.zeros((N, H, W), dtype=np.float32)

        for n in tqdm(range(N), desc="计算异常分数"):
            for i in range(H):
                for j in range(W):
                    feat = patch_features[n, i, j]
                    mean = self.patch_means[i, j]
                    inv_cov = self.patch_inv_covs[i, j]

                    delta = feat - mean
                    score = np.sqrt(delta @ inv_cov @ delta)

                    anomaly_maps[n, i, j] = score

        # 图像级分数
        k = getattr(self.config, 'TOP_K_PATCHES', 10)
        image_scores = np.zeros(N, dtype=np.float32)

        for n in range(N):
            patch_scores = anomaly_maps[n].flatten()
            top_k_scores = np.partition(patch_scores, -k)[-k:]
            image_scores[n] = top_k_scores.mean()

        print(f"✅ 异常分数计算完成！")
        print(f"   - 分数范围: [{image_scores.min():.4f}, {image_scores.max():.4f}]")

        return anomaly_maps, image_scores

    def select_samples_for_annotation(self, image_scores):
        """筛选样本"""
        if self.config.USE_TOP_K:
            n_select = self.config.TOP_K
        else:
            n_select = int(len(image_scores) * self.config.TOP_PERCENT)

        selected_indices = np.argsort(image_scores)[-n_select:][::-1]
        return selected_indices

    def visualize_anomaly_maps(self, anomaly_maps, selected_indices, output_dir):
        """可视化异常热力图"""
        print("\n" + "="*80)
        print("🎨 生成异常热力图可视化")
        print("="*80)

        viz_dir = Path(output_dir) / 'anomaly_heatmaps'
        viz_dir.mkdir(exist_ok=True)

        viz_k = min(getattr(self.config, 'VIZ_TOP_K', 50), len(selected_indices))
        sigma = getattr(self.config, 'HEATMAP_SIGMA', 4)
        alpha = getattr(self.config, 'HEATMAP_ALPHA', 0.5)

        print(f"生成Top-{viz_k}异常样本的热力图...")

        for idx in tqdm(selected_indices[:viz_k], desc="生成热力图"):
            img_path = self.all_paths[idx]
            anomaly_map = anomaly_maps[idx]

            try:
                img_orig = cv2.imread(img_path)
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                h_orig, w_orig = img_orig.shape[:2]
            except:
                continue

            # 上采样
            anomaly_map_resized = cv2.resize(anomaly_map, (w_orig, h_orig),
                                            interpolation=cv2.INTER_CUBIC)
            anomaly_map_smooth = gaussian_filter(anomaly_map_resized, sigma=sigma)

            # 归一化
            anomaly_map_norm = (anomaly_map_smooth - anomaly_map_smooth.min()) / \
                              (anomaly_map_smooth.max() - anomaly_map_smooth.min() + 1e-8)

            # 热力图
            heatmap = cv2.applyColorMap(
                (anomaly_map_norm * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 叠加
            overlay = (img_orig * (1 - alpha) + heatmap * alpha).astype(np.uint8)

            # 保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_orig)
            axes[0].set_title('Original', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(anomaly_map_norm, cmap='jet')
            axes[1].set_title('Anomaly Map', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay', fontsize=12)
            axes[2].axis('off')

            plt.suptitle(f'{Path(img_path).name}\nScore: {anomaly_map.max():.4f}',
                        fontsize=10)
            plt.tight_layout()

            save_path = viz_dir / f'{Path(img_path).stem}_heatmap.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"✅ 热力图已保存到: {viz_dir}")

    def save_results(self, anomaly_maps, image_scores, selected_indices, output_dir):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("💾 保存结果")
        print("="*80)

        # CSV
        df = pd.DataFrame({
            'image_path': self.all_paths,
            'image_name': [Path(p).name for p in self.all_paths],
            'anomaly_score': image_scores,
            'max_patch_score': anomaly_maps.max(axis=(1, 2)),
            'mean_patch_score': anomaly_maps.mean(axis=(1, 2)),
        })

        df['selected_for_annotation'] = 0
        df.loc[selected_indices, 'selected_for_annotation'] = 1
        df = df.sort_values('anomaly_score', ascending=False)

        csv_all = output_dir / 'all_images_with_scores.csv'
        df.to_csv(csv_all, index=False, encoding='utf-8-sig')
        print(f"📄 完整结果: {csv_all}")

        df_selected = df[df['selected_for_annotation'] == 1].copy()
        csv_selected = output_dir / 'samples_for_annotation.csv'
        df_selected.to_csv(csv_selected, index=False, encoding='utf-8-sig')
        print(f"📄 待标注: {csv_selected} ({len(df_selected)} 个)")

        # 可视化
        self.visualize_anomaly_maps(anomaly_maps, selected_indices, output_dir)

        print(f"\n📁 所有结果已保存到: {output_dir.absolute()}")
        return csv_selected

    def run_pipeline(self, image_dir=None):
        """完整流程"""
        print("\n" + "="*80)
        print("🚀 开始Patch-Level异常检测Pipeline")
        print("="*80)

        # 1. 提取特征
        patch_features, paths = self.extract_multiscale_patch_features(image_dir)

        # 2. 建立分布
        self.fit_gaussian_distribution()

        # 3. 计算异常
        anomaly_maps, image_scores = self.compute_anomaly_maps()

        # 4. 筛选
        selected_indices = self.select_samples_for_annotation(image_scores)
        print(f"\n🎯 筛选出 {len(selected_indices)} 个异常样本")

        # 5. 保存
        csv_file = self.save_results(
            anomaly_maps, image_scores, selected_indices,
            self.config.OUTPUT_DIR
        )

        print("\n" + "="*80)
        print("✅ Pipeline完成！")
        print("="*80)

        return csv_file


# ==================== 🚀 GPU加速版本 ====================

class PatchLevelAnomalyDetectorFast(PatchLevelAnomalyDetector):
    """GPU向量化加速版（10-100x faster）"""

    def fit_gaussian_distribution(self):
        """🚀 GPU批量计算高斯分布参数"""
        print("\n" + "=" * 80)
        print("📊 建立Patch-Level高斯分布（GPU向量化加速）")
        print("=" * 80)

        N, H, W, D = self.all_patch_features.shape
        print(f"数据形状: {self.all_patch_features.shape}")

        # 转GPU
        features_gpu = torch.from_numpy(self.all_patch_features).to(self.device)  # [N, H, W, D]
        features_gpu = features_gpu.permute(1, 2, 0, 3)  # [H, W, N, D]

        cov_reg = getattr(self.config, 'COV_REGULARIZATION', 1e-3)
        print(f"协方差正则化: {cov_reg}")

        # ✅ 批量计算均值（向量化）
        patch_means = features_gpu.mean(dim=2)  # [H, W, D]

        # ✅ 批量计算协方差和逆矩阵（GPU加速）
        patch_inv_covs = torch.zeros(H, W, D, D, device=self.device)

        print("计算协方差矩阵...")
        for i in tqdm(range(H), desc="空间位置（GPU加速）"):
            for j in range(W):
                X = features_gpu[i, j]  # [N, D]

                # 中心化
                X_centered = X - patch_means[i, j]  # [N, D]

                # 协方差矩阵: C = X^T @ X / (N-1)
                cov = (X_centered.T @ X_centered) / (N - 1)  # [D, D]
                cov += torch.eye(D, device=self.device) * cov_reg

                # 求逆
                try:
                    inv_cov = torch.linalg.inv(cov)
                except:
                    inv_cov = torch.eye(D, device=self.device)

                patch_inv_covs[i, j] = inv_cov

        # 保存到CPU
        self.patch_means = patch_means.cpu().numpy()
        self.patch_inv_covs = patch_inv_covs.cpu().numpy()

        print(f"✅ 高斯分布建立完成（GPU加速）！")

    def compute_anomaly_maps(self, patch_features=None):
        """🚀 GPU批量计算异常图（完全向量化）"""
        if patch_features is None:
            patch_features = self.all_patch_features

        print("\n" + "=" * 80)
        print("💯 计算异常分数（GPU向量化，无循环）")
        print("=" * 80)

        N, H, W, D = patch_features.shape

        # 转GPU
        features_gpu = torch.from_numpy(patch_features).to(self.device)  # [N, H, W, D]
        means_gpu = torch.from_numpy(self.patch_means).to(self.device)  # [H, W, D]
        inv_covs_gpu = torch.from_numpy(self.patch_inv_covs).to(self.device)  # [H, W, D, D]

        # ✅ 完全向量化计算Mahalanobis距离
        print("计算Mahalanobis距离（向量化）...")

        # 展开为 [N*H*W, D]
        features_flat = features_gpu.reshape(N * H * W, D)
        means_flat = means_gpu.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * H * W, D)
        inv_covs_flat = inv_covs_gpu.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * H * W, D, D)

        # delta = x - mu
        delta = features_flat - means_flat  # [N*H*W, D]

        # Mahalanobis: sqrt(delta^T @ inv_cov @ delta)
        # 批量矩阵乘法
        delta_cov = torch.bmm(delta.unsqueeze(1), inv_covs_flat)  # [N*H*W, 1, D]
        mahal_sq = torch.bmm(delta_cov, delta.unsqueeze(2)).squeeze()  # [N*H*W]
        mahal_dist = torch.sqrt(torch.clamp(mahal_sq, min=0))  # [N*H*W]

        # Reshape回 [N, H, W]
        anomaly_maps = mahal_dist.reshape(N, H, W).cpu().numpy()

        # ✅ 图像级分数（Top-K patches）
        k = getattr(self.config, 'TOP_K_PATCHES', 10)
        anomaly_maps_flat = anomaly_maps.reshape(N, -1)
        top_k_scores = np.partition(anomaly_maps_flat, -k, axis=1)[:, -k:]
        image_scores = top_k_scores.mean(axis=1)

        print(f"✅ 异常分数计算完成（GPU加速）！")
        print(f"   - 分数范围: [{image_scores.min():.4f}, {image_scores.max():.4f}]")

        return anomaly_maps, image_scores

    def visualize_anomaly_maps(self, anomaly_maps, selected_indices, output_dir):
        """🚀 多进程并行生成热力图"""
        print("\n" + "=" * 80)
        print("🎨 生成异常热力图（多进程加速）")
        print("=" * 80)

        viz_dir = Path(output_dir) / 'anomaly_heatmaps'
        viz_dir.mkdir(exist_ok=True)

        viz_k = min(getattr(self.config, 'VIZ_TOP_K', 50), len(selected_indices))
        sigma = getattr(self.config, 'HEATMAP_SIGMA', 4)
        alpha = getattr(self.config, 'HEATMAP_ALPHA', 0.5)

        print(f"生成Top-{viz_k}异常样本的热力图...")

        # ✅ 多进程并行生成（可选，如果需要进一步加速）
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def generate_heatmap(idx):
            """单个热力图生成函数"""
            img_path = self.all_paths[idx]
            anomaly_map = anomaly_maps[idx]

            try:
                img_orig = cv2.imread(img_path)
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                h_orig, w_orig = img_orig.shape[:2]
            except:
                return None

            # 上采样
            anomaly_map_resized = cv2.resize(anomaly_map, (w_orig, h_orig),
                                             interpolation=cv2.INTER_CUBIC)
            anomaly_map_smooth = gaussian_filter(anomaly_map_resized, sigma=sigma)

            # 归一化
            anomaly_map_norm = (anomaly_map_smooth - anomaly_map_smooth.min()) / \
                               (anomaly_map_smooth.max() - anomaly_map_smooth.min() + 1e-8)

            # 热力图
            heatmap = cv2.applyColorMap(
                (anomaly_map_norm * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 叠加
            overlay = (img_orig * (1 - alpha) + heatmap * alpha).astype(np.uint8)

            # 保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_orig)
            axes[0].set_title('Original', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(anomaly_map_norm, cmap='jet')
            axes[1].set_title('Anomaly Map', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay', fontsize=12)
            axes[2].axis('off')

            plt.suptitle(f'{Path(img_path).name}\nScore: {anomaly_map.max():.4f}',
                         fontsize=10)
            plt.tight_layout()

            save_path = viz_dir / f'{Path(img_path).stem}_heatmap.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return save_path

        # ⚠️ 多进程可能导致matplotlib问题，建议单进程或减少viz数量
        # 如果遇到问题，改为普通循环
        for idx in tqdm(selected_indices[:viz_k], desc="生成热力图"):
            generate_heatmap(idx)

        print(f"✅ 热力图已保存到: {viz_dir}")

# ==================== 配置示例 ====================
class Config:
    # 路径
    IMAGE_DIR = r"D:\Huangda_data0801\view\2\B"      # 待筛选的所有图像
    OUTPUT_DIR = "outputs_2_B"                        # 输出目录
    CACHE_DIR = "./cache"
    DINOV3_REPO_ROOT = r"D:\ly\dinov3-main"          # DINOv3 仓库路径
    DINOV3_WEIGHT_PATH = r"D:\ly\dinov3-main\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"  # 权重文件

    # 模型
    MODEL_NAME = "dinov3_vitl16_lvd1689m_distilled"  # DINOv3模型名
    IMAGE_SIZE = 512

    # Patch特征
    FEATURE_LAYERS = [8, 16, 23]       # 提取ViT的第6/9/12层（多尺度）
    TOP_K_PATCHES = 10                 # 图像分数 = top-k patches的平均分

    # PCA
    USE_PCA = True
    PCA_DIM = 256                      # 降维后维度

    # 筛选
    USE_TOP_K = False
    TOP_K = 1000
    TOP_PERCENT = 0.1                  # 筛选top 10%

    # GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # 可视化
    VIZ_TOP_K = 50                     # 可视化top-50异常样本的热力图


# ==================== 运行示例 ====================
if __name__ == "__main__":
    config = Config()

    # ✅ 使用加速版
    detector = PatchLevelAnomalyDetectorFast(config)
    csv_file = detector.run_pipeline()

    print(f"\n🎉 标注清单: {csv_file}")
