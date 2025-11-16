"""
🔥 工业级异常检测系统 v2.1 - 优化版
优化点：
1. ✅ AE训练加速（混合精度 + 更大batch + 数据预加载）
2. ✅ DINOv3小batch推理（梯度累积模拟大batch效果）
3. ✅ 内存优化（及时清理缓存）
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import faiss
from sklearn.neighbors import LocalOutlierFactor
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
from datetime import timedelta
import gc

warnings.filterwarnings('ignore')


# ==================== 配置参数 ====================
class Config:
    """DINOv3 专用配置"""
    # ========== 路径配置 ==========
    BASE_DIR = r"C:\Users\Administrator\Desktop\f-AnoGAN\RD4AD-main\mvtec\zhawa_guzhang_zhifangtujunhenghua"
    INITIAL_NORMAL_DIR = os.path.join(BASE_DIR, "train", "good")
    MIXED_DIR = r'D:\Huangda_data0801\view\2\B'
    NORMAL_DIR = INITIAL_NORMAL_DIR
    TEST_DIR = MIXED_DIR
    OUTPUT_DIR = "./outputs"

    # ========== DINOv3 模型配置 ==========
    BACKBONE = "dinov3_vitl16"
    DINOV3_REPO_ROOT = r"D:\ly\dinov3-main"
    DINOV3_WEIGHT_PATH = r"D:\ly\dinov3-main\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    DINOV3_MODEL_NAME = "dinov3_vitl16_lvd1689m_distilled"
    DINOV3_FEATURE_LAYERS = [3, 6, 9, 11]  # ViT-L/16 有12层
    DINOV3_PATCH_SIZE = 16

    # ========== 通用配置 ==========
    IMAGE_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0

    # ========== 🔥 优化1：AE训练加速配置 ==========
    AE_BATCH_SIZE = 128           # ✅ 大幅提升（原512太大，改128）
    AE_EPOCHS = 50                 # ✅ 保持轻量
    AE_LR = 2e-3                  # ✅ 提高学习率
    AE_USE_AMP = True             # ✅ 混合精度训练
    AE_NUM_WORKERS = 4            # ✅ 多线程加载
    AE_PREFETCH_FACTOR = 2        # ✅ 预加载

    # ========== 🔥 优化2：DINOv3小batch配置 ==========
    DINO_BATCH_SIZE = 8           # ✅ 小batch（显存友好）
    DINO_ACCUMULATION_STEPS = 4   # ✅ 梯度累积模拟32的效果
    DINO_NUM_WORKERS = 2          # ✅ 适度并行

    # ========== 步骤1：粗筛配置 ==========
    EXPAND_RATIO = 0.7

    # ========== 步骤2：清洗配置 ==========
    LOF_NEIGHBORS = 20
    CONTAMINATION = 0.05

    # ========== 步骤3：FAISS配置 ==========
    FAISS_NLIST = 100
    FAISS_NPROBE = 10
    USE_CORESET = True
    CORESET_RATIO = 0.05

    # ========== 步骤6：异常评分配置 ==========
    N_NEIGHBORS = 9
    ANOMALY_PERCENTILE = 90


# ==================== 数据集 ====================
class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None, image_size=224):
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))

        if len(self.image_paths) == 0:
            raise ValueError(f"未在 {image_dir} 中找到图片！")

        self.transform = transform
        self.image_size = image_size
        print(f"   加载 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.image_paths[idx]
        except Exception as e:
            print(f"⚠️  加载图片失败: {self.image_paths[idx]} - {e}")
            return torch.zeros(3, self.image_size, self.image_size), self.image_paths[idx]


class PathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, img_path  # ✅ 返回元组
        except Exception as e:
            print(f"⚠️  加载图片失败: {img_path} - {e}")
            return torch.zeros(3, 224, 224), img_path


# ==================== 🔥 优化后的AE训练 ====================
class SimpleAutoEncoder(nn.Module):
    """轻量级自编码器（保持不变）"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(dataloader, config):
    """🔥 优化：混合精度 + 更快训练"""
    print("\n" + "=" * 60)
    print("📌 [步骤1] 训练自编码器进行粗筛 (加速版)")
    print("=" * 60)
    print(f"   ⚡ 混合精度: {'启用' if config.AE_USE_AMP else '禁用'}")
    print(f"   ⚡ Batch Size: {config.AE_BATCH_SIZE}")
    print(f"   ⚡ Learning Rate: {config.AE_LR}")

    model = SimpleAutoEncoder().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
    criterion = nn.MSELoss()

    # ✅ 混合精度scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.AE_USE_AMP)

    model.train()
    for epoch in range(config.AE_EPOCHS):
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.AE_EPOCHS}")

        for imgs, _ in pbar:
            imgs = imgs.to(config.DEVICE)

            optimizer.zero_grad()

            # ✅ 自动混合精度
            with torch.cuda.amp.autocast(enabled=config.AE_USE_AMP):
                recon = model(imgs)
                loss = criterion(recon, imgs)

            # ✅ 使用scaler进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(losses):.4f}"})

        print(f"   Epoch {epoch + 1}/{config.AE_EPOCHS} - Loss: {np.mean(losses):.4f}")

    # ✅ 清理缓存
    torch.cuda.empty_cache()
    gc.collect()

    return model


def expand_normal_gallery(ae_model, mixed_dataloader, config):
    """扩充Normal Gallery（保持不变）"""
    print("\n   使用AE筛选正常图...")
    ae_model.eval()
    reconstruction_errors = []
    image_paths = []

    with torch.no_grad():
        for imgs, paths in tqdm(mixed_dataloader, desc="   计算重建误差"):
            imgs = imgs.to(config.DEVICE)

            # ✅ 混合精度推理
            with torch.cuda.amp.autocast(enabled=config.AE_USE_AMP):
                recon = ae_model(imgs)
                errors = F.mse_loss(recon, imgs, reduction='none')

            errors = errors.view(imgs.size(0), -1).mean(dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
            image_paths.extend(paths)

    errors = np.array(reconstruction_errors)
    threshold = np.percentile(errors, config.EXPAND_RATIO * 100)
    selected_indices = np.where(errors <= threshold)[0]

    expanded_paths = [image_paths[i] for i in selected_indices]
    print(f"   ✅ 从 {len(image_paths)} 张混合图中筛选出 {len(expanded_paths)} 张正常图")
    print(f"   ✅ 重建误差阈值: {threshold:.6f}")

    # ✅ 清理
    torch.cuda.empty_cache()
    gc.collect()

    return expanded_paths, errors


# ==================== 步骤2：特征提取器 ====================
class FeatureExtractor(nn.Module):
    """DINOv3 专用特征提取器（保持不变）"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.backbone_type = self._parse_backbone_type()

        if self.backbone_type == 'dinov3':
            self.model, self.feature_layers = self._load_dinov3()
        else:
            raise ValueError(f"当前仅支持DINOv3，收到: {config.BACKBONE}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _parse_backbone_type(self) -> str:
        backbone = self.config.BACKBONE.lower()
        if 'dinov3' in backbone or 'dino_v3' in backbone:
            return 'dinov3'
        else:
            raise ValueError(f"请使用DINOv3模型，当前配置: {self.config.BACKBONE}")

    def _load_dinov3(self):
        """从本地加载DINOv3模型"""
        print("\n" + "=" * 60)
        print("📥 从本地加载 DINOv3 模型")
        print("=" * 60)

        repo_root = self.config.DINOV3_REPO_ROOT
        weight_path = self.config.DINOV3_WEIGHT_PATH

        if not os.path.exists(repo_root):
            raise FileNotFoundError(f"DINOv3仓库路径不存在: {repo_root}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        print(f"   📂 仓库路径: {repo_root}")
        print(f"   📂 权重路径: {weight_path}")

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            from dinov3.models.vision_transformer import vit_large

            model = vit_large(
                patch_size=self.config.DINOV3_PATCH_SIZE,
                num_register_tokens=0,
                interpolate_antialias=False,
                interpolate_offset=0.1,
            )

            print(f"   📥 加载权重...")
            state_dict = torch.load(weight_path, map_location='cpu')

            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '').replace('backbone.', '')
                new_state_dict[new_key] = v

            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"   ✅ 权重加载完成")

        except Exception as e:
            raise RuntimeError(f"DINOv3加载失败: {e}")

        model = model.to(self.config.DEVICE)

        feature_layers = self.config.DINOV3_FEATURE_LAYERS
        self.features = {}

        print(f"\n   🔧 注册特征提取hook...")
        print(f"   📊 模型总层数: {len(model.blocks)}")
        print(f"   📌 提取层索引: {feature_layers}")

        for layer_idx in feature_layers:
            if layer_idx >= len(model.blocks):
                raise ValueError(f"层索引{layer_idx}超出范围(总共{len(model.blocks)}层)")

            def make_hook(idx):
                def hook(module, input, output):
                    self.features[f'block_{idx}'] = output
                return hook

            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

        print(f"   ✅ Hook注册完成")
        print("=" * 60)

        return model, [f'block_{i}' for i in feature_layers]

    def forward(self, x):
        """前向传播 - DINOv3专用（修复版）"""
        # 执行前向传播，触发hooks
        with torch.no_grad():
            _ = self.model(x)

        # 提取并处理特征
        output_features = {}
        B = x.shape[0]
        patch_size = self.config.DINOV3_PATCH_SIZE
        H = W = self.config.IMAGE_SIZE // patch_size  # 224/16 = 14

        for layer_name, feat in self.features.items():
            try:
                # ✅ 处理可能的元组/列表输出
                if isinstance(feat, (tuple, list)):
                    # DINOv3 可能返回 (class_token, patch_tokens, register_tokens)
                    # 或者 [output1, output2, ...]
                    if len(feat) >= 2:
                        # 通常第2个元素是patch tokens
                        feat = feat[1]
                    else:
                        feat = feat[0]
                # ✅ 确保是Tensor
                if not isinstance(feat, torch.Tensor):
                    print(f"   ⚠️  跳过 {layer_name}：不是Tensor (类型: {type(feat)})")
                    continue

                # ✅ 处理不同的特征形状
                if feat.dim() == 3:  # [B, N+1, D] 或 [B, N, D]
                    expected_patches = H * W

                    if feat.shape[1] == expected_patches + 1:
                        # 包含CLS token，去掉第一个token
                        feat = feat[:, 1:, :]  # [B, N, D]
                    elif feat.shape[1] == expected_patches:
                        # 已经是纯patch tokens
                        pass
                    else:
                        print(f"   ⚠️  {layer_name} patch数量异常: {feat.shape[1]} (期望 {expected_patches})")
                        # 尝试自适应调整
                        actual_H = actual_W = int(np.sqrt(feat.shape[1]))
                        if actual_H * actual_W == feat.shape[1]:
                            H = W = actual_H
                            print(f"   🔧 自动调整为 {H}x{W}")
                        else:
                            continue

                    D = feat.shape[-1]
                    # reshape为 [B, D, H, W]
                    feat = feat.transpose(1, 2).reshape(B, D, H, W)

                elif feat.dim() == 4:  # [B, D, H, W] (已经是特征图格式)
                    pass

                else:
                    print(f"   ⚠️  跳过 {layer_name}：维度异常 ({feat.dim()}维)")
                    continue

                output_features[layer_name] = feat

            except Exception as e:
                print(f"   ❌ 处理 {layer_name} 时出错: {e}")
                continue

        if len(output_features) == 0:
            raise RuntimeError("没有成功提取任何特征！请检查hook配置和模型输出格式")

        return output_features


# ==================== 🔥 优化后的DataLoader ====================
def get_ae_dataloader(dataset, config, shuffle=False):
    """AE专用DataLoader（大batch + 预加载）"""
    return DataLoader(
        dataset,
        batch_size=config.AE_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.AE_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.AE_NUM_WORKERS > 0 else False,
        prefetch_factor=config.AE_PREFETCH_FACTOR if config.AE_NUM_WORKERS > 0 else None,
        drop_last=False
    )


def get_dino_dataloader(dataset, config, shuffle=False):
    """DINOv3专用DataLoader（小batch）"""
    return DataLoader(
        dataset,
        batch_size=config.DINO_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.DINO_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,  # Windows兼容
        drop_last=False
    )


def extract_features(dataloader, feature_extractor, config):
    """提取多层patch特征（小batch优化）"""
    print("\n" + "=" * 60)
    print("📌 [步骤2] 提取多层特征 (小Batch优化)")
    print("=" * 60)
    print(f"   📊 Batch Size: {config.DINO_BATCH_SIZE}")

    all_features = []
    all_paths = []

    with torch.no_grad():
        for imgs, paths in tqdm(dataloader, desc="   提取特征"):
            imgs = imgs.to(config.DEVICE)
            features = feature_extractor(imgs)

            # 融合多层特征
            embeddings = []
            for layer in feature_extractor.feature_layers:
                feat = features[layer]
                feat = F.adaptive_avg_pool2d(feat, (16, 16))
                embeddings.append(feat)

            embedding = torch.cat(embeddings, dim=1)
            B, C, H, W = embedding.shape

            embedding = embedding.permute(0, 2, 3, 1).reshape(B, H * W, C)

            all_features.append(embedding.cpu().numpy())
            all_paths.extend(paths)

            # ✅ 及时清理
            del imgs, features, embeddings, embedding
            if len(all_features) % 50 == 0:  # 每50个batch清理一次
                torch.cuda.empty_cache()

    all_features = np.concatenate(all_features, axis=0)
    print(f"   ✅ 提取特征形状: {all_features.shape}")
    print(f"   ✅ 特征维度: {all_features.shape[-1]} (多层融合)")

    # ✅ 最终清理
    torch.cuda.empty_cache()
    gc.collect()

    return all_features, all_paths


def clean_gallery_with_lof(features, paths, config):
    """使用LOF清洗（保持不变）"""
    print("\n" + "=" * 60)
    print("📌 [步骤2] 使用LOF清洗Gallery")
    print("=" * 60)

    global_features = features.mean(axis=1)

    lof = LocalOutlierFactor(
        n_neighbors=config.LOF_NEIGHBORS,
        contamination=config.CONTAMINATION,
        n_jobs=-1
    )

    print(f"   🔍 LOF参数: n_neighbors={config.LOF_NEIGHBORS}, contamination={config.CONTAMINATION}")
    labels = lof.fit_predict(global_features)

    clean_indices = np.where(labels == 1)[0]
    outlier_indices = np.where(labels == -1)[0]

    clean_features = features[clean_indices]
    clean_paths = [paths[i] for i in clean_indices]

    print(f"   ✅ 清洗前: {len(paths)} 张")
    print(f"   ✅ 清洗后: {len(clean_paths)} 张 (保留 {len(clean_paths) / len(paths) * 100:.1f}%)")
    print(f"   🗑️  剔除离群点: {len(outlier_indices)} 张")

    return clean_features, clean_paths


# ==================== 步骤3：Coreset + FAISS ====================
def greedy_coreset_sampling(features: np.ndarray, ratio: float = 0.1):
    """🔥 优化版：快速Coreset采样（使用FAISS加速 + 修复进度条）"""
    print("\n" + "=" * 60)
    print(f"📌 [步骤3] Coreset采样 (保留 {ratio * 100:.1f}%) - FAISS加速版")
    print("=" * 60)

    N, D = features.shape
    target_n = max(int(N * ratio), 1000)

    if target_n >= N:
        print(f"   ⚠️  目标数量({target_n}) >= 总数({N})，跳过采样")
        return np.arange(N)

    print(f"   📊 总样本数: {N:,}")
    print(f"   🎯 目标数量: {target_n:,}")

    # 使用FAISS加速
    features_normalized = features.astype('float32')
    faiss.normalize_L2(features_normalized)

    index = faiss.IndexFlatL2(D)
    index.add(features_normalized)

    # 随机初始化种子点
    num_seeds = min(10, target_n // 100)
    selected_indices = np.random.choice(N, num_seeds, replace=False).tolist()

    print(f"   🌱 初始化种子点: {num_seeds} 个")

    # 初始化最小距离
    min_distances = np.full(N, np.inf, dtype='float32')

    # 计算到所有种子点的距离
    for seed_idx in selected_indices:
        distances, _ = index.search(features_normalized[seed_idx:seed_idx + 1], N)
        min_distances = np.minimum(min_distances, distances[0])

    # ✅ 修复：使用更清晰的进度条逻辑
    batch_size = min(100, max(1, target_n // 100))  # 每次采样100个
    remaining = target_n - len(selected_indices)

    print(f"   🔄 剩余采样数: {remaining:,} (每批 {batch_size} 个)")

    # 使用时间估计的进度条
    from time import time
    start_time = time()

    pbar = tqdm(
        total=remaining,
        desc="   采样进度",
        unit="samples",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    iteration = 0
    while len(selected_indices) < target_n:
        iteration += 1
        current_batch = min(batch_size, target_n - len(selected_indices))

        # 选择距离最大的k个点
        if current_batch == 1:
            next_indices = [np.argmax(min_distances)]
        else:
            next_indices = np.argpartition(min_distances, -current_batch)[-current_batch:]

        # 更新
        added_count = 0
        for next_idx in next_indices:
            if next_idx not in selected_indices and len(selected_indices) < target_n:
                selected_indices.append(int(next_idx))
                added_count += 1

                # 更新距离（每10个点批量更新一次，进一步加速）
                if added_count % 10 == 0 or len(selected_indices) >= target_n:
                    distances, _ = index.search(
                        features_normalized[next_idx:next_idx + 1], N
                    )
                    min_distances = np.minimum(min_distances, distances[0])

        # ✅ 修复：只更新实际添加的数量
        pbar.update(added_count)

        # 防止死循环
        if added_count == 0:
            print(f"\n   ⚠️  无法继续采样，当前数量: {len(selected_indices)}")
            break

    pbar.close()

    elapsed = time() - start_time
    print(f"   ⏱️  采样耗时: {elapsed:.1f} 秒")
    print(f"   ✅ 采样完成: {len(selected_indices):,} / {N:,} patches")
    print(f"   ✅ 采样效率: {len(selected_indices) / elapsed:.0f} samples/s")

    return np.array(selected_indices[:target_n])


def build_faiss_index(features, config):
    """构建FAISS索引（保持不变）"""
    print("\n   🔨 构建FAISS索引...")

    N_images, N_patches, D = features.shape
    patch_features = features.reshape(-1, D).astype('float32')

    print(f"   📊 原始patch数: {patch_features.shape[0]:,}")

    if config.USE_CORESET and patch_features.shape[0] > 10000:
        coreset_indices = greedy_coreset_sampling(patch_features, config.CORESET_RATIO)
        patch_features = patch_features[coreset_indices]

    faiss.normalize_L2(patch_features)

    if patch_features.shape[0] < 50000:
        index = faiss.IndexFlatL2(D)
        index.add(patch_features)
        print(f"   ✅ 使用精确索引 (IndexFlatL2)")
    else:
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFPQ(quantizer, D, config.FAISS_NLIST, 64, 8)
        index.train(patch_features)
        index.add(patch_features)
        index.nprobe = config.FAISS_NPROBE
        print(f"   ✅ 使用IVF-PQ索引 (nlist={config.FAISS_NLIST})")

    print(f"   ✅ 索引包含 {index.ntotal:,} 个patch特征")

    return index, patch_features


# ==================== 步骤6：异常评分 ====================
def compute_anomaly_scores(test_features, faiss_index, config):
    """PatchCore软对齐（保持不变）"""
    print("\n" + "=" * 60)
    print("📌 [步骤4-6] 计算异常分数 (PatchCore方式)")
    print("=" * 60)

    N_test, N_patches, D = test_features.shape
    anomaly_scores = []
    anomaly_maps = []

    for i in tqdm(range(N_test), desc="   计算分数"):
        query = test_features[i].astype('float32')
        faiss.normalize_L2(query)

        distances, _ = faiss_index.search(query, config.N_NEIGHBORS)

        patch_scores = distances.mean(axis=1)

        image_score = patch_scores.max()
        anomaly_scores.append(image_score)

        h = w = int(np.sqrt(N_patches))
        anomaly_map = patch_scores.reshape(h, w)
        anomaly_maps.append(anomaly_map)

    scores = np.array(anomaly_scores)
    print(f"   ✅ 异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   ✅ 异常分数均值: {scores.mean():.4f}")

    return scores, anomaly_maps


def generate_anomaly_heatmap(anomaly_map, original_img_path, save_path):
    """生成异常热力图（保持不变）"""
    img = cv2.imread(original_img_path)
    if img is None:
        print(f"⚠️  无法读取图片: {original_img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    heatmap = cv2.resize(anomaly_map, (W, H))
    heatmap = gaussian_filter(heatmap, sigma=4)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(save_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))


# ==================== 主流程 ====================
class AnomalyDetectionPipeline:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def run(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("🚀 开始训练流程 (优化版)")
        print("=" * 60)

        # ========== 步骤1：AE粗筛（大batch加速）==========
        print(f"\n📂 加载初始正常图: {self.config.INITIAL_NORMAL_DIR}")
        initial_dataset = ImageDataset(
            self.config.INITIAL_NORMAL_DIR,
            self.transform,
            self.config.IMAGE_SIZE
        )
        initial_loader = get_ae_dataloader(initial_dataset, self.config, shuffle=True)

        ae_model = train_autoencoder(initial_loader, self.config)

        print(f"\n📂 加载混合图: {self.config.MIXED_DIR}")
        mixed_dataset = ImageDataset(
            self.config.MIXED_DIR,
            self.transform,
            self.config.IMAGE_SIZE
        )
        mixed_loader = get_ae_dataloader(mixed_dataset, self.config, shuffle=False)

        expanded_paths, _ = expand_normal_gallery(ae_model, mixed_loader, self.config)

        # ✅ 删除AE模型释放显存
        del ae_model
        torch.cuda.empty_cache()
        gc.collect()

        # ========== 步骤2：DINOv3特征提取（小batch）==========
        all_normal_paths = initial_dataset.image_paths + expanded_paths

        gallery_dataset = PathDataset(all_normal_paths, self.transform)
        gallery_loader = get_dino_dataloader(gallery_dataset, self.config, shuffle=False)

        feature_extractor = FeatureExtractor(self.config).to(self.config.DEVICE)

        gallery_features, gallery_paths = extract_features(
            gallery_loader, feature_extractor, self.config
        )

        clean_features, clean_paths = clean_gallery_with_lof(
            gallery_features, gallery_paths, self.config
        )

        # ========== 步骤3：FAISS索引 ==========
        faiss_index, memory_bank = build_faiss_index(clean_features, self.config)

        # ========== 保存模型 ==========
        model_save_path = os.path.join(self.config.OUTPUT_DIR, 'model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'faiss_index': faiss.serialize_index(faiss_index),
                'clean_paths': clean_paths,
                'config': self.config
            }, f)

        print("\n" + "=" * 60)
        print(f"✅ 训练完成！模型已保存至: {model_save_path}")
        print("=" * 60)

        return feature_extractor, faiss_index, clean_paths

    def inference(self, test_dir: str, feature_extractor, faiss_index):
        """推理（小batch）"""
        print("\n" + "=" * 60)
        print("🔍 开始异常检测 (小Batch优化)")
        print("=" * 60)

        test_dataset = ImageDataset(test_dir, self.transform, self.config.IMAGE_SIZE)
        test_loader = get_dino_dataloader(test_dataset, self.config, shuffle=False)

        test_features, test_paths = extract_features(
            test_loader, feature_extractor, self.config
        )

        scores, anomaly_maps = compute_anomaly_scores(
            test_features, faiss_index, self.config
        )

        threshold = np.percentile(scores, self.config.ANOMALY_PERCENTILE)
        print(f"\n   🎯 动态阈值 ({self.config.ANOMALY_PERCENTILE}%分位): {threshold:.4f}")

        sorted_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices):
            results.append({
                'rank': rank + 1,
                'path': test_paths[idx],
                'score': scores[idx],
                'is_anomaly': scores[idx] > threshold
            })

        # 保存结果
        result_path = os.path.join(self.config.OUTPUT_DIR, 'anomaly_results.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"{'Rank':<6} {'Score':<12} {'Status':<12} {'Path'}\n")
            f.write("=" * 80 + "\n")

            # ✅ 修改：保存所有结果而不是只保存前100
            for r in results:  # 改为全部
                status = '🔴 ANOMALY' if r['is_anomaly'] else '🟢 NORMAL'
                f.write(f"{r['rank']:<6} {r['score']:<12.6f} {status:<12} {r['path']}\n")

        print(f"\n   ✅ 结果已保存: {result_path}")
        print(f"   📊 保存数量: {len(results)} 张")  # ✅ 添加提示

        # 热力图
        heatmap_dir = os.path.join(self.config.OUTPUT_DIR, 'heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)

        print("\n   🎨 生成热力图...")
        for i, idx in enumerate(tqdm(sorted_indices[:20], desc="   生成中")):
            save_path = os.path.join(
                heatmap_dir,
                f"rank{i + 1}_score{scores[idx]:.4f}.jpg"
            )
            generate_anomaly_heatmap(
                anomaly_maps[idx],
                test_paths[idx],
                save_path
            )

        print(f"   ✅ 热力图已保存: {heatmap_dir}")

        # 统计
        n_anomalies = sum(r['is_anomaly'] for r in results)
        print("\n" + "=" * 60)
        print(f"📊 检测统计")
        print("=" * 60)
        print(f"   总计: {len(results)} 张")
        print(f"   异常: {n_anomalies} 张 ({n_anomalies / len(results) * 100:.1f}%)")
        print(f"   正常: {len(results) - n_anomalies} 张")

        return results


# ==================== 在线学习 ====================
def incremental_update(new_normal_dir: str, model_path: str):
    """增量更新Normal Gallery（保持不变）"""
    print("\n" + "=" * 60)
    print("🔄 增量更新Normal Gallery")
    print("=" * 60)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    faiss_index = faiss.deserialize_index(data['faiss_index'])
    clean_paths = data['clean_paths']
    config = data['config']

    print(f"   📂 当前Gallery: {len(clean_paths)} 张")

    feature_extractor = FeatureExtractor(config).to(config.DEVICE)

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    new_dataset = ImageDataset(new_normal_dir, transform, config.IMAGE_SIZE)
    new_loader = get_dino_dataloader(new_dataset, config, shuffle=False)

    new_features, new_paths = extract_features(new_loader, feature_extractor, config)

    N, P, D = new_features.shape
    new_patch_features = new_features.reshape(-1, D).astype('float32')
    faiss.normalize_L2(new_patch_features)
    faiss_index.add(new_patch_features)

    clean_paths.extend(new_paths)

    with open(model_path, 'wb') as f:
        pickle.dump({
            'faiss_index': faiss.serialize_index(faiss_index),
            'clean_paths': clean_paths,
            'config': config
        }, f)

    print(f"   ✅ 新增 {len(new_paths)} 张正常图")
    print(f"   ✅ 更新后Gallery: {len(clean_paths)} 张")
    print(f"   ✅ 索引大小: {faiss_index.ntotal:,} patches")


# ==================== 主函数 ====================
if __name__ == '__main__':
    config = Config()

    print("\n" + "=" * 60)
    print("🔍 验证配置")
    print("=" * 60)
    print(f"✅ 初始正常图路径: {config.INITIAL_NORMAL_DIR}")
    print(f"✅ 混合数据路径: {config.MIXED_DIR}")
    print(f"✅ 测试数据路径: {config.TEST_DIR}")
    print(f"✅ DINOv3仓库: {config.DINOV3_REPO_ROOT}")
    print(f"✅ DINOv3权重: {config.DINOV3_WEIGHT_PATH}")
    print(f"✅ Backbone: {config.BACKBONE}")
    print(f"✅ 设备: {config.DEVICE}")
    print(f"\n🔥 优化配置:")
    print(f"   AE Batch: {config.AE_BATCH_SIZE} (混合精度: {config.AE_USE_AMP})")
    print(f"   DINOv3 Batch: {config.DINO_BATCH_SIZE}")

    if not os.path.exists(config.INITIAL_NORMAL_DIR):
        raise FileNotFoundError(f"训练数据路径不存在: {config.INITIAL_NORMAL_DIR}")
    if not os.path.exists(config.DINOV3_REPO_ROOT):
        raise FileNotFoundError(f"DINOv3仓库不存在: {config.DINOV3_REPO_ROOT}")
    if not os.path.exists(config.DINOV3_WEIGHT_PATH):
        raise FileNotFoundError(f"DINOv3权重不存在: {config.DINOV3_WEIGHT_PATH}")

    pipeline = AnomalyDetectionPipeline(config)

    # 训练
    feature_extractor, faiss_index, clean_paths = pipeline.run()

    # 推理
    if os.path.exists(config.TEST_DIR):
        results = pipeline.inference(config.TEST_DIR, feature_extractor, faiss_index)

        print("\n" + "=" * 60)
        print("🔝 Top 10 最可疑异常")
        print("=" * 60)
        for r in results[:10]:
            status = '🔴' if r['is_anomaly'] else '🟢'
            print(f"{status} Rank {r['rank']}: {r['score']:.6f} - {os.path.basename(r['path'])}")
    else:
        print(f"⚠️  测试路径不存在，跳过推理")
