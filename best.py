"""
🔥 工业级异常检测系统 v2.0 - 支持 ResNet + DINOv3
集成所有优化：PatchCore软对齐 + Coreset + 多尺度 + 在线学习
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

warnings.filterwarnings('ignore')


# ==================== 配置参数 ====================
class Config:
    """DINOv3 专用配置"""
    # ========== 路径配置 ==========
    BASE_DIR = r"C:\Users\Administrator\Desktop\f-AnoGAN\RD4AD-main\mvtec\zhawa_guzhang_zhifangtujunhenghua"
    INITIAL_NORMAL_DIR = os.path.join(BASE_DIR, "train", "good")
    MIXED_DIR = r'D:\Huangda_data0801\mix'  #
    NORMAL_DIR = INITIAL_NORMAL_DIR
    TEST_DIR = MIXED_DIR
    OUTPUT_DIR = "./outputs"

    # ========== DINOv3 模型配置 ========== ✅ 关键修复
    BACKBONE = "dinov3_vitl16"  # ✅ 明确使用DINOv3
    DINOV3_REPO_ROOT = r"D:\ly\dinov3-main"
    DINOV3_WEIGHT_PATH = r"D:\ly\dinov3-main\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    DINOV3_MODEL_NAME = "dinov3_vitl16_lvd1689m_distilled"  # 可选的完整名称
    DINOV3_FEATURE_LAYERS = [3, 6, 9, 11]  # ViT-L/16 有12层，选择中间层
    DINOV3_PATCH_SIZE = 16  # ✅ 注意：vitl16的patch size是16

    # ========== 通用配置 ==========
    IMAGE_SIZE = 224
    BATCH_SIZE = 512  # DINOv3显存占用大
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 12

    # ========== 步骤1：粗筛配置 ==========
    AE_EPOCHS = 50
    AE_LR = 1e-3
    EXPAND_RATIO = 0.7

    # ========== 步骤2：清洗配置 ==========
    LOF_NEIGHBORS = 20
    CONTAMINATION = 0.05

    # ========== 步骤3：FAISS配置 ==========
    FAISS_NLIST = 100
    FAISS_NPROBE = 10
    USE_CORESET = True
    CORESET_RATIO = 0.1

    # ========== 步骤6：异常评分配置 ==========
    N_NEIGHBORS = 9
    ANOMALY_PERCENTILE = 90

# ==================== 数据集 ====================
class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None, image_size=224):  # ✅ 添加参数
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))

        if len(self.image_paths) == 0:
            raise ValueError(f"未在 {image_dir} 中找到图片！")

        self.transform = transform
        self.image_size = image_size  # ✅ 保存
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
            # ✅ 使用动态尺寸
            return torch.zeros(3, self.image_size, self.image_size), self.image_paths[idx]

class PathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
# ==================== 步骤1：自监督粗筛 ====================
class SimpleAutoEncoder(nn.Module):
    """轻量级自编码器"""

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
    """训练自编码器"""
    print("\n" + "=" * 60)
    print("📌 [步骤1] 训练自编码器进行粗筛")
    print("=" * 60)

    model = SimpleAutoEncoder().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(config.AE_EPOCHS):
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.AE_EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(config.DEVICE)
            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(losses):.4f}"})

        print(f"   Epoch {epoch + 1}/{config.AE_EPOCHS} - Loss: {np.mean(losses):.4f}")

    return model


def expand_normal_gallery(ae_model, mixed_dataloader, config):
    """扩充Normal Gallery"""
    print("\n   使用AE筛选正常图...")
    ae_model.eval()
    reconstruction_errors = []
    image_paths = []

    with torch.no_grad():
        for imgs, paths in tqdm(mixed_dataloader, desc="   计算重建误差"):
            imgs = imgs.to(config.DEVICE)
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

    return expanded_paths, errors


# ==================== 步骤2：多层特征提取器 ====================
class FeatureExtractor(nn.Module):
    """🔥 DINOv3 专用特征提取器"""

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
        """判断backbone类型"""
        backbone = self.config.BACKBONE.lower()
        if 'dinov3' in backbone or 'dino_v3' in backbone:
            return 'dinov3'
        else:
            raise ValueError(f"请使用DINOv3模型，当前配置: {self.config.BACKBONE}")

    def _load_dinov3(self):
        """🔥 从本地加载DINOv3模型"""
        print("\n" + "=" * 60)
        print("📥 从本地加载 DINOv3 模型")
        print("=" * 60)

        repo_root = self.config.DINOV3_REPO_ROOT
        weight_path = self.config.DINOV3_WEIGHT_PATH

        # 验证路径
        if not os.path.exists(repo_root):
            raise FileNotFoundError(f"DINOv3仓库路径不存在: {repo_root}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        print(f"   📂 仓库路径: {repo_root}")
        print(f"   📂 权重路径: {weight_path}")

        # 添加路径
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            # 方法1：直接导入（适用于新版DINOv3）
            print("\n   🔄 尝试方法1: 直接导入模型...")
            from dinov3.models.vision_transformer import vit_large

            # 创建模型
            model = vit_large(
                patch_size=self.config.DINOV3_PATCH_SIZE,
                num_register_tokens=0,
                interpolate_antialias=False,
                interpolate_offset=0.1,
            )

            # 加载权重
            print(f"   📥 加载权重...")
            state_dict = torch.load(weight_path, map_location='cpu')

            # 处理权重键名（可能需要去掉'teacher'前缀）
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # 去掉'module.'前缀（如果有）
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '').replace('backbone.', '')
                new_state_dict[new_key] = v

            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"   ✅ 权重加载完成")
            if msg.missing_keys:
                print(f"   ⚠️  缺失键: {msg.missing_keys[:5]}...")
            if msg.unexpected_keys:
                print(f"   ⚠️  多余键: {msg.unexpected_keys[:5]}...")

        except Exception as e1:
            print(f"   ❌ 方法1失败: {e1}")
            print(f"\n   🔄 尝试方法2: 使用torch.load直接加载...")

            try:
                # 方法2：如果权重文件包含完整模型
                checkpoint = torch.load(weight_path, map_location='cpu')

                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model = checkpoint['model']
                    elif 'teacher' in checkpoint:
                        model = checkpoint['teacher']
                    else:
                        raise ValueError("无法从checkpoint中找到模型")
                else:
                    model = checkpoint

                print(f"   ✅ 直接加载模型成功")

            except Exception as e2:
                print(f"   ❌ 方法2也失败: {e2}")

                # 方法3：最后的尝试 - 使用dinov3仓库的builder
                print(f"\n   🔄 尝试方法3: 使用官方builder...")
                try:
                    # 这个需要根据你的dinov3仓库结构调整
                    import dinov3
                    from dinov3.models import build_model

                    # 构建配置
                    model_config = {
                        'arch': 'vit_large',
                        'patch_size': self.config.DINOV3_PATCH_SIZE,
                        'pretrained_weights': weight_path,
                    }

                    model = build_model(model_config)
                    print(f"   ✅ 使用builder加载成功")

                except Exception as e3:
                    raise RuntimeError(
                        f"所有加载方法都失败了！\n"
                        f"方法1: {e1}\n"
                        f"方法2: {e2}\n"
                        f"方法3: {e3}\n"
                        f"请检查:\n"
                        f"1. DINOv3仓库结构是否完整\n"
                        f"2. 权重文件是否匹配模型架构\n"
                        f"3. 依赖包是否安装完整"
                    )

        # 移动到设备
        model = model.to(self.config.DEVICE)

        # 注册特征提取hook
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
        """前向传播 - DINOv3专用"""
        # DINOv3的输出是 [B, N+1, D]
        # N = (H/patch_size) * (W/patch_size)
        # +1 是CLS token

        _ = self.model(x)

        # 提取并处理特征
        output_features = {}
        B = x.shape[0]
        patch_size = self.config.DINOV3_PATCH_SIZE
        H = W = self.config.IMAGE_SIZE // patch_size  # 224/16 = 14

        for layer_name, feat in self.features.items():
            # feat shape: [B, N+1, D]
            # 去掉CLS token: [B, N, D]
            feat = feat[:, 1:, :]
            D = feat.shape[-1]

            # reshape为 [B, D, H, W]
            feat = feat.transpose(1, 2).reshape(B, D, H, W)
            output_features[layer_name] = feat

        return output_features

def get_optimized_dataloader(dataset, config, shuffle=False):
    """统一的DataLoader配置"""
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,  # 12个worker
        pin_memory=True,                 # ✅ GPU加速
        persistent_workers=True,         # ✅ worker持久化
        prefetch_factor=2,               # ✅ 每个worker预加载2个batch
        drop_last=False
    )
def extract_features(dataloader, feature_extractor, config):
    """提取多层patch特征"""
    print("\n" + "=" * 60)
    print("📌 [步骤2] 提取多层特征")
    print("=" * 60)

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
                # 统一调整到16x16大小
                feat = F.adaptive_avg_pool2d(feat, (16, 16))
                embeddings.append(feat)

            # 拼接
            embedding = torch.cat(embeddings, dim=1)  # [B, C_total, 16, 16]
            B, C, H, W = embedding.shape

            # 转换为 [B, H*W, C]
            embedding = embedding.permute(0, 2, 3, 1).reshape(B, H * W, C)

            all_features.append(embedding.cpu().numpy())
            all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)  # [N, 256, C]
    print(f"   ✅ 提取特征形状: {all_features.shape}")
    print(f"   ✅ 特征维度: {all_features.shape[-1]} (多层融合)")

    return all_features, all_paths


def clean_gallery_with_lof(features, paths, config):
    """使用LOF清洗"""
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
    """🔥 优化1：贪心Coreset采样"""
    print("\n" + "=" * 60)
    print(f"📌 [步骤3] Coreset采样 (保留 {ratio * 100:.1f}%)")
    print("=" * 60)

    N, D = features.shape
    target_n = max(int(N * ratio), 1000)  # 至少保留1000个

    if target_n >= N:
        print(f"   ⚠️  目标数量({target_n}) >= 总数({N})，跳过采样")
        return np.arange(N)

    selected_indices = [np.random.randint(N)]
    min_distances = np.full(N, np.inf)

    for _ in tqdm(range(1, target_n), desc="   贪心采样"):
        last_selected = features[selected_indices[-1]]
        distances = np.linalg.norm(features - last_selected, axis=1)
        min_distances = np.minimum(min_distances, distances)
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

    print(f"   ✅ 采样完成: {len(selected_indices)} / {N} patches")
    return np.array(selected_indices)


def build_faiss_index(features, config):
    """构建FAISS索引"""
    print("\n   🔨 构建FAISS索引...")

    N_images, N_patches, D = features.shape
    patch_features = features.reshape(-1, D).astype('float32')

    print(f"   📊 原始patch数: {patch_features.shape[0]:,}")

    # Coreset采样
    if config.USE_CORESET and patch_features.shape[0] > 10000:
        coreset_indices = greedy_coreset_sampling(patch_features, config.CORESET_RATIO)
        patch_features = patch_features[coreset_indices]

    # 归一化
    faiss.normalize_L2(patch_features)

    # 构建索引
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
    """🔥 优化2：PatchCore软对齐"""
    print("\n" + "=" * 60)
    print("📌 [步骤4-6] 计算异常分数 (PatchCore方式)")
    print("=" * 60)

    N_test, N_patches, D = test_features.shape
    anomaly_scores = []
    anomaly_maps = []

    for i in tqdm(range(N_test), desc="   计算分数"):
        query = test_features[i].astype('float32')
        faiss.normalize_L2(query)

        # k-NN搜索
        distances, _ = faiss_index.search(query, config.N_NEIGHBORS)

        # 每个patch的异常分数
        patch_scores = distances.mean(axis=1)

        # 图像级分数
        image_score = patch_scores.max()
        anomaly_scores.append(image_score)

        # 热力图
        h = w = int(np.sqrt(N_patches))
        anomaly_map = patch_scores.reshape(h, w)
        anomaly_maps.append(anomaly_map)

    scores = np.array(anomaly_scores)
    print(f"   ✅ 异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   ✅ 异常分数均值: {scores.mean():.4f}")

    return scores, anomaly_maps


def generate_anomaly_heatmap(anomaly_map, original_img_path, save_path):
    """生成异常热力图"""
    img = cv2.imread(original_img_path)
    if img is None:
        print(f"⚠️  无法读取图片: {original_img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 上采样
    heatmap = cv2.resize(anomaly_map, (W, H))
    heatmap = gaussian_filter(heatmap, sigma=4)

    # 归一化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # 颜色映射
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加
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
        print("🚀 开始训练流程")
        print("=" * 60)

        # 步骤1
        print(f"\n📂 加载初始正常图: {self.config.INITIAL_NORMAL_DIR}")
        initial_dataset = ImageDataset(self.config.INITIAL_NORMAL_DIR, self.transform)
        initial_loader = get_optimized_dataloader(
            initial_dataset, self.config, shuffle=True
        )  # ✅ 使用优化的loader

        ae_model = train_autoencoder(initial_loader, self.config)

        print(f"\n📂 加载混合图: {self.config.MIXED_DIR}")
        mixed_dataset = ImageDataset(self.config.MIXED_DIR, self.transform)
        mixed_loader = get_optimized_dataloader(
            mixed_dataset, self.config, shuffle=False
        )  # ✅ 使用优化的loader

        expanded_paths, _ = expand_normal_gallery(ae_model, mixed_loader, self.config)

        # 步骤2
        all_normal_paths = initial_dataset.image_paths + expanded_paths


        gallery_dataset = PathDataset(all_normal_paths, self.transform)
        gallery_loader = get_optimized_dataloader(
            gallery_dataset, self.config, shuffle=False
        )  # ✅ 使用优化的loader

        feature_extractor = FeatureExtractor(self.config).to(self.config.DEVICE)

        gallery_features, gallery_paths = extract_features(
            gallery_loader, feature_extractor, self.config
        )

        clean_features, clean_paths = clean_gallery_with_lof(
            gallery_features, gallery_paths, self.config
        )

        # 步骤3
        faiss_index, memory_bank = build_faiss_index(clean_features, self.config)

        # 保存
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
        """推理"""
        print("\n" + "=" * 60)
        print("🔍 开始异常检测")
        print("=" * 60)

        test_dataset = ImageDataset(test_dir, self.transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )

        test_features, test_paths = extract_features(
            test_loader, feature_extractor, self.config
        )

        scores, anomaly_maps = compute_anomaly_scores(
            test_features, faiss_index, self.config
        )

        # 动态阈值
        threshold = np.percentile(scores, self.config.ANOMALY_PERCENTILE)
        print(f"\n   🎯 动态阈值 ({self.config.ANOMALY_PERCENTILE}%分位): {threshold:.4f}")

        # 排序
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
            for r in results[:100]:
                status = '🔴 ANOMALY' if r['is_anomaly'] else '🟢 NORMAL'
                f.write(f"{r['rank']:<6} {r['score']:<12.6f} {status:<12} {r['path']}\n")

        print(f"\n   ✅ 结果已保存: {result_path}")

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


# ==================== 🔥 优化4：在线学习 ====================
def incremental_update(new_normal_dir: str, model_path: str):
    """增量更新Normal Gallery"""
    print("\n" + "=" * 60)
    print("🔄 增量更新Normal Gallery")
    print("=" * 60)

    # 加载模型
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    faiss_index = faiss.deserialize_index(data['faiss_index'])
    clean_paths = data['clean_paths']
    config = data['config']

    print(f"   📂 当前Gallery: {len(clean_paths)} 张")

    # 提取新样本特征
    feature_extractor = FeatureExtractor(config).to(config.DEVICE)

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    new_dataset = ImageDataset(new_normal_dir, transform)
    new_loader = DataLoader(
        new_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    new_features, new_paths = extract_features(new_loader, feature_extractor, config)

    # 添加到索引
    N, P, D = new_features.shape
    new_patch_features = new_features.reshape(-1, D).astype('float32')
    faiss.normalize_L2(new_patch_features)
    faiss_index.add(new_patch_features)

    # 更新路径
    clean_paths.extend(new_paths)

    # 保存
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
    # 配置
    config = Config()

    # ✅ 验证路径
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

    # 检查路径
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
    if os.path.exists(config.TEST_DIR):  # ✅ 使用config.TEST_DIR
        results = pipeline.inference(config.TEST_DIR, feature_extractor, faiss_index)

        # 打印Top 10
        print("\n" + "=" * 60)
        print("🔝 Top 10 最可疑异常")
        print("=" * 60)
        for r in results[:10]:
            status = '🔴' if r['is_anomaly'] else '🟢'
            print(f"{status} Rank {r['rank']}: {r['score']:.6f} - {os.path.basename(r['path'])}")
    else:
        print(f"⚠️  测试路径不存在，跳过推理")
