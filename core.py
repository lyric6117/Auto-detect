"""核心筛选逻辑（严格遵循DINOv3 demo）"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import datetime
from typing import List, Dict, Tuple
from torchvision import transforms

class AnomalyScreener:
    """异常样本筛选器（供人工标注使用）"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print("="*70)
        print("🎯 异常样本筛选系统（供人工标注）")
        print("="*70)
        print(f"设备: {self.device}")

        # ✅ 初始化标准预处理管道（遵循demo）
        self.transform = self._get_dinov3_transform()

        # 加载DINOv3模型
        self.model = self._load_dinov3()

        # 数据存储
        self.all_features = None
        self.all_paths = None
        self.pca = None
        self.scaler = None

    def _get_dinov3_transform(self):
        """✅ DINOv3官方推荐的预处理管道（完全遵循demo）"""
        return transforms.Compose([
            transforms.Resize(
                (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),  # 自动转为float32并归一化到[0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_dinov3(self):
        """加载DINOv3模型（严格遵循demo）"""
        print(f"\n📥 加载DINOv3模型: {self.config.MODEL_NAME}")

        # 添加DINOv3路径
        repo_root = self.config.DINOV3_REPO_ROOT
        sys.path.insert(0, repo_root)

        # ✅ 正确的导入方式
        import dinov3.distributed as distributed
        from dinov3.configs import setup_config, DinoV3SetupArgs, setup_job
        from dinov3.models import build_model_for_eval

        # 配置路径
        config_path = os.path.join(
            repo_root, "dinov3", "configs", "train", f"{self.config.MODEL_NAME}.yaml"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 初始化分布式环境
        output_dir = "./outputs/dinov3_tmp"
        os.makedirs(output_dir, exist_ok=True)

        setup_job(
            output_dir=output_dir,
            distributed_enabled=False,
            seed=42,
            distributed_timeout=datetime.timedelta(minutes=30)
        )

        # 创建配置
        setup_args = DinoV3SetupArgs(
            config_file=config_path,
            pretrained_weights=self.config.DINOV3_WEIGHT_PATH,
            output_dir=output_dir,
            opts=[]
        )

        # 加载配置和模型
        cfg = setup_config(setup_args, strict_cfg=False)
        model = build_model_for_eval(
            config=cfg,
            pretrained_weights=self.config.DINOV3_WEIGHT_PATH
        )

        # 设置为评估模式
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False

        print(f"✅ DINOv3模型加载完成")
        print(f"   进程数: {distributed.get_world_size()}")

        return model

    def _get_image_paths(self, directory):
        """获取所有图像路径"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        paths = []
        directory = Path(directory)
        for ext in extensions:
            paths.extend(directory.glob(f"*{ext}"))
            paths.extend(directory.rglob(f"*{ext}"))  # 包含子目录
        # 去重并排序
        paths = sorted(list(set(paths)))
        return [str(p) for p in paths]

    @torch.no_grad()
    def extract_all_features(self, image_dir=None):
        """✅ 提取所有图像特征（完全遵循demo的批量处理方式）"""
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR

        print("\n" + "="*70)
        print("🔍 提取图像特征")
        print("="*70)

        # 1. 获取所有图像
        image_paths = self._get_image_paths(image_dir)
        print(f"📂 找到图像: {len(image_paths)} 张")

        if len(image_paths) == 0:
            raise ValueError(f"未找到图像: {image_dir}")

        # 2. 批量提取特征（遵循demo）
        features_list = []
        valid_paths = []

        for i in tqdm(range(0, len(image_paths), self.config.BATCH_SIZE),
                     desc="提取特征"):
            batch_paths = image_paths[i:i + self.config.BATCH_SIZE]
            batch_images = []
            batch_valid_paths = []

            # ✅ 批量加载图像（demo方式）
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)  # 使用transforms管道
                    batch_images.append(img_tensor)
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️  跳过 {path}: {e}")
                    continue

            if len(batch_images) == 0:
                continue

            # ✅ 批量推理（demo方式）
            batch_tensor = torch.stack(batch_images).to(self.device)

            # ✅ 使用forward_features提取（demo方式）
            with torch.no_grad():
                feats_dict = self.model.forward_features(batch_tensor)

                # ✅ 提取patch tokens特征（demo推荐）
                if 'x_norm_patchtokens' in feats_dict:
                    feats = feats_dict['x_norm_patchtokens']
                elif 'x_prenorm' in feats_dict:
                    feats = feats_dict['x_prenorm']
                else:
                    # 如果都没有，尝试取第一个tensor
                    feats = list(feats_dict.values())[0]

                # ✅ L2归一化（demo推荐）
                if self.config.NORMALIZE_FEATURES:
                    feats = F.normalize(feats, dim=-1)

                # ✅ Flatten patch维度（demo方式）
                # feats shape: (batch, n_patches, dim)
                if feats.dim() == 3:
                    batch_size, n_patches, dim = feats.shape
                    feats = feats.reshape(batch_size * n_patches, dim)

                features_list.append(feats.cpu().numpy())

            valid_paths.extend(batch_valid_paths)

        # 3. 合并特征
        features = np.vstack(features_list)
        self.all_paths = valid_paths

        print(f"✅ 特征提取完成: {features.shape}")

        # 4. 可选的PCA降维
        if self.config.USE_PCA:
            # 计算每张图的patch数量
            n_images = len(valid_paths)
            n_total_patches = features.shape[0]
            patches_per_image = n_total_patches // n_images

            print(f"📊 特征统计: {n_images} 张图像, 每张 {patches_per_image} 个patches")

            # 先对patch特征做平均，得到图像级特征
            features_img = features.reshape(n_images, patches_per_image, -1).mean(axis=1)

            pca_dim = min(self.config.PCA_DIM, features_img.shape[1], len(features_img) - 1)
            print(f"📉 PCA降维: {features_img.shape[1]} → {pca_dim}")

            self.pca = PCA(n_components=pca_dim, random_state=42)
            features_img = self.pca.fit_transform(features_img)

            var_ratio = self.pca.explained_variance_ratio_.sum()
            print(f"   解释方差比: {var_ratio:.2%}")

            self.all_features = features_img.astype(np.float32)
        else:
            # 不降维：直接对patch做平均得到图像级特征
            n_images = len(valid_paths)
            n_total_patches = features.shape[0]
            patches_per_image = n_total_patches // n_images

            features_img = features.reshape(n_images, patches_per_image, -1).mean(axis=1)
            self.all_features = features_img.astype(np.float32)

        print(f"📊 最终特征: {self.all_features.shape}")

        return self.all_features, self.all_paths

    def compute_anomaly_scores(self):
        """计算异常分数（多种方法融合）"""
        print("\n" + "="*70)
        print("💯 计算异常分数")
        print("="*70)

        scores_dict = {}
        methods = self.config.METHODS

        # 1. KNN距离
        if 'knn' in methods:
            print("📊 KNN距离...")
            scores_dict['knn'] = self._compute_knn_score()

        # 2. LOF
        if 'lof' in methods:
            print("📊 局部离群因子(LOF)...")
            scores_dict['lof'] = self._compute_lof_score()

        # 3. Isolation Forest
        if 'isolation' in methods:
            print("📊 孤立森林...")
            scores_dict['isolation'] = self._compute_isolation_score()

        # 4. 核密度估计
        if 'density' in methods:
            print("📊 核密度估计...")
            scores_dict['density'] = self._compute_density_score()

        # 5. 融合分数
        print("🔄 融合异常分数...")
        ensemble_score = self._ensemble_scores(scores_dict)
        scores_dict['ensemble'] = ensemble_score

        # 统计信息
        print(f"\n📈 分数统计:")
        for name, scores in scores_dict.items():
            print(f"   {name:12s}: mean={scores.mean():.4f}, "
                  f"std={scores.std():.4f}, "
                  f"max={scores.max():.4f}")

        return scores_dict

    def _compute_knn_score(self):
        """KNN距离分数"""
        k = min(self.config.KNN_K, len(self.all_features) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
        nbrs.fit(self.all_features)
        distances, _ = nbrs.kneighbors(self.all_features)

        # 排除自身，取均值
        knn_dist = distances[:, 1:].mean(axis=1)

        return knn_dist

    def _compute_lof_score(self):
        """LOF分数"""
        n_neighbors = min(self.config.LOF_NEIGHBORS, len(self.all_features) - 1)

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination='auto',
            n_jobs=-1
        )

        lof.fit(self.all_features)
        # 负的离群因子，转为正值（越大越异常）
        lof_scores = -lof.negative_outlier_factor_

        return lof_scores

    def _compute_isolation_score(self):
        """孤立森林分数"""
        iso = IsolationForest(
            contamination=self.config.ISO_CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )

        iso.fit(self.all_features)
        # 异常分数（越负越异常，转为正值）
        iso_scores = -iso.score_samples(self.all_features)

        return iso_scores

    def _compute_density_score(self):
        """核密度估计分数"""
        kde = KernelDensity(
            bandwidth=self.config.DENSITY_BANDWIDTH,
            kernel='gaussian'
        )
        kde.fit(self.all_features)

        # 对数似然（越低越异常，取负）
        log_density = kde.score_samples(self.all_features)
        density_scores = -log_density

        return density_scores

    def _ensemble_scores(self, scores_dict):
        """融合多个分数"""
        if len(scores_dict) == 0:
            raise ValueError("没有可用的异常分数")

        # 归一化每个分数到[0, 1]
        normalized = []

        for name, scores in scores_dict.items():
            # Min-Max归一化
            min_val = scores.min()
            max_val = scores.max()

            if max_val > min_val:
                norm_scores = (scores - min_val) / (max_val - min_val)
            else:
                norm_scores = np.zeros_like(scores)

            normalized.append(norm_scores)

        # 简单平均融合
        ensemble = np.mean(normalized, axis=0)

        return ensemble

    def select_samples_for_annotation(self, scores):
        """选择样本供人工标注"""
        print("\n" + "="*70)
        print("🎯 筛选异常样本")
        print("="*70)

        # 确定筛选数量
        if self.config.USE_TOP_K:
            n_select = min(self.config.TOP_K, len(scores))
        else:
            n_select = int(len(scores) * self.config.TOP_PERCENT)

        print(f"📌 筛选策略: {'Top-K' if self.config.USE_TOP_K else '百分比'}")
        print(f"📌 筛选数量: {n_select} / {len(scores)} ({n_select/len(scores)*100:.1f}%)")

        if self.config.USE_DIVERSITY_SAMPLING:
            # 多样性采样：先聚类，再从每类选异常样本
            selected_indices = self._diversity_sampling(scores, n_select)
        else:
            # 简单Top-K
            selected_indices = np.argsort(scores)[-n_select:][::-1]

        print(f"✅ 筛选完成: {len(selected_indices)} 个样本")

        return selected_indices

    def _diversity_sampling(self, scores, n_select):
        """多样性采样（避免只选同一类异常）"""
        print("🎨 使用多样性采样...")

        # 1. 聚类
        n_clusters = min(self.config.N_CLUSTERS, len(self.all_features) // 10)
        n_clusters = max(2, n_clusters)  # 至少2个类

        print(f"   聚类数: {n_clusters}")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1024, len(self.all_features)),
            n_init=3
        )
        cluster_labels = kmeans.fit_predict(self.all_features)

        # 2. 每个类选异常分数最高的样本
        samples_per_cluster = max(1, n_select // n_clusters)

        selected_indices = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # 在该类中选异常分数最高的
            cluster_scores = scores[cluster_indices]
            n_select_cluster = min(samples_per_cluster, len(cluster_indices))
            top_in_cluster = np.argsort(cluster_scores)[-n_select_cluster:][::-1]

            selected_indices.extend(cluster_indices[top_in_cluster])

        # 3. 如果不够，补充全局Top
        if len(selected_indices) < n_select:
            remaining = n_select - len(selected_indices)
            all_top = np.argsort(scores)[::-1]

            for idx in all_top:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= n_select:
                        break

        # 4. 按分数排序
        selected_indices = np.array(selected_indices[:n_select])
        selected_scores = scores[selected_indices]
        sort_order = np.argsort(selected_scores)[::-1]
        selected_indices = selected_indices[sort_order]

        print(f"   采样分布: 每类约{samples_per_cluster}个")

        return selected_indices

    def save_results(self, scores_dict, selected_indices):
        """保存筛选结果"""
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("💾 保存结果")
        print("="*70)

        # 1. 完整结果CSV（所有图像）
        df_all = pd.DataFrame({
            'image_path': self.all_paths,
            'image_name': [Path(p).name for p in self.all_paths],
            'ensemble_score': scores_dict['ensemble'],
        })

        # 添加各指标分数
        for name, scores in scores_dict.items():
            if name != 'ensemble':
                df_all[f'{name}_score'] = scores

        # 添加是否被选中标记
        df_all['selected_for_annotation'] = 0
        df_all.loc[selected_indices, 'selected_for_annotation'] = 1

        # 按分数排序
        df_all = df_all.sort_values('ensemble_score', ascending=False)

        csv_all = output_dir / 'all_images_with_scores.csv'
        df_all.to_csv(csv_all, index=False, encoding='utf-8-sig')
        print(f"📄 完整结果: {csv_all}")

        # 2. 筛选出的异常样本CSV（供标注）
        df_selected = df_all[df_all['selected_for_annotation'] == 1].copy()
        df_selected['annotation_label'] = ''  # 空列供人工填写
        df_selected['annotation_notes'] = ''  # 备注列

        csv_selected = output_dir / 'samples_for_annotation.csv'
        df_selected.to_csv(csv_selected, index=False, encoding='utf-8-sig')
        print(f"📄 待标注样本: {csv_selected} ({len(df_selected)} 个)")

        # 3. 统计摘要
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

        # 4. 可视化
        if self.config.SAVE_VISUALIZATIONS:
            self._visualize_results(scores_dict, selected_indices, output_dir)

        print(f"\n📁 所有结果已保存到: {output_dir}")

        return csv_selected

    def _visualize_results(self, scores_dict, selected_indices, output_dir):
        """可视化结果"""
        print("\n📊 生成可视化...")

        scores = scores_dict['ensemble']

        # 1. 分数分布图
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=100, alpha=0.7, color='skyblue', edgecolor='black')

        # 标记选中样本的分数范围
        selected_scores = scores[selected_indices]
        threshold = selected_scores.min()
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Selection Threshold: {threshold:.3f}')

        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Score Distribution (All Images)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 选中样本的分数分布
        plt.subplot(1, 2, 2)
        plt.hist(selected_scores, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Score Distribution (Selected for Annotation)', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 分数分布图")

        # 3. Top异常样本可视化
        viz_k = min(self.config.VIZ_TOP_K, len(selected_indices))
        top_indices = selected_indices[:viz_k]

        n_cols = self.config.GRID_COLS
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            axes[i].imshow(img)
            axes[i].set_title(f'Rank {i+1}\nScore: {score:.4f}\n{Path(img_path).name}',
                            fontsize=9)
            axes[i].axis('off')

        # 隐藏多余子图
        for i in range(len(top_indices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'top_{viz_k}_anomalies.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Top-{viz_k} 异常样本图")
