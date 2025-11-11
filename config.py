"""配置文件（性能优化版）"""
import torch

class Config:
    # ==================== 路径配置 ====================
    # DINOv3模型路径
    DINOV3_REPO_ROOT = r"D:\ly\dinov3-main"
    DINOV3_WEIGHT_PATH = r"D:\ly\dinov3-main\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    MODEL_NAME = "dinov3_vitl16_lvd1689m_distilled"

    # 数据路径
    IMAGE_DIR = r"D:\Huangda_data0801\view\4\E"      # 待筛选的所有图像
    OUTPUT_DIR = "outputs_4_E"# 输出目录

    # ⚡ 缓存目录（第二次运行秒级完成）
    CACHE_DIR = "cache"

    # ==================== 模型配置 ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ⚡⚡⚡ 性能优化配置
    BATCH_SIZE = 64                      # ⬆️ 16→32 (根据显存调整: 16/32/64)
    IMAGE_SIZE = 518                     # vitl16原生尺寸（保持不变）
    NUM_WORKERS = 8                      # ⬆️ 4→6 (建议CPU核心数的50-75%)

    # ⚡ 混合精度训练（2x加速，显存减半）
    USE_AMP = False

    # ⚡⚡ 特征提取策略（重要！）
    USE_CLS_TOKEN = False                 # True: 快10倍 | False: 精度稍高

    # ⚡ PyTorch 2.0编译加速（需PyTorch>=2.0，可能不稳定）
    USE_COMPILE = False

    # ==================== 筛选策略 ====================
    # ⚡ 筛选方法（移除最慢的lof和density）
    METHODS = [
        'knn',           # K近邻距离 (快)
        'isolation',     # 孤立森林 (快)
        'lof',         # ❌ 局部离群因子 (慢，大数据集>10K不推荐)
        'density',     # ❌ 核密度估计 (慢)
    ]

    # KNN参数
    KNN_K = 10                          # K值（可保持）

    # LOF参数（如启用）
    LOF_NEIGHBORS = 20                  # 邻居数

    # Isolation Forest参数
    ISO_CONTAMINATION = 0.1             # 预期异常比例

    # 密度估计参数（如启用）
    DENSITY_BANDWIDTH = 'scott'         # 带宽选择

    # ==================== 输出配置 ====================
    # 筛选数量/比例
    TOP_K = 500                         # 输出Top-K异常样本
    TOP_PERCENT = 0.1                   # 或输出前10%（二选一）
    USE_TOP_K = True                    # True用TOP_K，False用TOP_PERCENT

    # 聚类多样性（避免筛选的都是同一类异常）
    USE_DIVERSITY_SAMPLING = True       # 是否使用多样性采样
    N_CLUSTERS = 15                     # ⬇️ 20→15 聚类数量（减少计算量）
    SAMPLES_PER_CLUSTER = 25            # 每类采样数

    # 特征处理
    USE_PCA = True                      # 是否PCA降维
    PCA_DIM = 256                       # PCA维度（合理）
    NORMALIZE_FEATURES = True           # 是否L2归一化

    # 可视化
    SAVE_VISUALIZATIONS = True
    VIZ_TOP_K = 100                     # 可视化前N个
    GRID_COLS = 5                       # 网格列数

    # ==================== 高级优化 ====================
    # ⚡ FAISS加速（需安装faiss-gpu或faiss-cpu）
    USE_FAISS = True                    # 自动检测是否安装

    # ⚡ 大数据集采样策略（图像>20K时启用）
    USE_SAMPLING_FOR_LOF = True         # LOF采样计算
    LOF_SAMPLE_SIZE = 10000             # 采样数量
    USE_GPU_PREPROCESSING = True
    PREFETCH_FACTOR = 2
    # ⚡⚡ 批量GPU处理（新增）
    BATCH_EXTRACT_ON_GPU = True   # 🔥 特征提取全程GPU



class SpeedConfig(Config):
    """极速模式（牺牲少量精度）"""
    BATCH_SIZE = 160              # 更大batch
    IMAGE_SIZE = 224              # 降低分辨率 518→224
    USE_CLS_TOKEN = True
    METHODS = ['knn']             # 只用最快的方法
    USE_DIVERSITY_SAMPLING = False
    SAVE_VISUALIZATIONS = False
    NUM_WORKERS = 16


class BalanceConfig(Config):
    """平衡模式（推荐）"""
    BATCH_SIZE = 128
    IMAGE_SIZE = 518
    USE_CLS_TOKEN = True
    METHODS = ['knn', 'isolation']
    NUM_WORKERS = 12


class AccuracyConfig(Config):
    """精度模式（最高质量）"""
    BATCH_SIZE = 32
    IMAGE_SIZE = 518
    USE_CLS_TOKEN = False         # Patch平均，更精细
    METHODS = ['knn', 'lof', 'isolation']
    NUM_WORKERS = 8
