"""主运行脚本"""

import argparse
from config import Config
from core import AnomalyScreener


def main():
    parser = argparse.ArgumentParser(description='异常样本筛选（供人工标注）')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='图像目录（待筛选的所有图像）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--top_k', type=int, default=None,
                        help='筛选Top-K个样本')
    parser.add_argument('--top_percent', type=float, default=None,
                        help='筛选前X%的样本（0-1之间）')
    args = parser.parse_args()

    # 更新配置
    config = Config()
    if args.image_dir:
        config.IMAGE_DIR = args.image_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.top_k:
        config.TOP_K = args.top_k
        config.USE_TOP_K = True
    if args.top_percent:
        config.TOP_PERCENT = args.top_percent
        config.USE_TOP_K = False

    print("\n" + "=" * 70)
    print("配置信息")
    print("=" * 70)
    print(f"图像目录: {config.IMAGE_DIR}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"模型: {config.MODEL_NAME}")
    print(f"筛选方法: {', '.join(config.METHODS)}")
    if config.USE_TOP_K:
        print(f"筛选数量: Top-{config.TOP_K}")
    else:
        print(f"筛选比例: {config.TOP_PERCENT * 100}%")
    print(f"多样性采样: {'是' if config.USE_DIVERSITY_SAMPLING else '否'}")
    print("=" * 70 + "\n")

    # 初始化筛选器
    screener = AnomalyScreener(config)

    # 提取特征
    features, paths = screener.extract_all_features()

    # 计算异常分数
    scores_dict = screener.compute_anomaly_scores()

    # 筛选样本
    selected_indices = screener.select_samples_for_annotation(
        scores_dict['ensemble']
    )

    # 保存结果
    result_path = screener.save_results(scores_dict, selected_indices)

    print("\n" + "=" * 70)
    print("✅ 筛选完成!")
    print("=" * 70)
    print(f"\n📋 下一步:")
    print(f"   1. 打开文件: {result_path}")
    print(f"   2. 在 'annotation_label' 列填写标注结果")
    print(f"   3. 在 'annotation_notes' 列添加备注")
    print(f"\n💡 提示: 图像按异常分数从高到低排序")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
