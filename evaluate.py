"""
Reward Model 评估脚本。

用法:
    python evaluate.py --checkpoint /ssd/cyfu/yuhan/reward_model/outputs/best.pth

功能:
    1. 加载训练好的 checkpoint
    2. 在验证集上计算 accuracy / precision / recall / F1
    3. 可视化一些预测样本 (可选)
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets
from model import ResnetRewModel


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        probs = model.forward_train(images).squeeze(-1).cpu()  # [B], 连续值
        preds = (probs > 0.5).float()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 指标
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    acc = correct / total

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("=" * 50)
    print("Evaluation Results:")
    print(f"  Total samples: {total}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print("=" * 50)

    # 概率分布
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    if len(pos_probs) > 0:
        print(f"  Positive samples prob: mean={pos_probs.mean():.4f} "
              f"std={pos_probs.std():.4f} min={pos_probs.min():.4f} max={pos_probs.max():.4f}")
    if len(neg_probs) > 0:
        print(f"  Negative samples prob: mean={neg_probs.mean():.4f} "
              f"std={neg_probs.std():.4f} min={neg_probs.min():.4f} max={neg_probs.max():.4f}")

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device)

    # 数据
    print("Building datasets...")
    _, val_ds = build_datasets(cfg)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # 模型
    print(f"Loading model from {args.checkpoint}")
    model = ResnetRewModel(checkpoint_path=args.checkpoint)
    model = model.to(device)

    # 评估
    results = evaluate_model(model, val_loader, device)

    # 保存结果
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
