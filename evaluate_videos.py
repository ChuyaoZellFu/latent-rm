"""
Latent 级别任务成功判断脚本。

用法:
python reward_model_latent/evaluate_videos.py \
    --checkpoint /ssd/cyfu/yuhan/reward_model_latent/outputs/best.pth \
    --latent_dir /ssd/cyfu/yuhan/latent_eval_place_container

功能:
    1. 扫描 latent_dir 下所有 batch 目录中的 latent 文件
    2. 加载 gt_latents.pth 和 pred_latents.pth
    3. 使用 LatentRewModel 分别判断 GT 和 Pred 的成功/失败
    4. 汇总结果，计算 Pred 与 GT 的相似度指标

判断逻辑:
    对于机器人任务，成功与否取决于最终状态，因此使用末尾帧（最后 tail_ratio
    比例的帧）的平均成功概率作为序列级别的判断依据，而非取所有帧的最大值。
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from model import LatentRewModel


@torch.no_grad()
def evaluate_latents(model, latent, device, threshold=0.5, tail_ratio=0.25,
                     judgment="max", debug=False, debug_label=""):
    """
    对单条 latent 序列进行推理，返回序列级别的成功判断。

    judgment 参数控制判断依据：
        "max"  (默认): max_prob — 全帧最大成功概率 > threshold。
                适用于短预测窗口数据（如 latent_eval，T≈56 帧），
                成功状态可能出现在窗口任意位置。
        "tail": tail_mean — 末尾 tail_ratio 比例帧的平均成功概率 > threshold。
                适用于完整 episode 数据（如 task_eval，成功 T≈42 帧终止，
                失败 T=195 帧跑完全程），避免长失败序列因某中间帧偶然得高分
                而被误判。

    参数:
        latent: [T, P, D]  — 形状已经过 reshape 归一化
        tail_ratio: 末尾帧比例（judgment="tail" 时作为判断依据）
        judgment: "max" 或 "tail"
        debug: 打印逐帧概率分布

    返回:
        is_success: bool
        score: float, 判断依据的分数（max_prob 或 tail_mean）
        all_probs: np.ndarray [T], 所有帧的成功概率（供分析用）
    """
    model.eval()
    latent = latent.to(device)  # [T, P, D]
    all_probs = model.forward_train(latent).squeeze(-1).cpu().numpy()  # [T]

    T = len(all_probs)
    tail_start = max(0, T - max(1, int(T * tail_ratio)))
    tail_mean = float(all_probs[tail_start:].mean())
    max_prob = float(all_probs.max())

    if judgment == "tail":
        score = tail_mean
    else:
        score = max_prob
    is_success = score > threshold

    if debug:
        n_show = min(10, T)
        print(f"\n    [{debug_label}] T={T}, mean={all_probs.mean():.4f}, "
              f"max={max_prob:.4f} @frame{all_probs.argmax()}, "
              f"tail({tail_ratio:.0%}) mean={tail_mean:.4f}  "
              f"[judgment={judgment}] -> {'SUCCESS' if is_success else 'FAILURE'}")
        print(f"      First {n_show} frames: " +
              " ".join(f"{p:.3f}" for p in all_probs[:n_show]))
        print(f"      Last  {n_show} frames: " +
              " ".join(f"{p:.3f}" for p in all_probs[-n_show:]))

    return is_success, score, all_probs


def find_latent_batches(latent_dir):
    """扫描 latent_dir 下所有 batch 或 episode 目录。

    支持两种结构:
      1. batch_* 结构: latent_dir/batch_000/gt_latents.pth
      2. episode_* 结构: latent_dir/episode_0000/gt_latents.pth
      3. task/episode 结构: latent_dir/<task_name>/episode_XXXX/gt_latents.pth
    """
    batch_dirs = sorted(glob.glob(os.path.join(latent_dir, "batch_*")))
    if batch_dirs:
        return batch_dirs

    episode_dirs = sorted(glob.glob(os.path.join(latent_dir, "episode_*")))
    if episode_dirs:
        return episode_dirs

    task_dirs = sorted([
        d for d in glob.glob(os.path.join(latent_dir, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith(".")
    ])
    episode_dirs = []
    for task_dir in task_dirs:
        eps = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
        episode_dirs.extend(eps)
    return episode_dirs


def load_latent_file(path, num_patches=256, emb_dim=2048):
    """
    加载 latent 文件，返回形状 [T, num_patches, emb_dim] 的 float32 tensor。

    支持任意前缀维度，例如 [1, T, 1, 256, 2048] → reshape 为 [T, 256, 2048]。
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "latents" in data:
            latents = data["latents"]
        elif "features" in data:
            latents = data["features"]
        else:
            for k, v in data.items():
                if hasattr(v, 'shape') and len(v.shape) >= 2:
                    latents = v
                    break
            else:
                raise ValueError(f"Cannot find latent tensor in {path}")
    elif isinstance(data, list):
        latents = torch.stack(data) if isinstance(data[0], torch.Tensor) else data
    elif hasattr(data, 'shape'):
        latents = data
    else:
        raise ValueError(f"Unknown latent format in {path}")

    # 将任意形状 reshape 为 [T, num_patches, emb_dim]
    # 例如 [1, T, 1, 256, 2048] → [T, 256, 2048]
    latents = latents.float()
    latents = latents.reshape(-1, num_patches, emb_dim)
    return latents


def load_metadata(latent_dir):
    """
    加载 sample_metadata.csv 或 per_episode_metrics.csv（如果存在），
    返回 {batch_idx: label} 字典。label 为 'success' 或 'failure'。

    对于 per_episode_metrics.csv，没有直接的 success/failure 标签，跳过元数据。
    """
    csv_path = os.path.join(latent_dir, "sample_metadata.csv")
    if os.path.exists(csv_path):
        meta = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch_idx = int(row["batch_idx"])
                meta[batch_idx] = row["label"]
        return meta

    # 向上一级查找 per_episode_metrics.csv（task_eval_pcp 结构）
    parent_csv = os.path.join(os.path.dirname(latent_dir), "per_episode_metrics.csv")
    if not os.path.exists(parent_csv):
        parent_csv = os.path.join(latent_dir, "per_episode_metrics.csv")
    if os.path.exists(parent_csv):
        print(f"Found per_episode_metrics.csv at {parent_csv} (no success labels, skipping metadata)")
    return None


def compute_metrics(pred_arr, gt_arr):
    """以 gt_arr 为正例标准，计算 pred_arr 的分类指标。"""
    correct = (pred_arr == gt_arr).sum()
    total = len(gt_arr)
    accuracy = correct / total
    tp = ((pred_arr == 1) & (gt_arr == 1)).sum()
    fp = ((pred_arr == 1) & (gt_arr == 0)).sum()
    fn = ((pred_arr == 0) & (gt_arr == 1)).sum()
    tn = ((pred_arr == 0) & (gt_arr == 0)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), total=total)


def print_metrics(title, m):
    print(f"\n  [{title}]")
    print(f"    Accuracy:  {m['accuracy']:.4f}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1 Score:  {m['f1']:.4f}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate task success from latents using reward model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Root directory containing batch_* subdirectories with latent files")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for success (default: 0.5)")
    parser.add_argument("--tail_ratio", type=float, default=0.25,
                        help="Fraction of trailing frames to average for sequence-level score (default: 0.25)")
    parser.add_argument("--judgment", type=str, default="max", choices=["max", "tail"],
                        help="Judgment method: 'max' (max_prob, default) or 'tail' (tail_mean). "
                             "Use 'tail' for full-episode task_eval data to avoid long-failure false positives.")
    parser.add_argument("--debug_batch", type=str, default=None,
                        help="Print per-frame prob distribution for this batch name, e.g. 'batch_000' (or 'all')")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model = LatentRewModel(checkpoint_path=args.checkpoint)
    model = model.to(device)
    score_label = "tail_mean (last {:.0%} frames)".format(args.tail_ratio) if args.judgment == "tail" else "max_prob"
    print(f"Judgment: {args.judgment}  |  Score: {score_label}  |  Threshold: {args.threshold}\n")

    # 加载元数据（如有）
    meta = load_metadata(args.latent_dir)
    has_meta = meta is not None
    if has_meta:
        print(f"Loaded sample_metadata.csv ({len(meta)} entries)\n")

    # 扫描 batch 目录
    batches = find_latent_batches(args.latent_dir)
    print(f"Found {len(batches)} batches in {args.latent_dir}\n")

    if len(batches) == 0:
        print("No batch directories found. Check --latent_dir.")
        return

    # 逐 batch 评估
    results = []
    gt_success_list = []
    pred_success_list = []
    true_label_list = []    # 来自 metadata

    hdr_label = f"{'Label':<10}" if has_meta else ""
    print("=" * 90)
    print(f"  {'Batch':<12} {hdr_label} {'GT_Succ':>8} {'GT_Tail':>8} {'Pred_Succ':>10} {'Pred_Tail':>10}")
    print("-" * 90)

    for batch_dir in batches:
        batch_name = os.path.basename(batch_dir)
        batch_idx = int(batch_name.split("_")[-1])
        gt_path = os.path.join(batch_dir, "gt_latents.pth")
        pred_path = os.path.join(batch_dir, "pred_latents.pth")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Warning: Missing latent files in {batch_name}, skipping...")
            continue

        # 加载 latent
        gt_latent = load_latent_file(gt_path)
        pred_latent = load_latent_file(pred_path)

        do_debug = (args.debug_batch == "all" or args.debug_batch == batch_name)

        # 评估 GT
        gt_is_success, gt_score, gt_all_probs = evaluate_latents(
            model, gt_latent, device, threshold=args.threshold, tail_ratio=args.tail_ratio,
            judgment=args.judgment, debug=do_debug, debug_label=f"{batch_name} GT"
        )

        # 评估 Pred
        pred_is_success, pred_score, pred_all_probs = evaluate_latents(
            model, pred_latent, device, threshold=args.threshold, tail_ratio=args.tail_ratio,
            judgment=args.judgment, debug=do_debug, debug_label=f"{batch_name} Pred"
        )

        gt_status = "YES" if gt_is_success else "NO"
        pred_status = "YES" if pred_is_success else "NO"

        true_label = meta.get(batch_idx, "?") if has_meta else None
        label_str = f"{true_label:<10}" if has_meta else ""

        print(f"  {batch_name:<12} {label_str} {gt_status:>8} {gt_score:>8.4f} {pred_status:>10} {pred_score:>10.4f}")

        record = {
            "batch": batch_name,
            "gt_is_success": gt_is_success,
            "gt_score": gt_score,
            "gt_mean_prob": float(gt_all_probs.mean()),
            "pred_is_success": pred_is_success,
            "pred_score": pred_score,
            "pred_mean_prob": float(pred_all_probs.mean()),
        }
        if has_meta:
            record["true_label"] = true_label
        results.append(record)

        gt_success_list.append(gt_is_success)
        pred_success_list.append(pred_is_success)
        if has_meta:
            true_label_list.append(1 if true_label == "success" else 0)

    gt_success_arr = np.array(gt_success_list, dtype=int)
    pred_success_arr = np.array(pred_success_list, dtype=int)

    print("=" * 90)
    print("\n=== Summary ===")
    print(f"  Total batches: {len(results)}")
    print(f"  GT   Success Rate: {gt_success_arr.mean():.1%} ({gt_success_arr.sum()}/{len(gt_success_arr)})")
    print(f"  Pred Success Rate: {pred_success_arr.mean():.1%} ({pred_success_arr.sum()}/{len(pred_success_arr)})")

    # Pred vs GT 相似度
    m_pred_vs_gt = compute_metrics(pred_success_arr, gt_success_arr)
    print_metrics("Pred vs GT (GT 作为参考标签)", m_pred_vs_gt)

    # 如果有真实标签，额外输出 GT 和 Pred 各自与真实标签的对比
    if has_meta and len(true_label_list) > 0:
        true_arr = np.array(true_label_list, dtype=int)
        m_gt_vs_true = compute_metrics(gt_success_arr, true_arr)
        m_pred_vs_true = compute_metrics(pred_success_arr, true_arr)
        print_metrics("GT vs True Label", m_gt_vs_true)
        print_metrics("Pred vs True Label", m_pred_vs_true)

    # 保存结果
    out_path = os.path.join(args.latent_dir, "eval_latent_results.json")
    save_data = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "tail_ratio": args.tail_ratio,
        "n_batches": len(results),
        "gt_success_rate": float(gt_success_arr.mean()),
        "pred_success_rate": float(pred_success_arr.mean()),
        "pred_vs_gt": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in m_pred_vs_gt.items()},
        "details": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
