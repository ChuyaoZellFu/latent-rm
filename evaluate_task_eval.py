"""
对 task_eval 目录中多任务、多 episode 的 pred/gt latent 数据进行奖励模型评估。

用法:
    python reward_model_latent/evaluate_task_eval.py \
        --checkpoint /ssd/cyfu/yuhan/reward_model_latent/outputs/best.pth \
        --eval_dir /ssd/cyfu/yuhan/task_eval_4gpu_step16_final

支持的目录结构:
    eval_dir/<task_name>/<task_name>/episode_XXXX/gt_latents.pth
    eval_dir/<task_name>/<task_name>/episode_XXXX/pred_latents.pth
    eval_dir/<task_name>/<task_name>/episode_XXXX/episode_info.json

功能:
    1. 对每个 episode 的 pred_latents 和 gt_latents 分别用奖励模型打分
    2. 以 episode_info.json 中的 label (success/failure) 为真实标签，
       计算 GT 打分和 Pred 打分的分类准确性
    3. 计算奖励模型对 pred 的打分与 GT 质量指标 (rmse_1, cos_1, rmse_at_20)
       的 Pearson 相关系数
    4. 按任务和全局两个粒度汇总结果
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LatentRewModel
from evaluate_videos import evaluate_latents, compute_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_all_episodes(eval_dir):
    """
    扫描 eval_dir，返回 (task_name, episode_dir) 列表。

    期望结构:
        eval_dir/<task_name>/<task_name>/episode_XXXX/
    也兼容:
        eval_dir/<task_name>/episode_XXXX/
    """
    records = []
    task_dirs = sorted([
        d for d in glob.glob(os.path.join(eval_dir, "*"))
        if os.path.isdir(d)
    ])
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)

        # 尝试嵌套结构: task_dir/<task_name>/episode_XXXX
        nested = os.path.join(task_dir, task_name)
        if os.path.isdir(nested):
            eps = sorted(glob.glob(os.path.join(nested, "episode_*")))
            if eps:
                for ep in eps:
                    records.append((task_name, ep))
                continue

        # 尝试直接结构: task_dir/episode_XXXX
        eps = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
        for ep in eps:
            records.append((task_name, ep))

    return records


def load_episode_info(episode_dir):
    """从 episode_info.json 加载真实标签和质量指标，若不存在返回 None。"""
    info_path = os.path.join(episode_dir, "episode_info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Pearson correlation helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_pearsonr(x, y):
    """计算 Pearson 相关系数，若样本不足或标准差为 0 返回 NaN。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def print_pearson_table(title, scores, metrics_dict):
    """
    打印 scores 与 metrics_dict 中各指标的 Pearson 相关系数表格。

    Args:
        title: 表格标题
        scores: list of float (reward model scores)
        metrics_dict: {metric_name: list of float}
    """
    print(f"\n  [{title}] Pearson Correlation  (n={len(scores)})")
    print(f"    {'Metric':<20} {'r':>8} {'p-value':>10}  {'Interpretation'}")
    print("    " + "-" * 65)
    for metric, vals in metrics_dict.items():
        r, p = safe_pearsonr(scores, vals)
        if np.isnan(r):
            print(f"    {metric:<20} {'N/A':>8} {'N/A':>10}")
        else:
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            direction = "↑ positive" if r > 0.1 else ("↓ negative" if r < -0.1 else "≈ none")
            print(f"    {metric:<20} {r:>8.4f} {p:>10.4f}  {direction} {stars}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-task evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_task(task_name, episode_dirs, model, device, args):
    """
    对单个任务的所有 episode 进行评估，返回结果列表。

    latent 形状说明（已在 load_latent_file 中统一归一化）：
        task_eval 目录:  GT=[T,1,256,2048], Pred=[T,256,2048]  -> 均 reshape 为 [T,256,2048]
        latent_eval 目录: GT=Pred=[1,T,1,256,2048]            -> reshape 为 [T,256,2048]
    reshape(-1, 256, 2048) 对所有格式都等价正确。

    返回:
        list of dict, 每个 dict 包含:
            task, episode, true_label (0/1),
            gt_is_success, gt_tail_mean,
            pred_is_success, pred_tail_mean,
            rmse_1, cos_1, rmse_at_20  (来自 episode_info.json)
    """
    results = []
    print(f"\n{'=' * 90}")
    print(f"  Task: {task_name}  ({len(episode_dirs)} episodes)")
    print(f"{'=' * 90}")
    print(f"  {'Episode':<14} {'TrueLabel':<12} {'GT_Succ':>8} {'GT_Tail':>8} "
          f"{'Pred_Succ':>10} {'Pred_Tail':>10}  {'RMSE_1':>8} {'Cos_1':>8}")
    print(f"  {'-' * 88}")

    first_ep = True
    for ep_dir in episode_dirs:
        ep_name = os.path.basename(ep_dir)

        gt_path = os.path.join(ep_dir, "gt_latents.pth")
        pred_path = os.path.join(ep_dir, "pred_latents.pth")
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"  Warning: missing latent files in {ep_name}, skipping.")
            continue

        info = load_episode_info(ep_dir)
        true_label = None
        rmse_1 = cos_1 = rmse_at_20 = float("nan")
        if info is not None:
            true_label = 1 if info.get("label", "failure") == "success" else 0
            rmse_1 = float(info.get("rmse_1", float("nan")))
            cos_1 = float(info.get("cos_1", float("nan")))
            rmse_at_20 = float(info.get("rmse_at_20", float("nan")))

        gt_latent_raw = torch.load(gt_path, map_location="cpu", weights_only=False)
        pred_latent_raw = torch.load(pred_path, map_location="cpu", weights_only=False)

        gt_latent = gt_latent_raw.float().reshape(-1, 256, 2048)
        pred_latent = pred_latent_raw.float().reshape(-1, 256, 2048)

        if first_ep:
            print(f"  [shape check] gt_raw={tuple(gt_latent_raw.shape)} -> {tuple(gt_latent.shape)}  "
                  f"pred_raw={tuple(pred_latent_raw.shape)} -> {tuple(pred_latent.shape)}")
            first_ep = False

        do_debug = (args.debug == "all" or args.debug == ep_name)

        gt_is_success, gt_tail_mean, _ = evaluate_latents(
            model, gt_latent, device,
            threshold=args.threshold, tail_ratio=args.tail_ratio,
            judgment=args.judgment, debug=do_debug, debug_label=f"{ep_name} GT"
        )
        pred_is_success, pred_tail_mean, _ = evaluate_latents(
            model, pred_latent, device,
            threshold=args.threshold, tail_ratio=args.tail_ratio,
            judgment=args.judgment, debug=do_debug, debug_label=f"{ep_name} Pred"
        )

        true_str = ("success" if true_label == 1 else "failure") if true_label is not None else "?"
        print(f"  {ep_name:<14} {true_str:<12} "
              f"{'YES' if gt_is_success else 'NO':>8} {gt_tail_mean:>8.4f} "
              f"{'YES' if pred_is_success else 'NO':>10} {pred_tail_mean:>10.4f}  "
              f"{rmse_1:>8.4f} {cos_1:>8.4f}")

        results.append({
            "task": task_name,
            "episode": ep_name,
            "true_label": true_label,
            "gt_is_success": bool(gt_is_success),
            "gt_tail_mean": float(gt_tail_mean),
            "pred_is_success": bool(pred_is_success),
            "pred_tail_mean": float(pred_tail_mean),
            "rmse_1": rmse_1,
            "cos_1": cos_1,
            "rmse_at_20": rmse_at_20,
        })

    return results


def summarize_task(task_name, results):
    """打印单个任务的汇总统计，并返回用于全局汇总的向量。"""
    if not results:
        return {}

    gt_arr = np.array([r["gt_is_success"] for r in results], dtype=int)
    pred_arr = np.array([r["pred_is_success"] for r in results], dtype=int)
    pred_scores = [r["pred_tail_mean"] for r in results]
    gt_scores = [r["gt_tail_mean"] for r in results]

    print(f"\n  --- Task Summary: {task_name} ---")
    print(f"  GT   Success Rate: {gt_arr.mean():.1%} ({gt_arr.sum()}/{len(gt_arr)})")
    print(f"  Pred Success Rate: {pred_arr.mean():.1%} ({pred_arr.sum()}/{len(pred_arr)})")

    # Pred vs GT agreement
    m = compute_metrics(pred_arr, gt_arr)
    print_metrics(f"{task_name}: Pred vs GT (GT 作为参考标签)", m)

    # vs true label (if available)
    has_labels = all(r["true_label"] is not None for r in results)
    if has_labels:
        true_arr = np.array([r["true_label"] for r in results], dtype=int)
        m_gt_true = compute_metrics(gt_arr, true_arr)
        m_pred_true = compute_metrics(pred_arr, true_arr)
        print_metrics(f"{task_name}: GT vs True Label", m_gt_true)
        print_metrics(f"{task_name}: Pred vs True Label", m_pred_true)

    # Pearson correlation: pred_tail_mean vs GT quality metrics
    metric_vals = {
        "rmse_1":       [r["rmse_1"] for r in results],
        "cos_1":        [r["cos_1"] for r in results],
        "rmse_at_20":   [r["rmse_at_20"] for r in results],
        "gt_tail_mean": gt_scores,
    }
    print_pearson_table(
        f"{task_name}: pred_tail_mean vs GT metrics",
        pred_scores, metric_vals
    )

    return {
        "n": len(results),
        "gt_success_rate": float(gt_arr.mean()),
        "pred_success_rate": float(pred_arr.mean()),
        "pred_vs_gt": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in m.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pred/gt latents across multiple tasks using latent reward model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Root directory of task_eval results (contains task subdirs)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for success classification (default: 0.5)")
    parser.add_argument("--tail_ratio", type=float, default=0.25,
                        help="Trailing frame ratio used when judgment=tail (default: 0.25)")
    parser.add_argument("--judgment", type=str, default="tail", choices=["max", "tail"],
                        help="Judgment method: 'tail' (default, tail_mean of last tail_ratio frames) "
                             "or 'max' (max_prob over all frames). 'tail' is correct for full-episode "
                             "data where success episodes terminate early (short T) and failures run "
                             "to end (long T).")
    parser.add_argument("--debug", type=str, default=None,
                        help="Print per-frame probs for episode name or 'all'")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated list of tasks to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (default: eval_dir/reward_eval_results.json)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    model = LatentRewModel(checkpoint_path=args.checkpoint)
    model = model.to(device)
    model.eval()
    score_desc = f"tail_mean (last {args.tail_ratio:.0%} frames)" if args.judgment == "tail" else "max_prob"
    print(f"Threshold: {args.threshold}  |  Judgment: {args.judgment}  |  Score: {score_desc}\n")

    # discover episodes
    all_episodes = find_all_episodes(args.eval_dir)
    if not all_episodes:
        print(f"No episodes found in {args.eval_dir}. Check directory structure.")
        return

    # filter tasks if requested
    task_filter = set(args.tasks.split(",")) if args.tasks else None
    if task_filter:
        all_episodes = [(t, d) for t, d in all_episodes if t in task_filter]

    # group by task
    tasks_dict = {}
    for task_name, ep_dir in all_episodes:
        tasks_dict.setdefault(task_name, []).append(ep_dir)

    print(f"Found {len(all_episodes)} episodes across {len(tasks_dict)} tasks: "
          f"{list(tasks_dict.keys())}")

    # ── per-task evaluation ──────────────────────────────────────────────────
    all_results = []
    task_summaries = {}
    for task_name in sorted(tasks_dict.keys()):
        ep_dirs = tasks_dict[task_name]
        results = evaluate_task(task_name, ep_dirs, model, device, args)
        all_results.extend(results)
        task_summaries[task_name] = summarize_task(task_name, results)

    # ── global summary ───────────────────────────────────────────────────────
    print(f"\n{'#' * 90}")
    print("  GLOBAL SUMMARY  (all tasks combined)")
    print(f"{'#' * 90}")

    gt_arr_all = np.array([r["gt_is_success"] for r in all_results], dtype=int)
    pred_arr_all = np.array([r["pred_is_success"] for r in all_results], dtype=int)
    pred_scores_all = [r["pred_tail_mean"] for r in all_results]

    print(f"  Total episodes: {len(all_results)}")
    print(f"  GT   Success Rate: {gt_arr_all.mean():.1%} ({gt_arr_all.sum()}/{len(gt_arr_all)})")
    print(f"  Pred Success Rate: {pred_arr_all.mean():.1%} ({pred_arr_all.sum()}/{len(pred_arr_all)})")

    m_global = compute_metrics(pred_arr_all, gt_arr_all)
    print_metrics("Global: Pred vs GT", m_global)

    has_labels = all(r["true_label"] is not None for r in all_results)
    if has_labels:
        true_arr_all = np.array([r["true_label"] for r in all_results], dtype=int)
        m_gt_true = compute_metrics(gt_arr_all, true_arr_all)
        m_pred_true = compute_metrics(pred_arr_all, true_arr_all)
        print_metrics("Global: GT vs True Label", m_gt_true)
        print_metrics("Global: Pred vs True Label", m_pred_true)

    global_metrics = {
        "rmse_1":       [r["rmse_1"] for r in all_results],
        "cos_1":        [r["cos_1"] for r in all_results],
        "rmse_at_20":   [r["rmse_at_20"] for r in all_results],
        "gt_tail_mean": [r["gt_tail_mean"] for r in all_results],
    }
    print_pearson_table(
        "Global: pred_tail_mean vs GT metrics",
        pred_scores_all, global_metrics
    )

    # ── save results ─────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(args.eval_dir, "reward_eval_results.json")

    def _to_serializable(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, bool):
            return v
        return v

    save_data = {
        "checkpoint": args.checkpoint,
        "eval_dir": args.eval_dir,
        "threshold": args.threshold,
        "n_total": len(all_results),
        "global": {
            "gt_success_rate": float(gt_arr_all.mean()),
            "pred_success_rate": float(pred_arr_all.mean()),
            "pred_vs_gt": {k: _to_serializable(v) for k, v in m_global.items()},
            "pearson": {
                metric: {"r": float(safe_pearsonr(pred_scores_all, vals)[0]),
                         "p": float(safe_pearsonr(pred_scores_all, vals)[1])}
                for metric, vals in global_metrics.items()
            },
        },
        "per_task": task_summaries,
        "details": [
            {k: _to_serializable(v) for k, v in r.items()}
            for r in all_results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
