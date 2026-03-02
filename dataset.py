"""
Reward Model 训练数据集。

采样策略：
- 成功轨迹 (task 名含 'success')：
    - 最后 success_tail_ratio 比例的帧 → label=1 (正样本)
    - 其余帧 → label=0 (负样本)
- 失败轨迹 (task 名含 'fail')：
    - 所有帧 → label=0 (负样本)

每个 epoch 随机抽取不同的帧（而非固定帧），增加数据多样性。
每个 episode 内部做正负样本平衡采样：
    如果 neg 是 pos 的 n 倍，则采样时 neg 取 sqrt(n) * pos 个。
"""

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# 统计函数（独立）
# ============================================================

def collect_all_episodes(data_root: str, tasks: List[str], camera_id: int = 0) -> List[dict]:
    """从磁盘收集所有 episode 的元信息。"""
    all_episodes = []
    for task in tasks:
        ann_dir = os.path.join(data_root, f"annotation_{task}", "train")
        if not os.path.exists(ann_dir):
            print(f"WARNING: {ann_dir} not found, skipping")
            continue
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            ann_path = os.path.join(ann_dir, fname)
            with open(ann_path) as f:
                ann = json.load(f)
            is_success = "success" in ann["task"]
            num_frames = len(ann["state"])
            # 提取任务基础名 (去掉 -success_data-episodeXX)
            task_base = ann["task"].split("-")[0]
            # 获取 vlm_token latent 路径
            vlm_tokens = ann.get("vlm_tokens", [])
            if not vlm_tokens or camera_id >= len(vlm_tokens):
                print(f"WARNING: no vlm_tokens[{camera_id}] in {ann_path}, skipping")
                continue
            latent_path = os.path.join(data_root, vlm_tokens[camera_id]["vlm_token_path"])
            if not os.path.exists(latent_path):
                print(f"WARNING: latent not found: {latent_path}, skipping")
                continue
            all_episodes.append({
                "latent_path": latent_path,
                "is_success": is_success,
                "num_frames": num_frames,
                "task_name": ann["task"],
                "task_base": task_base,
                "ann_file": fname,
            })
    return all_episodes


def print_dataset_statistics(episodes: List[dict], label: str = "ALL"):
    """打印数据集统计信息：每类任务多少 episode，各自多少 frame。"""
    print(f"\n{'='*60}")
    print(f"  数据集统计 [{label}]")
    print(f"{'='*60}")

    by_task = defaultdict(lambda: {"success": [], "fail": []})
    for ep in episodes:
        key = "success" if ep["is_success"] else "fail"
        by_task[ep["task_base"]][key].append(ep["num_frames"])

    total_ep = 0
    total_frames = 0
    for task_name in sorted(by_task.keys()):
        info = by_task[task_name]
        s_eps = info["success"]
        f_eps = info["fail"]
        s_frames = sum(s_eps)
        f_frames = sum(f_eps)
        print(f"  {task_name}:")
        print(f"    success: {len(s_eps):3d} episodes, {s_frames:6d} frames "
              f"(avg {s_frames//max(len(s_eps),1):4d} frames/ep)")
        print(f"    fail:    {len(f_eps):3d} episodes, {f_frames:6d} frames "
              f"(avg {f_frames//max(len(f_eps),1):4d} frames/ep)")
        total_ep += len(s_eps) + len(f_eps)
        total_frames += s_frames + f_frames

    print(f"  {'─'*50}")
    print(f"  总计: {total_ep} episodes, {total_frames} frames")
    print(f"{'='*60}\n")


# ============================================================
# 采样函数（独立）
# ============================================================

def sample_frames_for_episode(
    num_frames: int,
    is_success: bool,
    success_tail_ratio: float,
    max_frames: int = 50,
    rng: random.Random = None,
) -> List[Tuple[int, int]]:
    """
    为单个 episode 采样帧，返回 [(frame_idx, label), ...]。

    正负样本平衡策略：
    - 成功轨迹：pos 区间 = 最后 tail_ratio 帧，neg 区间 = 前面的帧
      如果 neg 原始数量是 pos 的 n 倍，则采样 neg = sqrt(n) * pos 个
    - 失败轨迹：全部为 neg，最多采 max_frames 个
    """
    if rng is None:
        rng = random.Random()

    if not is_success:
        # 失败轨迹：全部 neg，随机采 max_frames 个
        n_sample = min(num_frames, max_frames)
        indices = sorted(rng.sample(range(num_frames), n_sample))
        return [(fi, 0) for fi in indices]

    # 成功轨迹
    tail_start = int(num_frames * (1.0 - success_tail_ratio))
    pos_indices = list(range(tail_start, num_frames))  # 正样本候选
    neg_indices = list(range(0, tail_start))            # 负样本候选

    n_pos_total = len(pos_indices)
    n_neg_total = len(neg_indices)

    # 正样本：最多采 max_frames 个
    n_pos_sample = min(n_pos_total, max_frames)
    sampled_pos = sorted(rng.sample(pos_indices, n_pos_sample))

    # 负样本平衡：neg 采 sqrt(n) * pos 个
    if n_neg_total > 0 and n_pos_sample > 0:
        ratio = n_neg_total / max(n_pos_total, 1)
        n_neg_sample = int(math.sqrt(ratio) * n_pos_sample)
        n_neg_sample = max(1, min(n_neg_sample, n_neg_total, max_frames))
    else:
        n_neg_sample = 0

    sampled_neg = sorted(rng.sample(neg_indices, n_neg_sample)) if n_neg_sample > 0 else []

    result = [(fi, 1) for fi in sampled_pos] + [(fi, 0) for fi in sampled_neg]
    return result


def print_sampling_summary(episodes: List[dict], all_samples: List, split: str, epoch: int = None):
    """打印本次采样的摘要信息。"""
    epoch_str = f" Epoch {epoch}" if epoch is not None else ""
    print(f"\n  [{split}{epoch_str}] 采样摘要:")

    n_pos = sum(1 for _, _, l in all_samples if l == 1)
    n_neg = len(all_samples) - n_pos
    print(f"    总样本: {len(all_samples)} (pos={n_pos}, neg={n_neg}, "
          f"pos_ratio={n_pos/max(len(all_samples),1):.2%})")

    # 按任务统计
    by_task = defaultdict(lambda: {"pos": 0, "neg": 0, "episodes": set()})
    for ep_idx, frame_idx, label in all_samples:
        task_base = episodes[ep_idx]["task_base"]
        by_task[task_base]["pos" if label == 1 else "neg"] += 1
        by_task[task_base]["episodes"].add(ep_idx)

    for task_name in sorted(by_task.keys()):
        info = by_task[task_name]
        print(f"    {task_name}: {len(info['episodes'])} eps, "
              f"pos={info['pos']}, neg={info['neg']}")


# ============================================================
# Dataset 类
# ============================================================

class RewardModelDataset(Dataset):
    """
    每个样本是一个 (latent, label) 对。
    latent: [num_patches, emb_dim]，float32（从 float16 转换）
    label: 0 或 1

    每个 epoch 调用 resample() 重新随机抽帧。
    latent 文件按需加载（不预缓存），避免 OOM。
    """

    def __init__(
        self,
        episodes: List[dict],
        image_size: int = 256,  # 保留参数，兼容 build_datasets 调用，不使用
        max_frames_per_episode: int = 50,
        success_tail_ratio: float = 0.2,
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.episodes = episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.success_tail_ratio = success_tail_ratio
        self.split = split
        self.seed = seed
        self._epoch = 0

        # samples: [(ep_idx, frame_idx, label), ...]
        self.samples = []
        # episode 级别 latent 缓存，避免每帧都重复 load 整个 pth 文件
        self._latent_cache: dict = {}
        self.resample(epoch=0)

    def resample(self, epoch: int = 0):
        """每个 epoch 调用，重新随机抽帧。同时清空 latent 缓存释放内存。"""
        self._epoch = epoch
        rng = random.Random(self.seed + epoch)
        self.samples = []
        self._latent_cache = {}  # 清空旧缓存

        for ep_idx, ep in enumerate(self.episodes):
            frame_labels = sample_frames_for_episode(
                num_frames=ep["num_frames"],
                is_success=ep["is_success"],
                success_tail_ratio=self.success_tail_ratio,
                max_frames=self.max_frames_per_episode,
                rng=rng,
            )
            for fi, label in frame_labels:
                self.samples.append((ep_idx, fi, label))

        print_sampling_summary(self.episodes, self.samples, self.split, epoch)

    def __len__(self):
        return len(self.samples)

    def _load_episode_latent(self, ep_idx: int) -> torch.Tensor:
        """加载并缓存 episode 的全帧 latent，[T, 256, 2048] float16。"""
        if ep_idx not in self._latent_cache:
            ep = self.episodes[ep_idx]
            self._latent_cache[ep_idx] = torch.load(
                ep["latent_path"], map_location="cpu", weights_only=False
            )
        return self._latent_cache[ep_idx]

    def __getitem__(self, idx):
        ep_idx, frame_idx, label = self.samples[idx]

        try:
            latent_all = self._load_episode_latent(ep_idx)  # [T, 256, 2048], float16
            frame_idx = min(frame_idx, latent_all.shape[0] - 1)
            latent = latent_all[frame_idx].float()  # [256, 2048], float32
        except Exception as e:
            ep = self.episodes[ep_idx]
            print(f"ERROR reading {ep['latent_path']} frame {frame_idx}: {e}")
            latent = torch.zeros(256, 2048, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)
        return latent, label


# ============================================================
# 构建 train/val 数据集
# ============================================================

def build_datasets(cfg) -> Tuple[RewardModelDataset, RewardModelDataset]:
    """
    按 episode 级别划分 train/val，避免数据泄露。
    训练前打印完整统计信息。
    """
    # 1. 收集所有 episode
    all_episodes = collect_all_episodes(
        cfg.data.data_root, cfg.data.tasks, cfg.data.camera_id
    )

    # 2. 打印全量统计
    print_dataset_statistics(all_episodes, label="全部数据")

    # 3. 按 episode 划分 train/val
    n_total = len(all_episodes)
    n_val = max(1, int(n_total * cfg.data.val_ratio))
    n_train = n_total - n_val

    indices = list(range(n_total))
    rng = random.Random(cfg.train.seed)
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_episodes = [all_episodes[i] for i in sorted(train_indices)]
    val_episodes = [all_episodes[i] for i in sorted(val_indices)]

    print(f"划分: train={n_train} episodes, val={n_val} episodes")

    # 4. 打印 train/val 各自统计
    print_dataset_statistics(train_episodes, label="Train")
    print_dataset_statistics(val_episodes, label="Val")

    # 5. 构建 Dataset
    train_ds = RewardModelDataset(
        episodes=train_episodes,
        max_frames_per_episode=cfg.data.frames_per_episode,
        success_tail_ratio=cfg.data.success_tail_ratio,
        split="train",
        seed=cfg.train.seed,
    )

    val_ds = RewardModelDataset(
        episodes=val_episodes,
        max_frames_per_episode=cfg.data.val_frames_per_episode,
        success_tail_ratio=cfg.data.success_tail_ratio,
        split="val",
        seed=cfg.train.seed + 9999,  # val 用不同的种子
    )

    return train_ds, val_ds
