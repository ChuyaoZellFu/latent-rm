"""
集中管理所有配置项，方便修改和排查。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # 数据集根目录
    data_root: str = "/ssd/cyfu/wm_exp_pixel/data/robotwin_multiview_task_ft"

    # 使用的任务列表（去掉 adjust_bottle）
    tasks: List[str] = field(default_factory=lambda: [
        "grab_roller",
        "stack_bowls_two",
        "place_container_plate",
        "place_burger_fries",
    ])

    # 使用哪个视图的 latent (0, 1, 2)
    camera_id: int = 0

    # latent 维度: [num_patches, emb_dim]
    latent_num_patches: int = 256
    latent_emb_dim: int = 2048

    # 每个 episode 最多采样多少帧
    # 成功轨迹内部会做正负样本平衡（neg = sqrt(n) * pos）
    # 失败轨迹最多采这么多帧
    frames_per_episode: int = 50       # train
    val_frames_per_episode: int = 100  # val（多采一些，评估更准）

    # 成功轨迹中，最后多少比例的帧标记为正样本 (label=1)
    # 例如 0.2 表示最后 20% 的帧为正样本
    success_tail_ratio: float = 0.2

    # 训练/验证划分比例
    val_ratio: float = 0.15

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # LatentRewModel: MLP on vlm_token latent
    # 输入: [B, 256, 2048]，vlm_token latent
    # 输出: [B, 1]，范围 [0, 1] (Sigmoid)
    pass


@dataclass
class TrainConfig:
    # 训练超参
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    seed: int = 42

    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 2

    # 保存与日志
    output_dir: str = "/ssd/cyfu/yuhan/reward_model_latent/outputs"
    save_every: int = 5        # 每 N 个 epoch 保存一次
    log_every: int = 10        # 每 N 个 batch 打印一次

    # 设备
    device: str = "cuda:0"

    # 正负样本平衡：对正样本加权 (因为正样本少)
    pos_weight: float = 3.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
