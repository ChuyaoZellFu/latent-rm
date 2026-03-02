"""
Reward Model 定义（Latent 版本）。

输入: [B, 256, 2048]，vlm_token latent
输出: [B, 1]，范围 [0, 1] (Sigmoid)

结构：先对 patch 维度做平均池化得到 [B, 2048]，再经过 MLP 输出 reward。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LatentRewModel(nn.Module):
    """
    基于 vlm_token latent 的 Reward Model。

    输入: [B, num_patches, emb_dim] = [B, 256, 2048]，float32
    输出: [B, 1]，范围 [0, 1] (Sigmoid)

    先对 patch 维度做平均池化 -> [B, 2048]，再经过 MLP。
    """

    def __init__(self, num_patches=256, emb_dim=2048, hidden_dims=(512, 128), checkpoint_path=None):
        super().__init__()
        self.num_patches = num_patches
        self.emb_dim = emb_dim

        layers = []
        in_dim = emb_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _encode(self, latent):
        """latent: [B, num_patches, emb_dim] -> [B, emb_dim]"""
        latent = latent.to(dtype=torch.float32)
        return latent.mean(dim=1)  # 平均池化 patch 维度

    def forward_train(self, latent):
        """训练时使用：返回连续的 [0,1] 值，用于计算 BCE loss。"""
        x = self._encode(latent)
        return self.mlp(x)  # [B, 1]

    @torch.no_grad()
    def predict_rew(self, latent):
        """推理时使用：返回 round 后的 0/1 值。"""
        x = self._encode(latent)
        x = self.mlp(x)
        return torch.round(x)

    def forward(self, latent=None):
        return self.predict_rew(latent)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")

    def save_checkpoint(self, path, epoch=None, optimizer=None, extra=None):
        """保存 checkpoint。"""
        ckpt = {"model_state_dict": self.state_dict()}
        if epoch is not None:
            ckpt["epoch"] = epoch
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"Checkpoint saved to {path}")
