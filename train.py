"""
Reward Model 训练入口。

用法:
    python train.py

可通过修改 config.py 中的参数来调整训练配置。
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import build_datasets
from model import LatentRewModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp, fp, fn, tn = 0, 0, 0, 0

    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)

        preds = model.forward_train(latents).squeeze(-1)
        loss = nn.functional.binary_cross_entropy(preds, labels)

        total_loss += loss.item() * latents.size(0)
        pred_binary = (preds > 0.5).float()
        correct += (pred_binary == labels).sum().item()
        total += latents.size(0)

        tp += ((pred_binary == 1) & (labels == 1)).sum().item()
        fp += ((pred_binary == 1) & (labels == 0)).sum().item()
        fn += ((pred_binary == 0) & (labels == 1)).sum().item()
        tn += ((pred_binary == 0) & (labels == 0)).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    global_step = epoch
    writer.add_scalar("val/loss", avg_loss, global_step)
    writer.add_scalar("val/acc", acc, global_step)
    writer.add_scalar("val/precision", precision, global_step)
    writer.add_scalar("val/recall", recall, global_step)
    writer.add_scalar("val/f1", f1, global_step)

    print(f"  [Val]   Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f} "
          f"prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return avg_loss, acc, f1


def main():
    cfg = Config()
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(cfg.train.output_dir, "tb_logs"))

    # 数据
    print("=" * 60)
    print("Building datasets...")
    train_ds, val_ds = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # 模型 (不传 checkpoint_path，从头训练)
    print("=" * 60)
    print("Building model...")
    model = LatentRewModel(
        num_patches=cfg.data.latent_num_patches,
        emb_dim=cfg.data.latent_emb_dim,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss: 训练用带正样本加权的 BCE，评估用标准 BCE
    # 模型输出已经过 Sigmoid，所以用 BCELoss 而不是 BCEWithLogitsLoss

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # LR Scheduler
    if cfg.train.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.num_epochs - cfg.train.warmup_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs
    )
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[cfg.train.warmup_epochs],
    )

    # 训练循环
    print("=" * 60)
    print(f"Starting training for {cfg.train.num_epochs} epochs...")
    print(f"Output dir: {cfg.train.output_dir}")
    print("=" * 60)

    best_val_f1 = 0.0

    for epoch in range(1, cfg.train.num_epochs + 1):
        t0 = time.time()

        # 每个 epoch 重新随机抽帧（train 和 val 是 episode 级别划分，不会泄露）
        train_ds.resample(epoch=epoch)
        val_ds.resample(epoch=epoch)

        # 训练时手动加权: loss = BCE * weight, weight=pos_weight for label=1
        train_loss = train_one_epoch_weighted(
            model, train_loader, optimizer, device, epoch, cfg, writer, cfg.train.pos_weight
        )

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, epoch, writer)

        combined_scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch} done in {elapsed:.1f}s, lr={optimizer.param_groups[0]['lr']:.6f}")

        # 保存 best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(cfg.train.output_dir, "best.pth")
            model.save_checkpoint(save_path, epoch=epoch, optimizer=optimizer,
                                  extra={"val_f1": val_f1, "val_acc": val_acc})
            print(f"  ★ New best model! val_f1={val_f1:.4f}")

        # 定期保存
        if epoch % cfg.train.save_every == 0:
            save_path = os.path.join(cfg.train.output_dir, f"epoch_{epoch}.pth")
            model.save_checkpoint(save_path, epoch=epoch, optimizer=optimizer)

    # 保存最终模型
    save_path = os.path.join(cfg.train.output_dir, "final.pth")
    model.save_checkpoint(save_path, epoch=cfg.train.num_epochs, optimizer=optimizer)

    # 同时保存一份纯 state_dict 版本
    save_path_sd = os.path.join(cfg.train.output_dir, "latent_rm.pth")
    torch.save({"model_state_dict": model.state_dict()}, save_path_sd)
    print(f"Latent RM checkpoint saved to {save_path_sd}")

    writer.close()
    print("=" * 60)
    print(f"Training complete! Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoints in: {cfg.train.output_dir}")


def train_one_epoch_weighted(model, loader, optimizer, device, epoch, cfg, writer, pos_weight):
    """带正样本加权的训练。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    tp, fp, fn, tn = 0, 0, 0, 0

    for batch_idx, (latents, labels) in enumerate(loader):
        latents = latents.to(device)
        labels = labels.to(device)

        preds = model.forward_train(latents).squeeze(-1)  # [B]
        loss_per_sample = nn.functional.binary_cross_entropy(preds, labels, reduction="none")  # [B]

        # 正样本加权
        weights = torch.where(labels == 1, pos_weight, 1.0)
        loss = (loss_per_sample * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录未加权 loss，与 val loss 可比
        with torch.no_grad():
            unweighted_loss = loss_per_sample.mean()
        total_loss += unweighted_loss.item() * latents.size(0)
        pred_binary = (preds > 0.5).float()
        correct += (pred_binary == labels).sum().item()
        total += latents.size(0)

        tp += ((pred_binary == 1) & (labels == 1)).sum().item()
        fp += ((pred_binary == 1) & (labels == 0)).sum().item()
        fn += ((pred_binary == 0) & (labels == 1)).sum().item()
        tn += ((pred_binary == 0) & (labels == 0)).sum().item()

        if (batch_idx + 1) % cfg.train.log_every == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                  f"loss={loss.item():.4f} acc={correct/total:.4f}")

    avg_loss = total_loss / total
    acc = correct / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/acc", acc, epoch)
    writer.add_scalar("train/precision", precision, epoch)
    writer.add_scalar("train/recall", recall, epoch)
    writer.add_scalar("train/f1", f1, epoch)

    print(f"  [Train] Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f} "
          f"prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return avg_loss


if __name__ == "__main__":
    main()
