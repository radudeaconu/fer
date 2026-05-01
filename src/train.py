"""Generic FER training loop driven by a YAML config.

Run: python -m src.train --config configs/dan_rafdb.yaml [--resume <ckpt>]

The training loop is intentionally model-agnostic: the model is constructed
from `src.models.build_model(...)` and is expected to return either logits
(Tensor of shape [B, C]) or a tuple `(logits, *aux)` where the first element
is logits. DAN returns (logits, x_features, heads); we unpack defensively.

Outputs:
    runs/<experiment_name>/best.pth        best UAR on val
    runs/<experiment_name>/last.pth        last epoch
    runs/<experiment_name>/log.jsonl       per-epoch metrics (one JSON per line)
    runs/<experiment_name>/config.yaml     resolved config (copied for reproducibility)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import (
    NUM_CLASSES,
    FER2013Dataset,
    RAFDBDataset,
    build_transforms,
    class_weighted_sampler,
)
from src.models import build_model

DATASETS = {"fer2013": FER2013Dataset, "rafdb": RAFDBDataset}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _logits_from_model_output(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    raise TypeError(f"Cannot extract logits from model output of type {type(out)}")


def make_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, list[int]]:
    ds_cfg = cfg["data"]
    Cls = DATASETS[ds_cfg["dataset"]]
    image_size = ds_cfg["image_size"]
    num_workers = ds_cfg["num_workers"]

    train_ds = Cls(ds_cfg["root"], split="train", transform=build_transforms(train=True, image_size=image_size))
    # FER-2013 has a separate val split; RAF-DB uses test as val (no official val).
    val_split = "val" if ds_cfg["dataset"] == "fer2013" else "test"
    val_ds = Cls(ds_cfg["root"], split=val_split, transform=build_transforms(train=False, image_size=image_size))

    sampler = None
    shuffle = True
    if ds_cfg.get("use_class_weighted_sampler", False):
        sampler = class_weighted_sampler(train_ds.labels)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.labels


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["train"]["optimizer"].lower()
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r}")


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)
    name = cfg["train"]["scheduler"].lower()
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler {name!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    loss_sum = 0.0
    n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = _logits_from_model_output(model(imgs))
        loss = F.cross_entropy(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    pred = np.array(all_pred)
    true = np.array(all_true)
    war = float((pred == true).mean())  # overall accuracy
    # UAR = mean per-class recall.
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = true == c
        if mask.sum() > 0:
            per_class_acc.append((pred[mask] == c).mean())
    uar = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    return {"loss": loss_sum / max(1, n), "war": war, "uar": uar}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    label_smoothing: float,
    grad_clip: float,
) -> dict:
    model.train()
    loss_sum = 0.0
    n = 0
    correct = 0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = _logits_from_model_output(model(imgs))
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return {"loss": loss_sum / max(1, n), "acc": correct / max(1, n)}


def main(cfg_path: Path, resume: Path | None = None) -> None:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path("runs") / cfg["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, run_dir / "config.yaml")
    log_path = run_dir / "log.jsonl"

    train_loader, val_loader, _ = make_dataloaders(cfg)
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"),
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device=device.type, enabled=cfg["train"].get("amp", False) and device.type == "cuda")

    start_epoch = 0
    best_uar = 0.0
    if resume is not None and resume.exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_uar = ckpt.get("best_uar", 0.0)
        print(f"Resumed from {resume} at epoch {start_epoch} (best UAR={best_uar:.4f})")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            label_smoothing=cfg["train"].get("label_smoothing", 0.0),
            grad_clip=cfg["train"].get("grad_clip", 0.0),
        )
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_war": val_metrics["war"],
            "val_uar": val_metrics["uar"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} WAR={val_metrics['war']:.4f} UAR={val_metrics['uar']:.4f} "
            f"({elapsed:.1f}s)"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_uar": best_uar,
            "config": cfg,
        }
        torch.save(ckpt, run_dir / "last.pth")
        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            torch.save(ckpt, run_dir / "best.pth")
            print(f"  -> new best UAR={best_uar:.4f}; saved best.pth")

    print(f"Done. Best UAR={best_uar:.4f}. Logs: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    main(args.config, args.resume)
