#!/usr/bin/env python3
"""Mission 2 ResNet18 학습 및 Validation Accuracy 기반 모델 선택."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Mission2ManifestDataset, steps_per_epoch
from feature_dataset import PrecomputedFeatureDataset
from model import build_model, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[permutation], y, y[permutation], lam


def accuracy_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    return float(np.mean((probabilities >= threshold).astype(np.int64) == labels))


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict:
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
            logits = model(features)
            loss = criterion(logits, labels)
        losses.append(float(loss.item()) * len(labels))
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(logits.softmax(dim=1)[:, 1].float().cpu().numpy())

    y_true = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    threshold = 0.5  # 대회 규정: Train/Validation/추론 모두 고정.
    default_accuracy = accuracy_at_threshold(y_true, probabilities, 0.5)
    predictions = (probabilities >= threshold).astype(np.int64)
    confusion = [[int(np.sum((y_true == truth) & (predictions == pred))) for pred in (0, 1)] for truth in (0, 1)]
    elapsed = time.perf_counter() - started
    return {
        "loss": sum(losses) / len(y_true),
        "accuracy_0.5": default_accuracy,
        "threshold": threshold,
        "confusion_matrix": confusion,
        "samples": int(len(y_true)),
        "elapsed_seconds": elapsed,
        "milliseconds_per_sample": elapsed / len(y_true) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 ResNet18 학습")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature-root", type=Path, help="precompute_features.py 출력 폴더")
    source.add_argument("--data-root", type=Path, help="원본 대학부 데이터 폴더")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--valid-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="기존 best_model.pt의 모델 가중치만 읽어 새 실험을 시작 (원본 실험 보존)",
    )
    parser.add_argument(
        "--model-name",
        choices=["resnet18", "resnet18_smallstem", "hybrid_resnet18_tdnn"],
        default="resnet18",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="output-dir의 last_checkpoint.pt가 있으면 마지막 완료 epoch부터 재개",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    if args.feature_root is not None:
        train_dataset = PrecomputedFeatureDataset(args.feature_root, "Training")
        valid_dataset = PrecomputedFeatureDataset(args.feature_root, "Validation")
        if train_dataset.feature_variant != valid_dataset.feature_variant:
            raise ValueError("Training/Validation 특징 종류가 서로 다릅니다.")
        feature_variant = train_dataset.feature_variant
        loader_shuffle = True
        input_source = "precomputed_logmel"
    else:
        if args.train_manifest is None or args.valid_manifest is None:
            parser.error("--data-root 사용 시 --train-manifest와 --valid-manifest가 필요합니다.")
        train_dataset = Mission2ManifestDataset(
            args.train_manifest, args.data_root, "Training", shuffle=True, seed=args.seed
        )
        valid_dataset = Mission2ManifestDataset(
            args.valid_manifest, args.data_root, "Validation", shuffle=False, seed=args.seed
        )
        loader_shuffle = False
        input_source = "raw_wav"
        feature_variant = "first_1p5"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=loader_shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0 and args.feature_root is not None,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    model = build_model(
        pretrained=not args.no_pretrained and args.init_checkpoint is None,
        dropout=args.dropout,
        spec_augment=True,
        model_name=args.model_name,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    print(f"device={device} train={len(train_dataset):,} valid={len(valid_dataset):,}")
    print(f"input_source={input_source}")
    print(f"feature_variant={feature_variant}")
    print(f"parameters={count_parameters(model):,} steps/epoch={steps_per_epoch(train_dataset, args.batch_size):,}")

    best_accuracy = -math.inf
    epochs_without_improvement = 0
    history: list[dict] = []
    start_epoch = 1
    checkpoint_path = args.output_dir / "best_model.pt"
    last_checkpoint_path = args.output_dir / "last_checkpoint.pt"

    if args.resume and last_checkpoint_path.is_file():
        resume_state = torch.load(last_checkpoint_path, map_location="cpu")
        if resume_state.get("selection_policy") != "accuracy_at_fixed_0.5":
            raise ValueError("이전 임계값 튜닝 실행은 재개할 수 없습니다. 새 output-dir에서 고정 0.5 학습을 시작하세요.")
        saved_model_name = str(resume_state.get("model_name", "resnet18"))
        if saved_model_name != args.model_name:
            raise ValueError(
                f"resume checkpoint model={saved_model_name}, requested model={args.model_name}"
            )
        if str(resume_state.get("feature_variant", "first_1p5")) != feature_variant:
            raise ValueError("resume checkpoint의 전처리 특징 종류가 현재 입력과 다릅니다.")
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict"):
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        completed_epoch = int(resume_state["epoch"])
        start_epoch = completed_epoch + 1
        best_accuracy = float(resume_state.get("best_accuracy", -math.inf))
        epochs_without_improvement = int(resume_state.get("epochs_without_improvement", 0))
        history = list(resume_state.get("history", []))
        print(f"resumed={last_checkpoint_path} completed_epoch={completed_epoch}")
    elif args.init_checkpoint is not None:
        init_state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if str(init_state.get("model_name", "resnet18")) != args.model_name:
            raise ValueError("초기 가중치와 학습 모델명이 다릅니다.")
        if str(init_state.get("feature_variant", "first_1p5")) != feature_variant:
            raise ValueError("초기 가중치와 학습 특징 종류가 다릅니다.")
        model.load_state_dict(init_state["model_state_dict"], strict=True)
        print(f"initialized_from={args.init_checkpoint} epoch={init_state.get('epoch')}")

    if start_epoch > args.epochs:
        print(f"이미 {args.epochs} epoch까지 완료되었습니다.")
        return
    if args.resume and history and epochs_without_improvement >= args.patience:
        print(f"이미 early stopping 조건({args.patience} epoch)을 충족했습니다.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        started = time.perf_counter()

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed, labels_a, labels_b, lam = mixup_batch(features, labels, args.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
                logits = model(mixed)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = len(labels)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            correct += int(((logits.softmax(dim=1)[:, 1] >= 0.5).long() == labels).sum().item())

        validation = evaluate(model, valid_loader, device, criterion)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy_without_mixup_correction": correct / seen,
            "train_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"valid_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        atomic_write_json(args.output_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False))

        if validation["accuracy_0.5"] > best_accuracy:
            best_accuracy = validation["accuracy_0.5"]
            epochs_without_improvement = 0
            atomic_torch_save(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": validation["threshold"],
                    "selection_policy": "accuracy_at_fixed_0.5",
                    "validation": validation,
                    "epoch": epoch,
                    "model_name": args.model_name,
                    "dropout": args.dropout,
                    "n_mels": 64,
                    "target_seconds": 1.5,
                    "sample_rate": 8000,
                    "input_source": input_source,
                    "feature_variant": feature_variant,
                    "parameters": count_parameters(model),
                    "training_args": {
                        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                    },
                },
            )
            print(f"saved: {checkpoint_path} (accuracy={best_accuracy:.6f})")
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            last_checkpoint_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_accuracy": best_accuracy,
                "selection_policy": "accuracy_at_fixed_0.5",
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "input_source": input_source,
                "model_name": args.model_name,
                "feature_variant": feature_variant,
            },
        )
        print(f"resume checkpoint: {last_checkpoint_path}")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping: {args.patience} epochs without improvement")
            break


if __name__ == "__main__":
    main()
