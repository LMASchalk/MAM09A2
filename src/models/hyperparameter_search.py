import itertools
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from utils import npz_filename,OCTMNISTDataset,SimpleMLP,SimpleCNN,train_one_epoch,evaluate,find_availableName,EarlyStopping,run_tag

# Same paths as fit_deeplearning.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"

# Config block.
SEED = 42
SIZE = 28
EPOCHS = 40

# Sweep grids. Total runs = product of the three lengths.
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
WEIGHT_DECAYS = [1e-3, 1e-4, 1e-5]   # L2 lambda for Adam
BATCH_SIZES = [32, 64, 128]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modeltype",
        type=str,
        default="MLP",
        choices=["MLP", "CNN"],
        help="The type of deep learning model, Options: MLP (Default), CNN",
    )
    return parser.parse_args()


def set_seed(seed):
    # Re-applied before every run so combinations differ only by hyperparameters.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(modeltype, num_classes, size, device):
    if modeltype == "MLP":
        return SimpleMLP(num_classes=num_classes, image_size=size).to(device)
    if modeltype == "CNN":
        return SimpleCNN(num_classes=num_classes, image_size=size).to(device)
    raise ValueError(f"Unknown modeltype: {modeltype}")


def run_one(lr, weight_decay, batch_size,
            train_dataset, val_dataset, class_weights, device):
    # Identical starting point for every combination.
    set_seed(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(MODELTYPE, 4, SIZE, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    val_accs = []
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_accs.append(val_acc)

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            break

    best_epoch = early_stopping.best_epoch + 1
    best_val_acc = val_accs[early_stopping.best_epoch]
    return best_val_acc, best_epoch


def write_sweep_report(results):
    tag = run_tag(MODELTYPE.lower(), SIZE)
    file_name = find_availableName(REPORTS_DIR, f"{tag}_sweep.md")
    report_path = REPORTS_DIR / file_name

    best = results[0]
    lines = [
        f"# Hyperparameter sweep: {MODELTYPE} ({SIZE}px)",
        "",
        "Selection metric: best validation accuracy at the early-stopped "
        "checkpoint. The test set is left untouched here; run "
        "fit_deeplearning.py once with the winning configuration to report it.",
        "",
        "| lr | weight_decay (L2) | batch_size | best_val_acc | best_epoch |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['lr']:g} | {r['weight_decay']:g} | {r['batch_size']} "
            f"| {r['best_val_acc']:.4f} | {r['best_epoch']} |")
    lines += [
        "",
        f"Best configuration: lr={best['lr']:g}, "
        f"weight_decay={best['weight_decay']:g}, "
        f"batch_size={best['batch_size']} "
        f"(val_acc={best['best_val_acc']:.4f}).",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(f"[done] sweep table saved to {report_path}")


def main():
    global MODELTYPE
    MODELTYPE = parse_args().modeltype
    
    print("Starting sweep ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the splits once, only the loaders depend on batch size.
    data_path = DATA_ROOT / npz_filename(SIZE)
    train_dataset = OCTMNISTDataset(data_path, split="train")
    val_dataset = OCTMNISTDataset(data_path, split="val")
    print("Dataset loaded!")

    # Balanced class weights, computed once (independent of hyperparameters).
    num_classes = 4
    counts = np.bincount(train_dataset.labels, minlength=num_classes)
    class_weights = torch.tensor(
        counts.sum() / (num_classes * counts),
        dtype=torch.float32,
        device=device,
    )

    grid = list(itertools.product(LEARNING_RATES, WEIGHT_DECAYS, BATCH_SIZES))
    print(f"Sweeping {len(grid)} combinations")

    results = []
    for i, (lr, weight_decay, batch_size) in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] lr={lr} wd={weight_decay} bs={batch_size}")
        best_val_acc, best_epoch = run_one(
            lr, weight_decay, batch_size,
            train_dataset, val_dataset, class_weights, device)
        print(f"    best_val_acc={best_val_acc:.4f} (epoch {best_epoch})")
        results.append({
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        })

    # Rank on validation accuracy, selection never touches the test set.
    results.sort(key=lambda r: r["best_val_acc"], reverse=True)
    write_sweep_report(results)


if __name__ == "__main__":
    main()
