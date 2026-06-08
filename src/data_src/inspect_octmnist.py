"""
Download the OCTMNIST dataset (default 64x64 resolution) and run a first-pass inspection of the official train / validation / test splits 
for the MAM09A2 project.

The script is deliberately read-only with respect to the data (no resplitting). The official MedMNIST splits are treated as fixed, for two
reasons documented in the project notes:
  - benchmark comparability with published OCTMNIST results, and
  - patient-level leakage: MedMNIST does not expose patient IDs, so a random image-level re-split would mix a patient's scans across train/test 
    with no way to detect or prevent it.

It addresses the following inspection points:
  1. Array shapes, dtype and pixel value range (per split).
  2. Per-split class distribution (highlighting the train imbalance vs the balanced 250/class test set).
  3. Split sizes and integrity checks against the official sample counts, plus the class-index -> label-name mapping.
  4. A visual sample of a few images per class.

Outputs (written to --out-dir):
  - inspection_report.md : human-readable summary of all findings
  - class_distribution.png : grouped bar chart of class proportions per split
  - sample_grid.png : example images, one row per class

The same text report is also printed to stdout, so it lands in the SLURM log.

Requires: medmnist>=3.0.0, numpy, matplotlib

Typical use
-----------
  # On a Snellius login node: just fetch the data, then exit.
  python inspect_octmnist.py --data-root $HOME/data/medmnist [not necessary if you want the defaults] --download-only

  # On a compute node (no internet): run the inspection on the cached data.
  python inspect_octmnist.py --data-root $HOME/data/medmnist --out-dir $HOME/9A2-octmnist/octmnist_inspection 
  [the last 2 are not necesary if you want the defaults]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np

from medmnist import INFO, OCTMNIST

DATAFLAG = "octmnist"
SPLITS = ("train", "val", "test")

# Resolve the repo root from this file's location:
#   <repo>/src/data_src/inspect_octmnist.py  ->  parents[2] == <repo>
# This keeps the defaults correct wherever the repo is cloned, while still allowing --data-root / --out-dir to override them.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"  # gitignored; holds octmnist_64.npz
DEFAULT_OUT_DIR = REPO_ROOT / "reports"  # report .md here, figures in ./figures


def parse_args():
    p = argparse.ArgumentParser(
        description="Download and inspect OCTMNIST (default 64x64)."
    )
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Directory where the MedMNIST .npz is stored / will be downloaded "
             "(default: <repo>/data/raw).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory for the report; figures are written to <out-dir>/figures "
             "(default: <repo>/reports).",
    )
    p.add_argument(
        "--size", type=int, default=64, choices=[28, 64, 128, 224],
        help="Image resolution to use (default: 64).",
    )
    p.add_argument(
        "--sample-split", default="train", choices=SPLITS,
        help="Split to draw the example image grid from (default: train).",
    )
    p.add_argument(
        "--n-per-class", type=int, default=6,
        help="Number of example images per class in the grid (default: 6).",
    )
    p.add_argument(
        "--download", action="store_true",
        help="Download the data if missing before inspecting "
             "(needs internet; run on a login node).",
    )
    p.add_argument(
        "--download-only", action="store_true",
        help="Only download the data, then exit (run on a login node).",
    )
    return p.parse_args()


def ensure_downloaded(data_root, size):
    """Download the OCTMNIST .npz file (all three splits live in one file).

    We instantiate the *test* split only: that triggers the download of the complete npz while decompressing just ~1000 images into memory. 
    Run this on a node with internet access (on Snellius: a login node).
    """
    data_root.mkdir(parents=True, exist_ok=True)
    OCTMNIST(split="test", download=True, size=size, root=str(data_root))


def load_split(split, data_root, size):
    """Return (images, raw_labels) for one split, without downloading.

    images : np.uint8 array, shape (N, H, W) for grayscale OCT
    raw_labels : integer array, shape (N, 1) as stored by MedMNIST
    """
    ds = OCTMNIST(split=split, download=False, size=size, root=str(data_root))
    return ds.imgs, np.asarray(ds.labels)


def class_names():
    """Ordered list of class names, indexed by integer label."""
    label_map = INFO[DATAFLAG]["label"]  # e.g. {"0": "choroidal neovascularization", ...}
    return [label_map[str(i)] for i in range(len(label_map))]


def plot_class_distribution(counts, names, out_path):
    """Grouped bar chart of per-split class *proportions*.

    Proportions (not raw counts) are plotted so the heavily-sampled train split does not visually dwarf the 1000-image test split; 
    this makes the shape of the imbalance directly comparable across splits.
    """
    n_classes = len(names)
    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, split in enumerate(SPLITS):
        c = counts[split]
        proportion = c / c.sum()
        ax.bar(x + (i - 1) * width, proportion, width, label=split)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{names[i]}" for i in range(n_classes)], fontsize=8)
    ax.set_ylabel("proportion within split")
    ax.set_title("OCTMNIST class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(sample_imgs, names, split, out_path, n_per_class):
    """One row per class, n_per_class example images per row."""
    n_classes = len(sample_imgs)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 1.6, n_classes * 1.6),
    )
    axes = np.atleast_2d(axes)

    for r in range(n_classes):
        row_imgs = sample_imgs[r]
        for col in range(n_per_class):
            ax = axes[r, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(row_imgs):
                ax.imshow(row_imgs[col], cmap="gray", vmin=0, vmax=255)
            if col == 0:
                ax.set_ylabel(f"{r}: {names[r]}", fontsize=8,
                              rotation=0, ha="right", va="center")

    fig.suptitle(f"OCTMNIST examples ({split} split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.download or args.download_only:
        print(f"[download] fetching OCTMNIST (size={args.size}) "
              f"into {args.data_root} ...")
        ensure_downloaded(args.data_root, args.size)
        print("[download] done.")
        if args.download_only:
            return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names()
    n_classes = len(names)
    expected = INFO[DATAFLAG]["n_samples"]  # {'train': 97477, 'val': 10832, 'test': 1000}

    report = []
    report.append(f"# OCTMNIST inspection (size = {args.size})\n")
    report.append(f"- Task: {INFO[DATAFLAG]['task']}")
    report.append(f"- Channels: {INFO[DATAFLAG]['n_channels']}")
    report.append("- Class index -> name:")
    for i, nm in enumerate(names):
        report.append(f"    - {i}: {nm}")
    report.append("")

    counts = {}  # split -> per-class count array
    sample_imgs = None  # collected from args.sample_split for the grid

    for split in SPLITS:
        imgs, labels_raw = load_split(split, args.data_root, args.size)
        labels = labels_raw.reshape(-1)  # (N, 1) -> (N,)

        n = len(labels)
        exp = expected[split]
        integrity = "OK" if n == exp else f"MISMATCH (expected {exp})"
        c = np.bincount(labels, minlength=n_classes)
        counts[split] = c

        report.append(f"## Split: {split}")
        report.append(f"- Number of images: {n}  [{integrity}]")
        report.append(f"- Image array shape: {tuple(imgs.shape)}")
        report.append(f"- Image dtype: {imgs.dtype}")
        report.append(f"- Pixel value range: [{int(imgs.min())}, {int(imgs.max())}]")
        report.append(f"- Label array shape: {tuple(labels_raw.shape)} "
                      f"(dtype {labels_raw.dtype})")
        report.append("- Class counts: "
                      + ", ".join(f"{names[k]}={int(c[k])}" for k in range(n_classes)))
        pct = c / c.sum() * 100
        report.append("- Class proportions (%): "
                      + ", ".join(f"{names[k]}={pct[k]:.1f}" for k in range(n_classes)))
        report.append("")

        if split == args.sample_split:
            sample_imgs = []
            for k in range(n_classes):
                idx = np.where(labels == k)[0][: args.n_per_class]
                sample_imgs.append(imgs[idx].copy())

        # Free the (potentially ~5 GB) train array before loading the next split.
        del imgs, labels_raw, labels

    # Interpretive notes derived from the counts.
    train_c = counts["train"]
    test_c = counts["test"]
    report.append("## Notes")
    report.append(f"- Train imbalance ratio (largest / smallest class): "
                  f"{train_c.max() / train_c.min():.1f}x")
    report.append(f"- Test set across classes: "
                  f"{'balanced' if len(set(test_c.tolist())) == 1 else 'imbalanced'} "
                  f"(counts: {test_c.tolist()})")
    report.append("- Implication: report per-class / macro metrics (balanced "
                  "accuracy, macro-AUC), not plain accuracy alone, because the "
                  "model trains under imbalance but is tested on a balanced set.")
    report.append("")

    report_text = "\n".join(report)
    (args.out_dir / "inspection_report.md").write_text(report_text)
    print(report_text)

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(counts, names, figures_dir / "class_distribution.png")
    if sample_imgs is not None:
        plot_sample_grid(sample_imgs, names, args.sample_split,
                         figures_dir / "sample_grid.png", args.n_per_class)

    print(f"[done] report and figures written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
