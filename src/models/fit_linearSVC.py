"""
LinearSVC baseline for the OCTMNIST classification project.

Flattened 28x28 grayscale images -> MInMaxScaler -> LinearSVC.

C is tuned on the official validation split (not k-fold cross-validation).
Test set is touched exactly once.

Metrics: balanced accuracy, macro one-vs-rest AUC, per-class recall, confusion matrix.
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

matplotlib.use("Agg")  # headless backend for the cluster (no display)

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--size",
        type=int,
        default=28,          # fallback if you don't pass --size
        help="Image size to use for OCTMNIST. Choices are: 244,128,64,28 (Default)",
    )
    
    parser.add_argument(
        "--modelname",
        type=str,
        default="linear_svc_octmnist",          # fallback if you don't pass --size
        help="The name of the saved model. Default: linear_svc_octmnist",
    )
        
    return parser.parse_args()

# Configuration
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
MAX_ITER = 5000
SEED = 42
# Tie-breaker for choosing C, both metrics are always computed and reported.
SELECTION_METRIC = "balanced_accuracy"
REFIT_ON_TRAINVAL = True
DOWNLOAD_IF_MISSING = False


# Paths and labels
# <repo>/src/models/fit_linearSVC.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

CLASS_LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "choroidal neovascularization (CNV)",
    1: "diabetic macular edema (DME)",
    2: "drusen",
    3: "normal",
}


def npz_filename(size):
    # MedMNIST stores 28px as octmnist.npz and other sizes as octmnist_<size>.npz.
    if size == 28:
        return "octmnist.npz"
    return f"octmnist_{size}.npz"


def download_dataset(size):
    # medmnist is imported here so it is only needed when actually downloading.
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    from medmnist import OCTMNIST
    OCTMNIST(split="test", download=True, size=size, root=str(DATA_ROOT))


def load_split(npz, split):
    images = npz[f"{split}_images"] # (N, H, W) uint8
    labels = npz[f"{split}_labels"].reshape(-1)  # (N, 1) -> (N,)
    X = images.reshape(images.shape[0], -1).astype(np.float32)
    return X, labels.astype(int)


def build_model(c):
    # Scaler sits inside the pipeline so it is fit on training data only.
    # Deviations from the LinearSVC defaults:
    #   class_weight="balanced" -> the training set is imbalanced
    #   dual=False              -> recommended when n_samples > n_features
    #   max_iter raised         -> liblinear rarely converges at the default 1000
    #   random_state            -> reproducibility
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("svc", LinearSVC(
            C=c,
            class_weight="balanced",
            dual=False,
            max_iter=MAX_ITER,
            random_state=SEED,
        )),
    ])


def compute_metrics(model, X, y):
    predictions = model.predict(X)
    scores = model.decision_function(X)  # (N, n_classes) OvR margins
    y_onehot = label_binarize(y, classes=CLASS_LABELS)
    # The multilabel path gives per-class OvR AUC, macro-averaged, and does not
    # require the scores to sum to 1, so raw decision margins are valid here.
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_auc": roc_auc_score(y_onehot, scores, average="macro"),
        "per_class_recall": recall_score(
            y, predictions, average=None, labels=CLASS_LABELS),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=CLASS_LABELS),
    }


def choose_c(X_train, y_train, X_val, y_val):
    sweep = []
    best_c = None
    best_score = -np.inf
    for c in C_GRID:
        model = build_model(c)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_val, y_val)
        sweep.append((c, metrics["balanced_accuracy"], metrics["macro_auc"]))
        print(f"  C={c:<8g} val balanced_acc={metrics['balanced_accuracy']:.4f} "
              f"val macro_auc={metrics['macro_auc']:.4f}")
        if metrics[SELECTION_METRIC] > best_score:
            best_score = metrics[SELECTION_METRIC]
            best_c = c
    print(f"  -> selected C={best_c:g} on val "
          f"{SELECTION_METRIC}={best_score:.4f}")
    return best_c, sweep


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["CNV", "DME", "drusen", "normal"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("LinearSVC baseline - test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path, best_c, sweep, test_metrics):
    refit_on = "train+val" if REFIT_ON_TRAINVAL else "train"
    lines = [f"# LinearSVC baseline (OCTMNIST {SIZE}x{SIZE})\n"]
    lines.append("## Setup")
    lines.append(f"- Input features: {SIZE * SIZE} flattened pixels")
    lines.append(f"- Pipeline: MinMaxScaler -> LinearSVC")
    lines.append(f"- LinearSVC: class_weight='balanced', dual=False, "
                 f"max_iter={MAX_ITER}, random_state={SEED}")
    lines.append(f"- C selected on validation by: {SELECTION_METRIC}")
    lines.append(f"- Final model refit on: {refit_on}")
    lines.append("")
    lines.append("## Validation sweep (selecting C)")
    lines.append("| C | val balanced_acc | val macro_AUC |")
    lines.append("|---|---|---|")
    for c, bacc, auc in sweep:
        mark = " **(selected)**" if c == best_c else ""
        lines.append(f"| {c:g}{mark} | {bacc:.4f} | {auc:.4f} |")
    lines.append("")
    lines.append("## Test set (evaluated once)")
    lines.append(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    lines.append(f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    lines.append(f"- Macro one-vs-rest AUC: {test_metrics['macro_auc']:.4f}")
    lines.append("- Per-class recall:")
    for k in CLASS_LABELS:
        recall = test_metrics["per_class_recall"][k]
        lines.append(f"    - {CLASS_NAMES[k]}: {recall:.4f}")
    lines.append(f"- Confusion matrix (rows = true, cols = predicted), "
                 f"label order {CLASS_LABELS}:")
    for row in test_metrics["confusion_matrix"]:
        lines.append(f"    {row.tolist()}")
    lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    return text


def main():
    args = parse_args()
    SIZE = args.size
    
    data_path = DATA_ROOT / npz_filename(SIZE)
    if not data_path.exists():
        if DOWNLOAD_IF_MISSING:
            print(f"[download] {data_path.name} not found, downloading ...")
            download_dataset(SIZE)
        else:
            raise FileNotFoundError(
                f"{data_path} not found. Set DOWNLOAD_IF_MISSING = True and "
                f"rerun on a node with internet, or download it beforehand."
            )

    npz = np.load(data_path)
    X_train, y_train = load_split(npz, "train")
    X_val, y_val = load_split(npz, "val")
    X_test, y_test = load_split(npz, "test")
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    print("[select] tuning C on the official validation split:")
    best_c, sweep = choose_c(X_train, y_train, X_val, y_val)

    if REFIT_ON_TRAINVAL:
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
    else:
        X_fit, y_fit = X_train, y_train
    refit_on = "train+val" if REFIT_ON_TRAINVAL else "train"
    print(f"[refit] fitting final model (C={best_c:g}) on {refit_on} "
          f"({X_fit.shape[0]} samples)")
    final_model = build_model(best_c)
    final_model.fit(X_fit, y_fit)

    print("[test] evaluating on the held-out test set")
    test_metrics = compute_metrics(final_model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.modelname
    model_path = MODELS_DIR / f"{model_name}_{SIZE}.joblib"
    dump(final_model, model_path)
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        figures_dir / f"linearsvc_confusion_{SIZE}.png",
    )
    report = write_report(
        REPORTS_DIR / f"baseline_linearsvc_{SIZE}.md",
        best_c, sweep, test_metrics,
    )

    print()
    print(report)
    print(f"[done] model saved to {model_path}")
    print(f"[done] report and figure under {REPORTS_DIR}")


if __name__ == "__main__":
    main()
