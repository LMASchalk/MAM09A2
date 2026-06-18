"""
LinearSVC baseline for the OCTMNIST classification experiment.

Flattened 28x28 grayscale images -> MinMaxScaler -> LinearSVC.

C is tuned on the official validation split (not k-fold cross-validation)
using balanced accuracy. The test set is touched exactly once and scored
with the shared metrics module: accuracy, macro one-vs-rest AUC, macro-F1,
and per-class recall.
"""

import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils import npz_filename, find_availableName, run_tag
import metrics

# Configuration
SIZE = 28
# MODELNAME = "linear_svc_octmnist"
C_GRID = [0.001, 0.01, 0.1, 1.0]
MAX_ITER = 3000
SEED = 42
SELECTION_METRIC = "accuracy"
REFIT_ON_TRAINVAL = True
DOWNLOAD_IF_MISSING = False

# Paths
# <repo>/src/models/fit_linearSVC.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
tag = run_tag("linearsvc", SIZE)

def download_dataset(size):
    # medmnist is imported here so it is only needed when actually downloading.
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    from medmnist import OCTMNIST
    OCTMNIST(split="test", download=True, size=size, root=str(DATA_ROOT))


def load_split(npz, split):
    images = npz[f"{split}_images"]  # (N, H, W) uint8
    labels = npz[f"{split}_labels"].reshape(-1)  # (N, 1) -> (N,)
    X = images.reshape(images.shape[0], -1).astype(np.float32)
    return X, labels.astype(int)


def build_model(c):
    # Scaler sits inside the pipeline so it is fit on training data only.
    # Non-default LinearSVC settings:
    # class_weight balanced: training set is imbalanced
    # dual=False: recommended when n_samples > n_features
    # max_iter raised: liblinear rarely converges at the default 1000
    # random_state: reproducibility
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


def choose_c(X_train, y_train, X_val, y_val):
    sweep = []
    best_c = None
    best_score = -np.inf
    for c in C_GRID:
        model = build_model(c)
        model.fit(X_train, y_train)
        val_bacc = accuracy_score(y_val, model.predict(X_val))
        sweep.append((c, val_bacc))
        print(f"  C={c:<8g} val acc={val_bacc:.4f}")
        if val_bacc > best_score:
            best_score = val_bacc
            best_c = c
    print(f"  -> selected C={best_c:g} on val acc={best_score:.4f}")
    return best_c, sweep


def write_report(path, best_c, sweep, test_metrics):
    refit_on = "train+val" if REFIT_ON_TRAINVAL else "train"
    lines = [f"# LinearSVC baseline (OCTMNIST {SIZE}x{SIZE})\n"]
    lines.append("## Setup")
    lines.append(f"- Input features: {SIZE * SIZE} flattened pixels")
    lines.append("- Pipeline: MinMaxScaler -> LinearSVC")
    lines.append(f"- LinearSVC: class_weight='balanced', dual=False, "
                 f"max_iter={MAX_ITER}, random_state={SEED}")
    lines.append(f"- C selected on validation by: {SELECTION_METRIC}")
    lines.append(f"- Final model refit on: {refit_on}")
    lines.append("")
    lines.append("## Validation sweep (selecting C)")
    lines.append("| C | val acc |")
    lines.append("|---|---|")
    for c, bacc in sweep:
        mark = " **(selected)**" if c == best_c else ""
        lines.append(f"| {c:g}{mark} | {bacc:.4f} |")
    lines.append("")
    lines.append("## Test set (evaluated once)")
    lines.append(metrics.format_metrics(test_metrics))
    lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    return text


def main():
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
    print(f"[data] train={X_train.shape} val={X_val.shape} "
          f"test={X_test.shape}")

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
    test_metrics = metrics.evaluate_sklearn(final_model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = find_availableName(MODELS_DIR, f"{tag}.joblib")
    model_path = MODELS_DIR / model_name
    dump(final_model, model_path)

    report_name = find_availableName(REPORTS_DIR, f"{tag}_metrics.md")
    report = write_report(REPORTS_DIR / report_name, best_c, sweep, test_metrics)

    print()
    print(report)
    print(metrics.format_metrics(test_metrics))
    print(f"[done] model saved to {model_path}")
    print(f"[done] report under {REPORTS_DIR}")


if __name__ == "__main__":
    main()