"""
Shared evaluation metrics for the OCTMNIST classification experiment.

Both baselines report the same four metrics on the full test set:
accuracy, macro one-vs-rest AUC, macro-F1, and per-class recall.

Metric computation is decoupled from the model. A caller turns its model
output into three arrays (true labels, predicted labels, per-class scores)
and passes them to compute_metrics. Two collectors cover the model types in
this project: one for an sklearn estimator, one for a torch model read from
a DataLoader.
"""

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

CLASS_LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "choroidal neovascularization (CNV)",
    1: "diabetic macular edema (DME)",
    2: "drusen",
    3: "normal",
}


def compute_metrics(y_true, y_pred, y_score, labels=CLASS_LABELS):
    # y_score is (N, n_classes): decision margins or class probabilities.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    y_onehot = label_binarize(y_true, classes=labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_auc": roc_auc_score(y_onehot, y_score, average="macro"),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=labels),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def collect_sklearn(model, X, y):
    # decision_function gives (N, n_classes) one-vs-rest margins for AUC.
    y_true = np.asarray(y)
    y_pred = model.predict(X)
    y_score = model.decision_function(X)
    return y_true, y_pred, y_score


def collect_torch(model, loader, device):
    # Softmax over logits gives probabilities for AUC; argmax gives the label.
    import torch

    model.eval()
    true_chunks = []
    pred_chunks = []
    score_chunks = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
            true_chunks.append(labels.numpy())
            pred_chunks.append(preds.cpu().numpy())
            score_chunks.append(probs.cpu().numpy())
    y_true = np.concatenate(true_chunks)
    y_pred = np.concatenate(pred_chunks)
    y_score = np.concatenate(score_chunks)
    return y_true, y_pred, y_score


def evaluate_sklearn(model, X, y, labels=CLASS_LABELS):
    y_true, y_pred, y_score = collect_sklearn(model, X, y)
    return compute_metrics(y_true, y_pred, y_score, labels)


def evaluate_torch(model, loader, device, labels=CLASS_LABELS):
    y_true, y_pred, y_score = collect_torch(model, loader, device)
    return compute_metrics(y_true, y_pred, y_score, labels)


def format_metrics(metrics, labels=CLASS_LABELS):
    lines = [
        f"Accuracy:        {metrics['accuracy']:.4f}",
        f"Macro AUC (OvR): {metrics['macro_auc']:.4f}",
        f"Macro F1:        {metrics['macro_f1']:.4f}",
        "Per-class recall:",
    ]
    for i, label in enumerate(labels):
        name = CLASS_NAMES.get(label, str(label))
        lines.append(f"  {name}: {metrics['per_class_recall'][i]:.4f}")
    cm = metrics["confusion_matrix"]
    lines.append("Confusion matrix (rows=true, cols=pred):")
    for label, row in zip(labels, cm):
        name = CLASS_NAMES.get(label, str(label))
        lines.append(f"  {name}: {row.tolist()}")
    return "\n".join(lines)
