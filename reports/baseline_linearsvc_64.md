# LinearSVC baseline (OCTMNIST 64x64)

## Setup
- Input features: 4096 flattened pixels, reduced to 400 PCA components
- Pipeline: MinMaxScaler -> PCA(400) -> LinearSVC
- LinearSVC: class_weight='balanced', dual=False, max_iter=5000, random_state=42
- C selected on validation by: balanced_accuracy
- Final model refit on: train+val

## Validation sweep (selecting C)
| C | val balanced_acc | val macro_AUC |
|---|---|---|
| 0.001 | 0.4802 | 0.7637 |
| 0.01 | 0.4928 | 0.7653 |
| 0.1 | 0.4953 | 0.7653 |
| 1 **(selected)** | 0.4954 | 0.7653 |
| 10 | 0.4953 | 0.7653 |

## Test set (evaluated once)
- Accuracy: 0.3600
- Balanced accuracy: 0.3600
- Macro one-vs-rest AUC: 0.6234
- Per-class recall:
    - choroidal neovascularization (CNV): 0.7720
    - diabetic macular edema (DME): 0.0120
    - drusen: 0.0240
    - normal: 0.6320
- Confusion matrix (rows = true, cols = predicted), label order [0, 1, 2, 3]:
    [193, 2, 2, 53]
    [150, 3, 9, 88]
    [157, 1, 6, 86]
    [88, 3, 1, 158]
