# LinearSVC baseline (OCTMNIST 28x28)

## Setup
- Input features: 784 flattened pixels
- Pipeline: MinMaxScaler -> LinearSVC
- LinearSVC: class_weight='balanced', dual=False, max_iter=3000, random_state=42
- C selected on validation by: accuracy
- Final model refit on: train+val

## Validation sweep (selecting C)
| C | val acc |
|---|---|
| 0.001 | 0.6488 |
| 0.01 | 0.6627 |
| 0.1 | 0.6662 |
| 1 **(selected)** | 0.6671 |

## Test set (evaluated once)
Accuracy:        0.3520
Macro AUC (OvR): 0.6272
Macro F1:        0.2566
Per-class recall:
  choroidal neovascularization (CNV): 0.7760
  diabetic macular edema (DME): 0.0400
  drusen: 0.0160
  normal: 0.5760
Confusion matrix (rows=true, cols=pred):
  choroidal neovascularization (CNV): [194, 6, 0, 50]
  diabetic macular edema (DME): [151, 10, 7, 82]
  drusen: [146, 2, 4, 98]
  normal: [101, 3, 2, 144]
