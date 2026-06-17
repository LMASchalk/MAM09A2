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
| 0.001 **(selected)** | 0.6488 |

## Test set (evaluated once)
- Accuracy: 0.3760
- AUC: 0.6393
- Macro-F1: 0.2588
- Per-class recall:
    - choroidal neovascularization (CNV): 0.7160
    - diabetic macular edema (DME): 0.0160
    - drusen: 0.0000
    - normal: 0.7720
