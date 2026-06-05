from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import dump
from pathlib import Path

data_folder = Path.home() / "MAM09A2" / "data" / "raw"
models_folder = Path.home() / "MAM09A2" / "models"
modelFileName = "linear_svc_octmnist.joblib"

seed = 2

# Load the data
data = np.load(data_folder / "octmnist_224.npz")

# Load the training data
X_images = data["train_images"]
y = data["train_labels"]

# Reshape the data for sklearn LinearSVC
n_samples = X_images.shape[0]
X_sub_flat = X_images.reshape(n_samples, -1)
y_1d = np.squeeze(y)

# fit the linear svc
clf = make_pipeline(
    StandardScaler(with_mean=False),
    LinearSVC()
)

clf.fit(X_sub_flat, y_1d)

# save the model
dump(clf, models_folder / modelFileName)
print("Model saved.")