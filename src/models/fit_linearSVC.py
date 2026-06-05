from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import dump

seed = 2
modelFileName = "linear_svc_octmnist"

# Load the data
data = np.load(r"data\raw\octmnist_224.npz")

# Load the training data
X_images = data["train_images"]
y = data["train_labels"]

# Reshape the data for sklearn LinearSVC
n_samples = X_images.shape[0]
X_sub_flat = X_images.reshape(n_samples, -1)
y_1d = np.squeeze(y)

# fit the linear svc
clf = make_pipeline(
    StandardScaler(),
    LinearSVC()
)

clf.fit(X_sub_flat, y_1d)

# save the model
dump(clf, f"models/{modelFileName}.joblib")
print("Model saved.")