from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import dump
from pathlib import Path
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
        "--modelname,
        type=str,
        default="linear_svc_octmnist",          # fallback if you don't pass --size
        help="The name of the saved model. Default: linear_svc_octmnist",
    )
        
    return parser.parse_args()

def main():
    args = parse_args()

    data_folder = Path.home() / "MAM09A2" / "data" / "raw"
    models_folder = Path.home() / "MAM09A2" / "models"
    modelFileName = f"{args.modelname}.joblib"

    seed = 2

    # Load the data
    if args.size == 28:
        # The default does not have a number in front
        data = np.load(data_folder / "octmnist.npz")
    else: 
        data = np.load(data_folder / f"octmnist_{args.size}.npz")

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