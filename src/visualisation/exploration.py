## This will visualize the learnt feature maps of the a trained CNN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import sys

def load_model(model_path, device):
    model = torch.load(model_path, weights_only=False)
    model.eval()
    return model

def main():
    
    ## Load in model
    # paths
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_ROOT = REPO_ROOT / "data" / "raw"
    MODEL_ROOT = REPO_ROOT / "models"

    data_path = DATA_ROOT / "octmnist.npz"
    model_path = MODEL_ROOT / "MAM09A2_CNN_8.pt"
    # Problems with the utils file not being recognized by pytorch 
    UTILS_DIR = REPO_ROOT / 'src' / "models"

    # make that directory importable
    sys.path.insert(0, str(UTILS_DIR))
    import utils 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = np.load(data_path)
    images = data["train_images"]  
    print(f"Loaded dataset with shape: {images.shape}")

    # example image
    img = images[0]
    # Make into tensor 
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    # Load model
    model = load_model(model_path, device)
    
    
if __name__ == "__main__":
    main()