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

def register_feature_hooks(layer1, layer2, storage_dict):
    
    # Part of a tutorial on how to make feature maps. This creates a hook that allows to inspect outputs inbetween layers.
        
    def hook_factory(key):
        def hook(module, inp, out):
            storage_dict[key] = out.detach().cpu()
        return hook

    h1 = layer1.register_forward_hook(hook_factory("conv1"))
    h2 = layer2.register_forward_hook(hook_factory("conv2"))
    return h1, h2


def plot_feature_maps(feature_maps, layer_name, max_cols=8):

    # Creates a plot of all the feature maps. Takes the output of the kernels on an image
    
    fm = feature_maps[0]  
    num_channels = fm.shape[0]
    n_cols = min(max_cols, num_channels)
    n_rows = math.ceil(num_channels / n_cols)

    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(num_channels):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(fm[i].numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{layer_name} #{i}", fontsize=6)
    plt.suptitle(f"Feature maps for {layer_name} ({num_channels} channels)")
    plt.tight_layout()
    plt.show()


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

    # Get first two conv layers
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    conv1 = conv_layers[0]
    conv2 = conv_layers[1]
    
    print(f"First conv layer: {conv1}")
    print(f"Second conv layer: {conv2}")

    # Save feature maps of the two layers
    feature_maps = {}
    h1, h2 = register_feature_hooks(conv1, conv2, feature_maps)

    # Forward pass
    with torch.no_grad():
        # Not saving the final forward pass since we will use the hooks to get intermediate output
        _ = model(img_tensor)

    # Need to remove them again otherwise it might mess with later forward passes
    h1.remove()
    h2.remove()

    print(f"conv1 feature maps shape: {feature_maps['conv1'].shape}")
    print(f"conv2 feature maps shape: {feature_maps['conv2'].shape}")

    # Plot feature maps for both layers
    plot_feature_maps(feature_maps["conv1"], "conv1", max_cols=8)
    plot_feature_maps(feature_maps["conv2"], "conv2", max_cols=8)


if __name__ == "__main__":
    main()