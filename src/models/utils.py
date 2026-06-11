import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

class OCTMNISTDataset(Dataset):
    # This class is a subclass of the dataset class of torch. This class should be able to 
    # extract single samples and batches
    
    def __init__(self, npz_path, split="train", transform=None):
        super().__init__()

        data = np.load(npz_path)

        if split == "train":
            self.images = data["train_images"] 
            self.labels = data["train_labels"].reshape(-1)   
        elif split == "val":
            self.images = data["val_images"]
            self.labels = data["val_labels"].reshape(-1) 
        elif split == "test":
            self.images = data["test_images"]
            self.labels = data["test_labels"].reshape(-1) 
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images assumed to be (H, W) grayscale; adjust if different
        img = self.images[idx].astype(np.float32)
        label = int(self.labels[idx])

        # Add channel dimension: (1, H, W)
        # This is because torch expects (Channels, H,W) And since this is grayscale it will just be 1 
        img = np.expand_dims(img, axis=0)

        # Convert to torch tensors
        img = torch.from_numpy(img)              # shape: (1, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4, channel_size = 32, image_size  = 28):  
        super().__init__()
        
        # This defines the feature extraction process of a CNN. This learns features such as lines patterns etc from images 
        self.features = nn.Sequential(
            nn.Conv2d(1, channel_size, kernel_size=3, padding=1),  # (B, 1, H, W) -> (B, 32, H, W)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 # (B, 32, H/2, W/2)

            nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, padding=1), # (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 # (B, 64, H/4, W/4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_size * 2 * image_size//4 * image_size//4 , 128),  
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    # Switches the mode the model is in. 
    model.train()
    
    # Initialize vars.
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        # Puts the images on the GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Reset the gradient.
        optimizer.zero_grad()
        
        # Creates the gradient and adjusts
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Results over a batch
        _, preds = outputs.max(1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    # Switch to evaluation mode
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def npz_filename(size):
    # MedMNIST stores 28px as octmnist.npz and other sizes as octmnist_<size>.npz.
    if size == 28:
        return "octmnist.npz"
    return f"octmnist_{size}.npz"

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, model_name, reports_dir):
    """
    Make a figure with loss and accuracy curves and save to REPORTS_DIR.
    
    Args:
        train_losses (list of float): training loss per epoch
        val_losses   (list of float): validation loss per epoch
        train_accs   (list of float): training accuracy per epoch
        val_accs     (list of float): validation accuracy per epoch
        model_name   (str): name for the model (used in filename/title)
        reports_dir  (Path): directory where the figure will be saved
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    # ---- Loss subplot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Accuracy subplot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"Learning Curves: {model_name}")
    plt.tight_layout()

    save_path = reports_dir / f"{model_name}_learning_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curves saved to: {save_path}")

if __name__ == "__main__":
    main()