import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import copy

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

class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int = 4, image_size: int = 28, hidden_dim: int = 512,hidden_dim2: int = 256):
        super().__init__()
        
        input_vector_size = image_size * image_size
        self.layers = nn.Sequential(
            # Simple architecture. Two fully connected layers of ReLu activation
            nn.Flatten(),
            
            nn.Linear(input_vector_size, hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim,hidden_dim2),
            nn.ReLU(inplace=True),
            
            # Classification part to num classes of logits 
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4, channel_size = 32, image_size  = 28):  
        super().__init__()
        
        # This defines the feature extraction process of a CNN. This learns features such as lines patterns etc from images 
        self.features = nn.Sequential(  
            nn.Conv2d(1, channel_size, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 

            nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)              
        )

        # This prepares the last step of classifying
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_size * 2 * image_size//4 * image_size//4 , 128),  
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EarlyStopping:
    # https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    # Early stopping class from geeksforgeeks.

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = None  # new

    def __call__(self, val_loss, model,epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def train_one_epoch(model, loader, criterion, optimizer, device):
    # Switches the mode the model is in. 
    model.train()
    
    # Initialize vars.
    running_loss = 0.0
    running_correct = 0
    total = 0

    # Loops over all batches in the entire dataset Thus images contains a batch size amount of images.
    for images, labels in loader:
        # Puts the images (BatchSize,1,W,H) on the GPU
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

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs,best_epoch,saveName, reports_dir):
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
    plt.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim((0,1))
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Accuracy subplot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim((0,1))
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)

    saveName = find_availableName(reports_dir, f"{saveName}_learning_curves.png")
    
    plt.suptitle(f"Learning Curves: {saveName}")
    plt.tight_layout()
    
    savePATH = reports_dir / f"{saveName}"
    plt.savefig(savePATH, dpi=150)
    plt.close()
    print(f"Learning curves saved to: {savePATH}")

def find_availableName(FOLDER_PATH,fileName) -> str:
    # Checks if the fileName is already in the FOLDER_PATH and otherwise adds an itterator until it does not yet exist in the FOLDER_PATH
    # Returns the string of the fileName not the path
    FILE_PATH =  FOLDER_PATH / fileName
    
    # Base case if it doesnt exist return original name 
    if not FILE_PATH.exists():
        return fileName 
    # Otherwise find a new fileName that does not exist.
    else:
        # Split the name of the file and the file extention
        p = Path(fileName)
        name = p.stem        
        extension = p.suffix
        
        i = 1
        # Continue endlessly until finds an empty name slot.
        while True:
            newFileName = f"{name}_{i}{extension}"
            
            newFILE_PATH = FOLDER_PATH / newFileName
            if not newFILE_PATH.exists():
                fileName = newFileName 
                return fileName 
            i += 1
        
if __name__ == "__main__":
    main()