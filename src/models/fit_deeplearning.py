import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from utils import npz_filename, OCTMNISTDataset,SimpleMLP,SimpleCNN,train_one_epoch,evaluate,plot_learning_curves,find_availableName,EarlyStopping

# All the important paths 
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--size",
        type=int,
        default=28,          
        help="Image size to use for OCTMNIST. Choices are: 244,128,64,28 (Default)",
    )
    
    parser.add_argument(
        "--modelname",
        type=str,
        default="MAM09A2",          
        help="The name of the saved model. Default: linear_svc_octmnist",
    )

    parser.add_argument(
        "--modeltype",
        type=str,
        default="MLP",          
        help="The type of deep learning model, Options: MLP (Default), CNN",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    return parser.parse_args()
        

def main():
    print("Starting ...")
    
    seed = 42
    
    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reproducability
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    
    
    # Unpack arguments
    args = parse_args()
    size = args.size
    name = args.modelname
    modeltype = args.modeltype
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.epochs    
    
    # Load the data
    data_path = DATA_ROOT / npz_filename(size)
    train_dataset = OCTMNISTDataset(data_path, split="train")
    val_dataset = OCTMNISTDataset(data_path, split="val")
    test_dataset = OCTMNISTDataset(data_path, split="test")
    print("Dataset loaded!")
    
    # Initialize the dataloader. This class handles the batch generation. 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # TODO: adjust for your server
        pin_memory=True,    # good with GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )    
    print("Dataloader Initialized!")
    
    # Initialize the model and optimizer
    num_classes = 4  
    if modeltype == "MLP":
        model = SimpleMLP(num_classes=num_classes,image_size = size).to(device)  
        print("MLP Initialized!") 
    elif modeltype == "CNN":
        model = SimpleCNN(num_classes=num_classes,image_size = size).to(device)  
        print("CNN Initialized!")  
    
    criterion = nn.CrossEntropyLoss()
    print(f"The loss function is: {criterion}")
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    print(f"The optimizer is: {optimizer}")
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    
    # To generate a figure of the loss over the training
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []    
    
    # Training loop
    print("Initializing training loop ...")
    for epoch in range(num_epochs):
        # Single epoch. A epoch is one pass through the entire data. Say I have 10.000 samples
        # and a batch size of 100 then after 100 iterations one epoch is finshed.
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)        
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )    
        
        # Detect early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Load the best model of the epochs
    print(f"Best model was from epoch {early_stopping.best_epoch + 1}") 
    early_stopping.load_best_model(model)
    
    # Evaluate the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save model file and ensure that it does not overwrite existing files.
    fileName = find_availableName(MODELS_DIR,f"{name}_{modeltype}.pt")
    savePATH = MODELS_DIR / fileName    
    torch.save(model, savePATH)
    print(f"Model saved to: {savePATH}")

    # Creates a visual on the training loss and accuracy over epochs
    p = Path(fileName)
    plot_learning_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        best_epoch = early_stopping.best_epoch + 1,
        saveName=p.stem,
        reports_dir=REPORTS_DIR
    )   

if __name__ == "__main__":
    main()