#!/usr/bin/env python3
"""
ml_utils.py

Author: Zane Milo
Created: 2025-03-02
Purpose: Provides a robust utility module for machine learning experiments.
         This module includes functions to load datasets, build models, train models,
         evaluate their performance, and manage model checkpoints. It is designed to be
         flexible and configurable for different datasets (e.g., MNIST and CIFAR-10) and
         can be easily integrated into other projects.

Usage Example:
    from ml_utils import load_dataset, build_model, train_model, evaluate_model, save_checkpoint, load_checkpoint

    # Load MNIST dataset with a batch size of 64
    train_loader, test_loader = load_dataset('mnist', batch_size=64)
    
    # Build a model for MNIST
    model = build_model('mnist')
    
    # Define loss and optimizer
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model for 5 epochs and optionally save it
    model = train_model(model, train_loader, criterion, optimizer, epochs=5, save_path='mnist_model.pth')
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    print(f"MNIST Test Accuracy: {accuracy:.2f}%")

License: MIT License
"""

import json
import logging
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Custom Exceptions
class DatasetNotSupportedError(Exception):
    """Exception raised when an unsupported dataset is specified."""
    pass

class ModelBuildError(Exception):
    """Exception raised when model building fails."""
    pass

def load_dataset(dataset: str, batch_size: int = 64, download: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Load a dataset and return train and test DataLoaders.

    Parameters:
        dataset (str): The dataset to load ('mnist' or 'cifar10').
        batch_size (int): Batch size for the DataLoaders.
        download (bool): Whether to download the dataset if not available.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dataset_lower = dataset.lower()
    if dataset_lower == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=download)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=download)
    elif dataset_lower == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=download)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=download)
    else:
        logging.error("Unsupported dataset: %s", dataset)
        raise DatasetNotSupportedError(f"Unsupported dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logging.info("Loaded %s dataset with batch size %d", dataset, batch_size)
    return train_loader, test_loader

def build_model(dataset: str) -> nn.Module:
    """
    Build a simple model for the specified dataset.

    Parameters:
        dataset (str): The dataset for which to build a model ('mnist' or 'cifar10').

    Returns:
        nn.Module: The constructed model.
    """
    dataset_lower = dataset.lower()
    if dataset_lower == 'mnist':
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.view(-1, 28 * 28)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        model = SimpleNN()
    elif dataset_lower == 'cifar10':
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64 * 8 * 8)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        model = CNN()
    else:
        logging.error("Unsupported dataset for model building: %s", dataset)
        raise ModelBuildError(f"Unsupported dataset for model building: {dataset}")

    logging.info("Built model for %s dataset", dataset)
    return model

def train_model(model: nn.Module,
                train_loader: DataLoader,
                criterion: Any,
                optimizer: Any,
                epochs: int = 5,
                device: Optional[torch.device] = None,
                save_path: Optional[str] = None) -> nn.Module:
    """
    Train the model with the given parameters.

    Parameters:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs (int): Number of training epochs.
        device (Optional[torch.device]): Device to run training on (CPU or GPU). Defaults to CPU if not provided.
        save_path (Optional[str]): If provided, saves the model's state dict after training.

    Returns:
        nn.Module: The trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Training on device: %s", device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logging.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg_loss)

    if save_path:
        torch.save(model.state_dict(), save_path)
        logging.info("Model saved to %s", save_path)

    return model

def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: Optional[torch.device] = None) -> float:
    """
    Evaluate the model on the test dataset.

    Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (Optional[torch.device]): Device to run evaluation on. Defaults to CPU if not provided.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info("Evaluation Accuracy: %.2f%%", accuracy)
    return accuracy

def save_checkpoint(model: nn.Module, path: str) -> None:
    """
    Save the model's state dict to the specified path.

    Parameters:
        model (nn.Module): The model to save.
        path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    logging.info("Checkpoint saved to %s", path)

def load_checkpoint(model: nn.Module, path: str, device: Optional[torch.device] = None) -> nn.Module:
    """
    Load the model's state dict from the specified path.

    Parameters:
        model (nn.Module): The model to load the state dict into.
        path (str): Path from where the model will be loaded.
        device (Optional[torch.device]): Device for map_location. Defaults to CPU if not provided.

    Returns:
        nn.Module: The model with loaded state.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    logging.info("Checkpoint loaded from %s", path)
    return model

def cli_main() -> None:
    """
    Command-line interface to test the ml_utils module.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ML Utilities Test")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
                        help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file for hyperparameters")
    args = parser.parse_args()

    # Load hyperparameters from configuration file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        args.dataset = config.get("dataset", args.dataset)
        args.epochs = config.get("epochs", args.epochs)
        args.batch_size = config.get("batch_size", args.batch_size)
        args.lr = config.get("lr", args.lr)
        args.save_path = config.get("save_path", args.save_path)

    # Set default epochs if not provided
    if args.epochs is None:
        args.epochs = 5 if args.dataset.lower() == "mnist" else 10

    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)
    model = build_model(args.dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(model, train_loader, criterion, optimizer, epochs=args.epochs, save_path=args.save_path)
    accuracy = evaluate_model(model, test_loader)
    print(f"{args.dataset.upper()} Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    cli_main()
