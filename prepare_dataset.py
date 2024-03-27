import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def prepare_dataset(data_dir, batch_size=32):
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),           # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    # Create dataset from the folder structure
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Print size of the first image in the dataset
    img, _ = dataset[0]
    print("Size of the first image:", img.size())

    # Define the size of your training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset into training and testing sets
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Create DataLoader for testing set
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Define classes
    classes = dataset.classes

    # Print number of samples in each set
    print(f"Number of samples in training set: {len(train_set)}")
    print(f"Number of samples in testing set: {len(test_set)}")
    print(f"Classes: {classes}")

    return train_loader, test_loader, classes
