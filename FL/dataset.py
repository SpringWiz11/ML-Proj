import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


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


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Download MNIST and generate IID partitions."""

    # download MNIST in case it's not already in the system
    trainset, testset, classes = prepare_dataset()

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_images = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_images] * num_partitions

    # split randomly. This returns a list of trainsets, each with `num_images` training examples
    # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
    # would induce heterogeneity in the partitions in the form of for example: each client getting a different
    # amount of training examples, each client having a different distribution over the labels (maybe even some
    # clients not having a single training example for certain classes). If you are curious, you can check online
    # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e. we don't partition it)
    # This test set will be left on the server side and we'll be used to evaluate the
    # performance of the global model after each round.
    # Please note that a more realistic setting would instead use a validation set on the server for
    # this purpose and only use the testset after the final round.
    # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
    # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
    # in main.py above the strategy definition for more details on this)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader