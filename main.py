import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models import ResNet18
from dataset import prepare_dataset

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define directory to save TensorBoard logs
log_dir = './logs'
writer = SummaryWriter(log_dir)

# Define the directory containing your dataset
data_dir = "/home/kishan/Documents/projects/Lung_disease_prediction/dataset"

# Prepare the dataset
train_loader, test_loader, classes = prepare_dataset(data_dir)

# Define number of classes
num_classes = len(classes)

# Initialize ResNet-18 model
model = ResNet18(num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function for training the model
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        # Log loss to TensorBoard
        writer.add_scalar('Train Loss', loss.item(), epoch*len(train_loader) + batch_idx)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# Function for testing the model
def test(model, test_loader, criterion, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    
    # Log accuracy to TensorBoard
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    
    return accuracy

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
    test_accuracy = test(model, test_loader, criterion, device, epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Close the TensorBoard writer
writer.close()
