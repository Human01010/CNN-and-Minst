import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

#noralize the data
# Define the transformations
transform = transforms.Compose(
    [
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
# print(transform)
# Load the original dataset
original_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# Load the augmented dataset
augmented_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)

# Combine the original and augmented datasets
combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

#train and validation dataset
train_set = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
val_set = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=combined_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(dataset=datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True), batch_size=10, shuffle=False)

# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.fc3(x)  # Calculate scores
        return scores  # Return the scores


# Initialize the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the appropriate device
model = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
epoch_losses = []  # Initialize before the training loop
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    epoch_loss = sum(losses)/len(losses) if losses else 0
    epoch_losses.append(epoch_loss)
    print(f"loss at epoch {epoch} is {epoch_loss}")

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")

# Plot the loss progression
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Progression Over Epochs')
plt.show()