import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.1

# --- Data ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

transform = transforms.ToTensor()

train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
# Multinomial logistic regression: one linear layer, no hidden layers.
# Input is a flattened 28x28 image (784 pixels), output is 10 class scores.
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # flatten: (batch, 1, 28, 28) -> (batch, 784)
        scores = self.linear(x)             # linear transformation
        log_probs = torch.log(torch.softmax(scores, dim=1))  # log of softmax probabilities
        return log_probs

model = LogisticRegression(input_dim=784, num_classes=10)

# --- Loss and Optimizer ---
# NLLLoss expects log-probabilities as input (which is what we return above)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluate on test set at end of each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            log_probs = model(images)
            predictions = log_probs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch + 1:2d}/{EPOCHS}  loss: {avg_loss:.4f}  test accuracy: {accuracy:.4f}")

print("\nDone.")
