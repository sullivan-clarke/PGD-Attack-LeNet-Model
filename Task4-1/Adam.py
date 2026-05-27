import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}", file=sys.stderr)
    print(f"Shape of y: {y.shape} {y.dtype}", file=sys.stderr)
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device", file=sys.stderr)

# Define model
# Define model
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [N, 1, 28, 28]
            nn.Conv2d(1, 6, kernel_size=5),   # → [N, 6, 24, 24]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # → [N, 6, 12, 12]

            nn.Conv2d(6, 16, kernel_size=5),  # → [N, 16, 8, 8]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)   # → [N, 16, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                    # → [N, 256]
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = LeNet().to(device)
print(model, file=sys.stderr)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer, num):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=sys.stderr)
    test_loss /= num_batches
    correct /= size
    if num % 5 == 0:
        print(f"{(100*correct):>0.1f}, {test_loss:>8f},", end=" ")

def test(dataloader, model, loss_fn, num):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", file=sys.stderr)
    if num % 5 == 0:
        print(f"{(100*correct):>0.1f}, {test_loss:>8f}")

print("Epochs, Training Accuracy, Training Loss, Accuracy, Loss")

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------", file=sys.stderr)
    if (t + 1) % 5 == 0:
        print(f"{t+1},", end=" ")
    train(train_dataloader, model, loss_fn, optimizer, t+1)
    test(test_dataloader, model, loss_fn, t+1)
print("Done!", file=sys.stderr)

torch.save(model.state_dict(), "Adam.pth")
print("Saved PyTorch Model State", file=sys.stderr)