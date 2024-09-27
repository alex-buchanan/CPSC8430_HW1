import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import random 

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nInput = nInput
        self.nOutput = nOutput
        self.nHidden = nHidden
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16)
        self.linear1 = nn.Linear(5408, self.nOutput)


    def forward(self, x):
        c1 = self.flatten(self.ReLU(self.conv1(x)))
        logits = self.linear1(c1)
        return logits

model_64 = NeuralNetwork(32, 256, 10).to(device)
model_1024 = NeuralNetwork(32, 256, 10).to(device)
model_agg = NeuralNetwork(32, 256, 10).to(device)
print(model_1024)

loss_fn = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.SGD(model_64.parameters(), lr=1e-3)
optimizer_2 = torch.optim.SGD(model_1024.parameters(), lr=1e-3)

epoch = []
sensitivity_test = [[] for a in range(5)]
cross_entropy_test = [[] for a in range(5)]
sensitivity_train = [[] for a in range(5)]
cross_entropy_train = [[] for a in range(5)]
grad_plot = [[] for a in range(5)]
print(grad_plot)

def train(dataloader, model, loss_fn, optimizer, l_train, i):
    size = len(training_data)
    model.train()
    train_loss = 0.0
    cross_entropy = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        cross_entropy += nn.functional.cross_entropy(pred,y).item()
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        grad_total = 0.0

        for p in model.parameters():
            p.requires_grad_(True)
            # print(f"{p.grad}")
            grad = 0.0
            if p.grad is not None:
                # print(f"Value! {p.grad.shape}")
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_total += grad
        grad_norm=grad_total**0.5
        grad_plot[i].append(grad_norm)

        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= len(dataloader)
    l_train[i].append(train_loss)
    cross_entropy_train[i].append(cross_entropy)

def test(dataloader, model, loss_fn, l_test, i):
    size = len(test_data)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct, cross_entropy = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cross_entropy += nn.functional.cross_entropy(pred,y).item()

    test_loss /= num_batches
    correct /= size
    l_test[i].append(correct*100)
    cross_entropy_test[i].append(cross_entropy)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20

l_test_1 = [[] for a in range(5)]
l_train_1 = [[] for a in range(5)]
batches = [64, 128, 256, 512, 1024]
indx = 0

for i in batches:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch.append(t+1)

        sub_size_train = list([random.randrange(0, len(training_data)) for a in range(len(training_data))])
        sub_size_test = list([random.randrange(0, len(test_data)) for a in range(len(test_data))])

        train_subset = torch.utils.data.Subset(training_data, sub_size_train)
        test_subset = torch.utils.data.Subset(test_data, sub_size_test)

        # Create data loaders.
        
        train_dataloader_1 = DataLoader(train_subset, batch_size=i, shuffle=True)
        test_dataloader_1 = DataLoader(test_subset, batch_size=i)

        train(train_dataloader_1, model_64, loss_fn, optimizer_1, l_train_1, indx)
        test(test_dataloader_1, model_64, loss_fn, l_test_1, indx)
    indx += 1

print("Done!")

model_params = model_64.state_dict()

plt.figure()

ax1 = plt.subplot(1,2,1)

ax1.set_xlabel('batch size (log scale)')
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy')
ax1.plot(batches, [l[-1] for l in l_train_1], 'b')
ax1.plot(batches, [l[-1] for l in l_test_1], 'b--')
ax1.axis([0, 1024, 0, 100])

ax2 = ax1.twinx()
ax2.set_ylabel('Sensitivity')
ax2.plot(batches, [g[-1] for g in grad_plot], 'r')

ax3 = plt.subplot(1,2,2)
ax3.set_xlabel('batch size (log scale)')
ax3.set_xscale('log')
ax3.set_ylabel('Cross Entropy Loss')
ax3.plot(batches, [l[-1] for l in cross_entropy_train], 'b')
ax3.plot(batches, [l[-1] for l in cross_entropy_test], 'b--')
ax3.axis([0, 1024, 0, 100])

ax4 = ax3.twinx()
ax4.set_ylabel('Sensitivity')
ax4.plot(batches, [g[-1] for g in grad_plot], 'r')

plt.suptitle('Accuracy Variation', fontsize=16)
plt.tight_layout()

plt.show()