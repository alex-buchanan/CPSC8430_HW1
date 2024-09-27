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

batch_size_1 = 64
batch_size_2 = 1024

# Get cpu, gpu or mps device for training.
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
        """225"""
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
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
optimizer_1 = torch.optim.SGD(model_64.parameters(), lr=1e-2)
optimizer_2 = torch.optim.SGD(model_1024.parameters(), lr=1e-2)

epoch = []

def train(dataloader, model, loss_fn, optimizer, l_train):
    size = len(training_data)
    model.train()
    train_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
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
        grad_plot.append(grad_norm)

        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= len(dataloader)
    l_train.append(train_loss)

def test(dataloader, model, loss_fn, l_test):
    size = len(test_data)
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
    l_test.append(correct*100)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20

grad_plot = []
l_test_1 = []
l_train_1 = []
l_test_2 = []
l_train_2 = []
theta_alpha_test_accy = []
cross_entropy_test = []
theta_alpha_train_accy = []
cross_entropy_train = []


for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    epoch.append(t+1)

    sub_size_train = list([random.randrange(0, len(training_data)) for a in range(len(training_data))])
    sub_size_test = list([random.randrange(0, len(test_data)) for a in range(len(test_data))])

    train_subset = torch.utils.data.Subset(training_data, sub_size_train)
    test_subset = torch.utils.data.Subset(test_data, sub_size_test)

    # Create data loaders.
    train_dataloader_1 = DataLoader(train_subset, batch_size=batch_size_1, shuffle=True)
    test_dataloader_1 = DataLoader(test_subset, batch_size=batch_size_1)

    train(train_dataloader_1, model_64, loss_fn, optimizer_1, l_train_1)
    test(test_dataloader_1, model_64, loss_fn, l_test_1)

    # Create data loaders.
    train_dataloader_2 = DataLoader(train_subset, batch_size=batch_size_2, shuffle=True)
    test_dataloader_2 = DataLoader(test_subset, batch_size=batch_size_2)

    train(train_dataloader_2, model_1024, loss_fn, optimizer_2, l_train_2)
    test(test_dataloader_2, model_1024, loss_fn, l_test_2)
print("Done!")

model_params = model_64.state_dict()

fig1, ax1 = plt.subplots()

for a in [(t/100.0)-1.0 for t in range(0,300)]:
    state_1 = model_64.state_dict()
    state_2 = model_1024.state_dict()
    for layer in state_1:
        model_params[layer] = (1-a)*state_1[layer] + a*state_2[layer]   
    model_agg.load_state_dict(model_params)

    train_loss, correct, cross_entropy = 0, 0, 0
    for X, y in train_dataloader_2:
        X, y = X.to(device), y.to(device)
        pred = model_agg(X)
        train_loss += loss_fn(pred, y).item()
        cross_entropy += nn.functional.cross_entropy(pred,y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    theta_alpha_train_accy.append(correct/len(training_data)*100)
    cross_entropy_train.append(cross_entropy)

    test_loss, correct, cross_entropy = 0, 0, 0
    for X, y in test_dataloader_2:
        X, y = X.to(device), y.to(device)
        pred = model_agg(X)
        test_loss += loss_fn(pred, y).item()
        cross_entropy += nn.functional.cross_entropy(pred,y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    theta_alpha_test_accy.append(correct/len(test_data)*100)
    cross_entropy_test.append(cross_entropy)

ax1.set_xlabel('$ \\alpha $')
ax1.set_ylabel('Accuracy')
ax1.plot([(t/100.0)-1.0 for t in range(0,300)], theta_alpha_train_accy, 'r')
ax1.plot([(t/100.0)-1.0 for t in range(0,300)], theta_alpha_test_accy, 'r--')
ax1.axis([-1, 2, 0, 100])
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('Entropy')
ax2.plot([(t/100.0)-1.0 for t in range(0,300)], cross_entropy_train, 'b')
ax2.plot([(t/100.0)-1.0 for t in range(0,300)], cross_entropy_test, 'b--')
ax2.set_yscale('log')
ax2.axis([-1, 2, 0.1, 1000])
ax2.legend()

fig1.suptitle('$ \\alpha $ Variation', fontsize=16)
fig1.tight_layout()

plt.show()