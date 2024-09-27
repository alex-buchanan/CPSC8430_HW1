import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 

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

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.linear1 = nn.Linear(3200, self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nOutput)
        # self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear4 = nn.Linear(self.nHidden, self.nOutput)

    def forward(self, x):
        # x = self.flatten(x)
        # print(x.shape)
        c1 = self.ReLU(self.conv1(x))
        # print(f"c1 weights: {c1.weights.data}")
        c2 = self.flatten(self.ReLU(self.conv2(c1)))
        # f = self.flatten(c2)
        l1 = self.ReLU(self.linear1(c2))
        # l2 = self.ReLU(self.linear2(l1))
        # l3 = self.ReLU(self.linear3(l2))
        logits = self.linear2(l1)
        return logits
num_models = 10
model = []
for i in range(num_models):
    model.append(NeuralNetwork(32, 1000*i, 10).to(device))


loss_fn = nn.CrossEntropyLoss()
optimizer = []
for i in range(num_models):
    optimizer.append(torch.optim.SGD(model[i].parameters(), lr=1e-3))

loss_train = []
loss_test  = []
accy_train = []
accy_test  = []

def train(dataloader, model, loss_fn, optimizer, loss_, accy_):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    loss_.append(train_loss/num_batches)
    accy_.append(correct/size * 100)

def test(dataloader, model, loss_fn, loss_, accy_):
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
    loss_.append(test_loss)
    accy_.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# epochs = 10

# for t in range(epochs):

for i in range(num_models):
    print(f"Model {i+1}\n-------------------------------")
    for epoch in range(3):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model[i], loss_fn, optimizer[i], loss_train, accy_train)
        test(test_dataloader, model[i], loss_fn, loss_test, accy_test)

print("Done!")

# plot the data
fig = plt.figure()

# training & testing losses
ax_1 = plt.subplot(1,2,1)
ax_1.set_xlabel('Parameters')
ax_1.set_ylabel('Loss')

ax_1.plot([1000*i*2 for i in range(num_models)], loss_test[0::3], 'r', label= "Loss_Test")
ax_1.plot([1000*i*2 for i in range(num_models)], loss_train[0::3], 'r--', label= "Loss_Train")

ax_1.legend()

ax_2 = plt.subplot(1,2,2)
ax_2.set_xlabel('Parameters')
ax_2.set_ylabel('Accuracy')

ax_2.plot([1000*i*2 for i in range(num_models)], accy_test[0::3], 'r', label= "Accy_Test")
ax_2.plot([1000*i*2 for i in range(num_models)], accy_train[0::3], 'r--', label= "Accy_Train")

ax_2.legend()

fig.suptitle('Loss & Accuracy')

plt.show()