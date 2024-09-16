import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

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

batch_size = 32

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")

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
        self.conv1 = nn.Conv2d(self.nInput, self.nInput, kernel_size=18)
        self.conv2 = nn.Conv2d(self.nInput, self.nInput, kernel_size=7)
        self.linear1 = nn.Linear(25, self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nOutput)


        ''' self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=18),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ) '''

    def forward(self, x):
        # x = self.flatten(x)
        # print(x.shape)
        c1 = self.ReLU(self.conv1(x))
        # print(f"c1 weights: {c1.weights.data}")
        c2 = self.flatten(self.ReLU(self.conv2(c1)))
        l1 = self.ReLU(self.linear1(c2))
        l2 = self.ReLU(self.linear2(l1))
        l3 = self.ReLU(self.linear3(l2))
        logits = self.linear4(l3)
        return logits

model = NeuralNetwork(1, 512, 10).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

l_1 = []
weight = []
epoch = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
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
    l_1.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20
dataTensor = torch.empty((1,25), dtype=torch.float32)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch.append(t+1)
    train(train_dataloader, model, loss_fn, optimizer)
    m = model.linear1
    wData = m.weight.data
    # print(f"wData Size: {wData.shape}")
    bData = m.bias.data
    # print(f"bData Size: {bData.shape}")
    result = torch.matmul(bData, wData)
    result = result[None,:]
    # print(f"result Size: {result.shape}")
    dataTensor = torch.cat((dataTensor, torch.Tensor.cpu(result)), 0)
    # print(f"dataTensor Size: {dataTensor.shape}")
    if t % 3 == 0:
        weight.append(np.array(torch.pca_lowrank(dataTensor, q=2)[1]))
        dataTensor = torch.empty((1,25), dtype=torch.float32)
        # print(f"Weights: {torch.pca_lowrank(m.weight.data, q=2)[1]}")
        # print(f"Bias: {m.bias.data}")
    test(test_dataloader, model, loss_fn)
print("Done!")

# plot the data
plt.figure()
plt.plot(epoch, l_1, color="r", alpha= 1.0, label= "Plt_1")
# plt.yscale('log')

plt.figure()
x = [x[0] for x in weight]
y = [y[1] for y in weight]
plt.scatter(x,y)
plt.show()