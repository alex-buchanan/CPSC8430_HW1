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
        """225"""
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.linear1 = nn.Linear(3200, self.nHidden)
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
        # f = self.flatten(c2)
        l1 = self.ReLU(self.linear1(c2))
        l2 = self.ReLU(self.linear2(l1))
        l3 = self.ReLU(self.linear3(l2))
        logits = self.linear4(l3)
        return logits

model = NeuralNetwork(32, 1024, 10).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

l_1 = []
weight_1 = []
weight_2 = []
weight_3 = []
weight_4 = []
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
dataTensor_1 = torch.empty((1,3200), dtype=torch.float32).fill_(0)
dataTensor_2 = torch.empty((1,1024), dtype=torch.float32).fill_(0)
dataTensor_3 = torch.empty((1,1024), dtype=torch.float32).fill_(0)
dataTensor_4 = torch.empty((1,1024), dtype=torch.float32).fill_(0)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch.append(t+1)
    train(train_dataloader, model, loss_fn, optimizer)

    ###################################################
    m_1 = model.linear1
    wData_1 = m_1.weight.data
    result_1 = np.mean(np.array(torch.Tensor.cpu(wData_1)), 0)
    dataTensor_1 = torch.cat((dataTensor_1, torch.tensor(result_1[None,:])), 0)
    if t % 3 == 0:
        weight_1.append(np.array(torch.pca_lowrank(dataTensor_1, q=2)[1]))
        dataTensor_1 = torch.empty((1,3200), dtype=torch.float32)
    ###################################################
    m_2 = model.linear2
    wData_2 = m_2.weight.data
    result_2 = np.mean(np.array(torch.Tensor.cpu(wData_2)), 0)
    dataTensor_2 = torch.cat((dataTensor_2, torch.tensor(result_2[None,:])), 0)
    if t % 3 == 0:
        weight_2.append(np.array(torch.pca_lowrank(dataTensor_2, q=2)[1]))
        dataTensor_2 = torch.empty((1,1024), dtype=torch.float32)
    ###################################################
    m_3 = model.linear3
    wData_3 = m_3.weight.data
    result_3 = np.mean(np.array(torch.Tensor.cpu(wData_3)), 0)
    dataTensor_3 = torch.cat((dataTensor_3, torch.tensor(result_3[None,:])), 0)
    if t % 3 == 0:
        weight_3.append(np.array(torch.pca_lowrank(dataTensor_3, q=2)[1]))
        dataTensor_3 = torch.empty((1,1024), dtype=torch.float32)
    ###################################################
    m_4 = model.linear4
    wData_4 = m_4.weight.data
    result_4 = np.mean(np.array(torch.Tensor.cpu(wData_4)), 0)
    dataTensor_4 = torch.cat((dataTensor_4, torch.tensor(result_4[None,:])), 0)
    if t % 3 == 0:
        weight_4.append(np.array(torch.pca_lowrank(dataTensor_4, q=2)[1]))
        dataTensor_4 = torch.empty((1,1024), dtype=torch.float32)
    ###################################################

    test(test_dataloader, model, loss_fn)
print("Done!")

# plot the data
# plt.figure()
# plt.plot(epoch, l_1, color="r", alpha= 1.0, label= "Plt_1")
# plt.yscale('log')

fig, ax = plt.subplots()

x_1 = [x[0] for x in weight_1]
y_1 = [y[1] for y in weight_1]
ax.scatter(x_1,y_1, c='tab:blue', label='L_1')

x_2 = [x[0] for x in weight_2]
y_2 = [y[1] for y in weight_2]
ax.scatter(x_2,y_2, c='tab:green', label='L_2')

x_3 = [x[0] for x in weight_3]
y_3 = [y[1] for y in weight_3]
ax.scatter(x_3,y_3, c='tab:red', label='L_3')

x_4 = [x[0] for x in weight_4]
y_4 = [y[1] for y in weight_4]
ax.scatter(x_4,y_4, c='tab:orange', label='L_4')

ax.legend()

plt.show()