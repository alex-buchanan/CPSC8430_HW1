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

batch_size = 64

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
        super(NeuralNetwork, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.nInput,  self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nOutput)
        # self.linear5 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear6 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear7 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear8 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        f = self.flatten(x)
        h1 = self.ReLU(self.linear1(f))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        # h4 = self.ReLU(self.linear4(h3))
        # h5 = self.ReLU(self.linear5(h4))
        # h6 = self.ReLU(self.linear6(h5))
        # h7 = self.ReLU(self.linear7(h5))
        output = self.linear4(h3) ## <- NOTE
        return(output)

model = NeuralNetwork(784, 256, 10).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

l_test = []
l_train = []
epoch = []

class MyTwistedMNIST(torch.utils.data.Dataset):
  def __init__(self, my_args):
    super(MyTwistedMNIST, self).__init__()
    self.orig_mnist = test_data 

  def __getitem__(self, index):
    x, y = self.orig_mnist[index]  # get the original item
    x1, y1 = self.orig_mnist[random.randrange(0, len(self)) ]
    my_x = x
    my_y = y1
    return my_x, my_y

  def __len__(self):
    return self.orig_mnist.__len__()

def train(dataloader, model, loss_fn, optimizer):
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

def test(dataloader, model, loss_fn):
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
    l_test.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100

grad_plot = []


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch.append(t+1)

    sub_size_train = list([random.randrange(0, len(training_data)) for a in range(len(training_data)//2)])
    sub_size_test = list([random.randrange(0, len(test_data)) for a in range(len(test_data)//2)])

    train_subset = MyTwistedMNIST(torch.utils.data.Subset(training_data, sub_size_train))
    test_subset = torch.utils.data.Subset(test_data, sub_size_test)

    # Create data loaders.
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(train_subset, batch_size=batch_size)

    train(train_dataloader, model, loss_fn, optimizer)

    test(test_dataloader, model, loss_fn)
print("Done!")

# plot the dataecurse=True
plt.figure()
plt.plot(range(epochs), l_test, color="r", alpha= 1.0, label= "Plt_Test")
plt.plot(range(epochs), l_train, color="b", alpha= 1.0, label= "Plt_Train")
# plt.yscale('log')

# fig, ax = plt.subplots()

#x_1 = [x[0] for x in weight_1]
#y_1 = [y[1] for y in weight_1]
#ax.scatter(x_1,y_1, c='tab:blue', label='L_1')

#ax.legend()

plt.show()