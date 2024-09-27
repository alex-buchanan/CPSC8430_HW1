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

batch_size = 32

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
principal = PCA(n_components=2)
principal_2 = PCA(n_components=2)
principal_3 = PCA(n_components=2)
principal_4 = PCA(n_components=2)

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

model_1 = NeuralNetwork(32, 16, 10).to(device)
model_2 = NeuralNetwork(32, 64, 10).to(device)
model_3 = NeuralNetwork(32, 256, 10).to(device)
model_4 = NeuralNetwork(32, 1024, 10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=1e-3)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=1e-3)
optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=1e-3)
optimizer_4 = torch.optim.SGD(model_4.parameters(), lr=1e-3)

loss_train_1 = []
loss_test_1  = []
accy_train_1 = []
accy_test_1  = []

loss_train_2 = []
loss_test_2  = []
accy_train_2 = []
accy_test_2  = []

loss_train_3 = []
loss_test_3  = []
accy_train_3 = []
accy_test_3  = []

loss_train_4 = []
loss_test_4  = []
accy_train_4 = []
accy_test_4  = []

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

epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print(f"Train model 1, dim 16: ")
    train(train_dataloader, model_1, loss_fn, optimizer_1, loss_train_1, accy_train_1)
    test(test_dataloader, model_1, loss_fn, loss_test_1, accy_test_1)

    print(f"Train model 2, dim 64: ")
    train(train_dataloader, model_2, loss_fn, optimizer_2, loss_train_2, accy_train_2)
    test(test_dataloader, model_2, loss_fn, loss_test_2, accy_test_2)

    print(f"Train model 3, dim 256: ")
    train(train_dataloader, model_3, loss_fn, optimizer_3, loss_train_3, accy_train_3)
    test(test_dataloader, model_3, loss_fn, loss_test_3, accy_test_3)

    print(f"Train model 4, dim 1024: ")
    train(train_dataloader, model_4, loss_fn, optimizer_4, loss_train_4, accy_train_4)
    test(test_dataloader, model_4, loss_fn, loss_test_4, accy_test_4)
'''
    ###################################################
    m_1 = model.linear1
    print(f"Model Data Shape: {m_1.weight.data.shape}")
    wData_1 = torch.sum(m_1.weight.data,1)[:,None]
    wData_1 = torch.transpose(wData_1, 0, 1)
    dataTensor_1 = torch.cat((dataTensor_1, torch.Tensor.cpu(wData_1)), 0)
    print(f"wData_1: {wData_1.shape}")
    print(f"wData_1: {wData_1}")
    if (t+1) % 2 == 0:
        principal.fit(torch.Tensor.cpu(dataTensor_1[1:]))
        weight_1.append(np.array(principal.transform(torch.Tensor.cpu(dataTensor_1[1:]))))
        print(f"Weight Values: {weight_1}")
        del dataTensor_1
        dataTensor_1 = torch.empty((1, 32), dtype=torch.float32).fill_(0)
    ###################################################
    m_2 = model.linear2
    wData_2 = torch.sum(m_2.weight.data,1)[:, None]
    wData_2 = torch.transpose(wData_2, 0, 1)
    dataTensor_2 = torch.cat((dataTensor_2, torch.Tensor.cpu(wData_2)), 0)
    print(f"wData_2 Shape: {dataTensor_2.shape}")
    if (t+1) % 2 == 0:
        principal_2.fit(torch.Tensor.cpu(dataTensor_2[1:]))
        weight_2.append(np.array(principal_2.transform(torch.Tensor.cpu(dataTensor_2[1:]))))
        del dataTensor_2
        dataTensor_2 = torch.empty((1, 32), dtype=torch.float32).fill_(0)
    ###################################################
    m_3 = model.linear3
    wData_3 = torch.sum(m_3.weight.data,1)[:, None]
    wData_3 = torch.transpose(wData_3, 0, 1)
    dataTensor_3 = torch.cat((dataTensor_3, torch.Tensor.cpu(wData_3)), 0)
    print(f"wData_3 Shape: {dataTensor_3.shape}")
    if (t+1) % 2 == 0:
        principal_3.fit(torch.Tensor.cpu(dataTensor_3[1:]))
        weight_3.append(np.array(principal_3.transform(torch.Tensor.cpu(dataTensor_3[1:]))))
        del dataTensor_3
        dataTensor_3 = torch.empty((1, 32), dtype=torch.float32).fill_(0)
    ###################################################
    m_4 = model.linear4
    wData_4 = torch.sum(m_4.weight.data,1)[:, None]
    wData_4 = torch.transpose(wData_4, 0, 1)
    dataTensor_4 = torch.cat((dataTensor_4, torch.Tensor.cpu(wData_4)), 0)
    print(f"wData_4 Shape: {dataTensor_4.shape}")
    if (t+1) % 2 == 0:
        principal_4.fit(torch.Tensor.cpu(dataTensor_4[1:]))
        weight_4.append(np.array(principal_4.transform(torch.Tensor.cpu(dataTensor_4[1:]))))
        del dataTensor_4
        dataTensor_4 = torch.empty((1, 10), dtype=torch.float32).fill_(0)
    ###################################################
'''
    # test(test_dataloader, model_1, loss_fn, loss_test_1, accy_test_1)

print("Done!")

# plot the data
# plt.figure()
# plt.plot(epoch, l_1, color="r", alpha= 1.0, label= "Plt_1")
# plt.yscale('log')
fig = plt.figure()

# training & testing losses
ax_1 = plt.subplot(1,2,1)
ax_1.set_xlabel('Epoch')
ax_1.set_ylabel('Loss')

ax_1.plot(range(epochs), loss_test_1, 'r', label= "Loss_Test_1")
ax_1.plot(range(epochs), loss_train_1, 'r--', label= "Loss_Train_1")

ax_1.plot(range(epochs), loss_test_2, 'b', label= "Loss_Test_2")
ax_1.plot(range(epochs), loss_train_2, 'b--', label= "Loss_Train_2")

ax_1.plot(range(epochs), loss_test_3, 'g', label= "Loss_Test_3")
ax_1.plot(range(epochs), loss_train_3, 'g--', label= "Loss_Train_3")

ax_1.plot(range(epochs), loss_test_4, 'y', label= "Loss_Test_4")
ax_1.plot(range(epochs), loss_train_4, 'y--', label= "Loss_Train_4")

ax_1.legend()

ax_2 = plt.subplot(1,2,2)
ax_2.set_xlabel('Epoch')
ax_2.set_ylabel('Accuracy')

ax_2.plot(range(epochs), accy_test_1, 'r', label= "Accy_Test_1")
ax_2.plot(range(epochs), accy_train_1, 'r--', label= "Accy_Train_1")

ax_2.plot(range(epochs), accy_test_2, 'b', label= "Accy_Test_2")
ax_2.plot(range(epochs), accy_train_2, 'b--', label= "Accy_Train_2")

ax_2.plot(range(epochs), accy_test_3, 'g', label= "Accy_Test_3")
ax_2.plot(range(epochs), accy_train_3, 'g--', label= "Accy_Train_3")

ax_2.plot(range(epochs), accy_test_4, 'y', label= "Accy_Test_4")
ax_2.plot(range(epochs), accy_train_4, 'y--', label= "Accy_Train_4")

ax_2.legend()

fig.suptitle('Loss & Accuracy')

plt.show()