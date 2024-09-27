import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

##################################################
## ground-truth model
##################################################

def goalFun(x):
    return(np.sin(5*np.pi*x)/(5*np.pi*x))

# create linear sequence (x) and apply goalFun (y)
x = np.linspace(start = 0, stop = 1.0, num = 1000)
y = goalFun(x)

# plot the function
# d = pd.DataFrame({'x' : x, 'y' : y})
# sns.lineplot(data = d, x = 'x', y = 'y')
# plt.show()

##################################################
## generate training data (with noise)
##################################################

nObs = 500 # number of observations

# get noise around y observations
yNormal = torch.distributions.Normal(loc=0.0, scale=10)
yNoise  = yNormal.sample([nObs])

# get observations
xObs = 1*torch.rand([nObs])    # uniform from [-5,5]
# xObs = torch.ones([nObs])
yObs = np.sin(5*np.pi*xObs)/(5*np.pi*xObs) # + yNoise

# plot the data
# d = pd.DataFrame({'xObs' : xObs, 'yObs' : yObs})
# sns.scatterplot(data = d, x = 'xObs', y = 'yObs')
# plt.show()

##################################################
## network 1 dimension parameters
##################################################

nInput  = 1
nHidden = 60
nOutput = 1

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- version 1 --
##################################################

class MLPexplicit_1(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit_1, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.linear1 = nn.Linear(self.nInput,  self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nOutput)
        # self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear4 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear5 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear6 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear7 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear8 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        # h2 = self.ReLU(self.linear2(h1))
        # h3 = self.ReLU(self.linear3(h2))
        # h4 = self.ReLU(self.linear4(h3))
        # h5 = self.ReLU(self.linear5(h4))
        # h6 = self.ReLU(self.linear6(h5))
        # h7 = self.ReLU(self.linear7(h5))
        output = self.linear2(h1) ## <- NOTE
        return(output)

class MLPexplicit_2(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit_2, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
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
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        # h4 = self.ReLU(self.linear4(h3))
        # h5 = self.ReLU(self.linear5(h4))
        # h6 = self.ReLU(self.linear6(h5))
        # h7 = self.ReLU(self.linear7(h5))
        output = self.linear4(h3) ## <- NOTE
        return(output)

mlpExplicit_1 = MLPexplicit_1(nInput, nHidden, nOutput)
mlpExplicit_2 = MLPexplicit_2(nInput, 30, nOutput)

# which model to use from here onwards
model_1 = mlpExplicit_1.to(device)
model_2 = mlpExplicit_2.to(device)

##################################################
## network 2 dimension parameters
##################################################

nInput  = 1
nHidden = 30
nOutput = 1

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- version 1 --
##################################################

class MLPexplicit_2(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit_2, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
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
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        # h4 = self.ReLU(self.linear4(h3))
        # h5 = self.ReLU(self.linear5(h4))
        # h6 = self.ReLU(self.linear6(h5))
        # h7 = self.ReLU(self.linear7(h5))
        output = self.linear4(h3) ## <- NOTE
        return(output)

mlpExplicit_2 = MLPexplicit_2(nInput, nHidden, nOutput)
# which model to use from here onwards
model_2 = mlpExplicit_2.to(device)

##################################################
## network 1 dimension parameters
##################################################

nInput  = 1
nHidden = 20
nOutput = 1

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- version 1 --
##################################################

class MLPexplicit_3(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit_3, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.linear1 = nn.Linear(self.nInput,  self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nHidden)
        self.linear5 = nn.Linear(self.nHidden, self.nHidden)
        self.linear6 = nn.Linear(self.nHidden, self.nOutput)
        # self.linear7 = nn.Linear(self.nHidden, self.nHidden)
        # self.linear8 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        h4 = self.ReLU(self.linear4(h3))
        h5 = self.ReLU(self.linear5(h4))
        # h6 = self.ReLU(self.linear6(h5))
        # h7 = self.ReLU(self.linear7(h5))
        output = self.linear6(h5) ## <- NOTE
        return(output)

mlpExplicit_3 = MLPexplicit_3(nInput, nHidden, nOutput)
# which model to use from here onwards
model_3 = mlpExplicit_3.to(device)


##################################################
## network 1 dimension parameters
##################################################

nInput  = 1
nHidden = 15
nOutput = 1

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- version 1 --
##################################################

class MLPexplicit_4(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit_4, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.linear1 = nn.Linear(self.nInput,  self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nHidden)
        self.linear5 = nn.Linear(self.nHidden, self.nHidden)
        self.linear6 = nn.Linear(self.nHidden, self.nHidden)
        self.linear7 = nn.Linear(self.nHidden, self.nHidden)
        self.linear8 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        h4 = self.ReLU(self.linear4(h3))
        h5 = self.ReLU(self.linear5(h4))
        h6 = self.ReLU(self.linear6(h5))
        h7 = self.ReLU(self.linear7(h5))
        output = self.linear8(h7) ## <- NOTE
        return(output)

mlpExplicit_4 = MLPexplicit_4(nInput, nHidden, nOutput)
# which model to use from here onwards
model_4 = mlpExplicit_4.to(device)
##################################################
## representing train data as a Dataset object
##################################################

class nonLinearRegressionData(Dataset):
    '''
    Custom 'Dataset' object for our regression data.
    Must implement these functions: __init__, __len__, and __getitem__.
    '''

    def __init__(self, xObs, yObs):
        self.xObs = torch.reshape(xObs, (len(xObs), 1))
        self.yObs = torch.reshape(yObs, (len(yObs), 1))

    def __len__(self):
        return(len(self.xObs))

    def __getitem__(self, idx):
        return(xObs[idx], yObs[idx])

# instantiate Dataset object for current training data
d = nonLinearRegressionData(xObs, yObs)

# instantiate DataLoader
#    we use the 4 batches of 25 observations each (full data  has 100 observations)
#    we also shuffle the data
train_dataloader = DataLoader(d, batch_size=25 , shuffle=True)

##################################################
## training the model
##################################################

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-4)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-4)
optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=1e-4)
optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=1e-4)
nTrainSteps = 1000
nViewSteps = 50
nSize = nTrainSteps/nViewSteps
e = []
l_1 = []
l_2 = []
l_3 = []
l_4 = []

# Run the training loop
for epoch in range(0, nTrainSteps):

  # Set current loss value
  current_loss_1 = 0.0
  current_loss_2 = 0.0
  current_loss_3 = 0.0
  current_loss_4 = 0.0

  # Iterate over the DataLoader for training data
  for i, data in enumerate(train_dataloader, 0):
    # Get inputs
    inputs, targets = data[0].to(device), data[1].to(device)
    # Zero the gradients
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    optimizer_3.zero_grad()
    optimizer_4.zero_grad()
    # Perform forward pass (make sure to supply the input in the right way)
    outputs_1 = model_1(torch.reshape(inputs, (len(inputs), 1))).squeeze()
    outputs_2 = model_2(torch.reshape(inputs, (len(inputs), 1))).squeeze()
    outputs_3 = model_3(torch.reshape(inputs, (len(inputs), 1))).squeeze()
    outputs_4 = model_4(torch.reshape(inputs, (len(inputs), 1))).squeeze()
    # Compute loss
    loss_1 = loss_function(outputs_1, targets)
    loss_2 = loss_function(outputs_2, targets)
    loss_3 = loss_function(outputs_3, targets)
    loss_4 = loss_function(outputs_4, targets)
    # Perform backward pass
    loss_1.backward()
    loss_2.backward()
    loss_3.backward()
    loss_4.backward()
    # Perform optimization
    optimizer_1.step()
    optimizer_2.step()
    optimizer_3.step()
    optimizer_4.step()
    # Print statistics
    current_loss_1 += loss_1.item()
    current_loss_2 += loss_2.item()
    current_loss_3 += loss_3.item()
    current_loss_4 += loss_4.item()

  e.append(epoch+1)
  l_1.append(current_loss_1)
  l_2.append(current_loss_2)
  l_3.append(current_loss_3)
  l_4.append(current_loss_4)

  if (epoch + 1) % nViewSteps == 0:
      print('Loss_1 after epoch %5d: %.3f' %
            (epoch + 1, current_loss_1))
      print('Loss_2 after epoch %5d: %.3f' %
            (epoch + 1, current_loss_2))
      print('Loss_3 after epoch %5d: %.3f' %
            (epoch + 1, current_loss_3))
      print('Loss_4 after epoch %5d: %.3f' %
            (epoch + 1, current_loss_4))
      current_loss_1 = 0.0
      current_loss_2 = 0.0
      current_loss_3 = 0.0
      current_loss_4 = 0.0

# Process is complete.
print('Training process has finished.')

yPred_1 = np.array([torch.Tensor.cpu(model_1.forward(torch.tensor([o]).to(device))).detach().numpy() for o in xObs]).flatten()
yPred_2 = np.array([torch.Tensor.cpu(model_2.forward(torch.tensor([o]).to(device))).detach().numpy() for o in xObs]).flatten()
yPred_3 = np.array([torch.Tensor.cpu(model_3.forward(torch.tensor([o]).to(device))).detach().numpy() for o in xObs]).flatten()
yPred_4 = np.array([torch.Tensor.cpu(model_4.forward(torch.tensor([o]).to(device))).detach().numpy() for o in xObs]).flatten()

# plot the data
plt.figure()
plt.plot(e, l_1, color="r", alpha= 1.0, label= "Plt_1")
plt.plot(e, l_2, color="b", alpha= 1.0, label= "Plt_2")
plt.plot(e, l_3, color="y", alpha= 1.0, label= "Plt_3")
plt.plot(e, l_4, color="g", alpha= 1.0, label= "Plt_4")
plt.yscale('log')

plt.figure()
d = pd.DataFrame({'xObs' : xObs.detach().numpy(),
                  'yObs' : yObs.detach().numpy(),
                  'yPred_1': yPred_1,
                  'yPred_2': yPred_2,
                  'yPred_3': yPred_3,
                  'yPred_4': yPred_4})

dWide = pd.melt(d, id_vars = 'xObs', value_vars= ['yObs', 'yPred_1', 'yPred_2', 'yPred_3', 'yPred_4'])
sns.scatterplot(data = dWide, x = 'xObs', y = 'value', hue = 'variable', alpha = 0.7)
x = np.linspace(start = 0, stop = 1, num = 1000)
y = goalFun(x)
plt.plot(x,y, color='g', alpha = 0.5)

plt.show()