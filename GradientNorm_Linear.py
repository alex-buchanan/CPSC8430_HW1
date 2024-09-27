import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch.optim import Adam

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# mu_null = torch.zeros(1)
# sigma_null_hat = Variable(torch.ones(1), requires_grad=True)

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
## network 4 dimension parameters
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
        self.linear3 = nn.Linear(self.nHidden, self.nOutput)
        #self.linear4 = nn.Linear(self.nHidden, self.nHidden)
        #self.linear5 = nn.Linear(self.nHidden, self.nHidden)
        #self.linear6 = nn.Linear(self.nHidden, self.nHidden)
        #self.linear7 = nn.Linear(self.nHidden, self.nHidden)
        #self.linear8 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        #h3 = self.ReLU(self.linear3(h2))
        #h4 = self.ReLU(self.linear4(h3))
        #h5 = self.ReLU(self.linear5(h4))
        #h6 = self.ReLU(self.linear6(h5))
        #h7 = self.ReLU(self.linear7(h5))
        output = self.linear3(h2) ## <- NOTE
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
optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=1e-4)
nTrainSteps = 100
nViewSteps = 10
nSize = nTrainSteps/nViewSteps
e = []
l_1 = []
l_2 = []
l_3 = []
l_4 = []
min_ratio = []

grad_plot = []

# opt = Adam([sigma_null_hat], lr=0.01)

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
    optimizer_4.zero_grad()

    # Perform forward pass (make sure to supply the input in the right way)
    outputs_4 = model_4(torch.reshape(inputs, (len(inputs), 1))).squeeze()

    # Compute loss
    loss_4 = loss_function(outputs_4, targets)

    # Perform backward pass
    loss_4.backward()

    # Perform optimization
    optimizer_4.step()

    # Print statistics
    current_loss_4 += loss_4.item()
    l_4.append(current_loss_4)

    grad_total = 0.0
    r = 0.0

    for p in model_4.parameters():
        p.requires_grad_(True)
        # print(f"{p.grad}")
        grad = 0.0
        if p.grad is not None:
            # print(f"Value! {p.grad.shape}")
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_total += grad
    grad_norm=grad_total**0.5
    grad_plot.append(grad_norm)

    '''
    I = torch.func.hessian(loss_function)(outputs_4, targets)
    L = torch.linalg.eigvals(I)
    for m in L:
        if m.real > 0.0001:
            r += 1
    ratio = r/len(np.array(torch.Tensor.cpu(L).detach().numpy()))
    min_ratio.append(ratio)
    '''
    
  current_loss_4 = 0.0

  if (epoch + 1) % nViewSteps == 0:
      print('Loss_4 after epoch %5d: %.3f' %
            (epoch + 1, current_loss_4))
      

# Process is complete.
print('Training process has finished.')

# print(f"Min Ratio: {min_ratio}")

yPred_4 = np.array([torch.Tensor.cpu(model_4.forward(torch.tensor([o]).to(device))).detach().numpy() for o in xObs]).flatten()

# plot the data
fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.plot(range(len(grad_plot)), grad_plot, color="r")
ax1.set_xlabel('Steps')
ax1.set_ylabel('Gradient')

ax2 = plt.subplot(2,1,2)
ax2.plot(range(len(l_4)), l_4, color="b")
ax2.set_xlabel('Steps')
ax2.set_ylabel('Loss')

# sin(5*np.pi*x)/(5*np.pi*x)
fig.suptitle(r"$\ y = \frac{\sin{5*\pi*x}}{5*\pi*x}$")

plt.show()