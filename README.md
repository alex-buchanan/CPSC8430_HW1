# CPSC8430_HW1
## 1 - Deep vs Shallow
### 1-1 Simulate a Function
![](./HW1-1.png)
### 1-2 Train on Tasks
![](./HW1-1.png)
### 1-1 Report Questions
#### Simulate a Function:
- Used 4 linear models with a rectified linear unit function as the activation function.  Functions were composed of the same number of parameters, but the depth of the models was varied from shallow to deep with the last model having 8 layers of 15 parameters per layer. 
- Results clearly showed that a balance between the number of layers (4) and parameters (30) per layer was necessary for the most accurate results after 1000 training cycles or epochs.
#### Train on Actual Tasks:
- MNIST training data was used to train a 2 layer convolution with different kernel sizes, and 2 linear layers of varying size depending on the model used.
- Model showed that after 10 epochs, the model with the most linear parameters over 2 linear layers was the best at both converging, and final accuracy.  This application differs slightly from the previous task in that each training sample has more different, and less smooth data that needs to be matched.  It seems that the addition of extra parameters to account for this is necessary when training the model and getting it to converge.
## 2 - Optimization
### 2-1 Visualize the Optimization Process
![](./HW2-1.png)
### 2-2 Observe Gradient Norm
![](./HW2-2.png)
### 2-3 Compute Minimal Ratio
# N/A
### 1-2 Report Questions
#### Visualize the Optimization Process
- Experiment Settings
* MNIST Data Set
* 2 Convolution Layers, followed by 4 linear layers with 32 parameters each
* Utilized the rectified linear unit function as the activation function
* Used the Cross Entropy Loss Function
* Weights in each layer were summed across each epoch training iterations, then fed into primary component analysis to reduce the dataset to a 2 component vector that could then be plotted for each epoch.
- Results were not as expected.  There was much difficulty in trying to determine how the professor wanted us to reduce and analyze the data.  
#### Observe Gradient Norm During Training
- Gradient norm and model loss reduced as expected.  The gradient norm of the loss appears to level out and reach an asymptote after a large number of training cycles.
#### What Happens When Gradient is almost Zero?
- I was unable to calculated the Hessian of the Loss of the Parameters as the gradient norm of the loss reached a minimum.  I assume this is what the professor was asking for, however the results I got were not in line with the example plots shown.
## 3 - Generalization
### 3-1 Label Randomization
![](./HW3-1.png)
### 3-2 Number of Parameters
![](./HW3-2.png)
### 3-3 Flatness v.s. Generalization
#### 1 - Loss, Accuracy, Cross Entropy
![](./HW3-3-1.png)
#### 2 - Sensitivity
![](./Accuracy_Batch.png)
### 1-3 Report Questions
#### Can A Network Fit Random Labels
#### Number of Parameters vs. Generalization
#### Flatness vs Generalization - Part 1
#### Flatness vs Generalization - Part 2
