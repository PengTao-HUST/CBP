## Chaotic Back-propagation (CBP)
cbpy is a light package for research purpose, which implements the CBP algorithm in the paper [Brain-inspired Chaotic Back-Propagation for MLP](https://www.sciencedirect.com/science/article/abs/pii/S0893608022003045) published in Neural Networks. You can download the full text on [ResearchGate](https://www.researchgate.net/publication/362618836_Brain-inspired_chaotic_backpropagation_for_MLP).

### Install
```bash
pip install cbpy
```

### Examples
#### Example1. reveal the principles of Chaotic Back-propagation (CBP) with single-neuron network.
###### 1. prepare the dataset
```python
import torch
import torch.nn as nn
import cbpy as cbp

inp = torch.FloatTensor([[1]]) # input sample
tgt = torch.FloatTensor([[0]]) # target of input
```
###### 2. define single-neuron network
```python
class SingleNet(nn.Module):
    def __init__(self, init_value=0):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.act_func = nn.Sigmoid()
        nn.init.constant_(self.layer.weight, init_value)
        nn.init.constant_(self.layer.bias, init_value)
        
    def forward(self, x):
        out = self.act_func(self.layer(x))
        return out
```
###### 3. training with BP algorithm
```python
net = SingleNet()
loss_func = nn.MSELoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=1)

loss_list = []
weight_list = []
bias_list = []
for i in range(1000):
    optimizer.zero_grad()
    out = net(inp)
    loss_bp = loss_func(out, tgt)
    loss_bp.backward()
    optimizer.step()

    loss_list.append(loss_bp.item())
    weight_list.append(net.layer.weight.item())
    bias_list.append(net.layer.bias.item())
```
###### 4. plot the learning curve of the weight for BP
```python
import seaborn as sns
sns.set(context='notebook', style='whitegrid', font_scale=1.2)

cbp.plot_series(weight_list, ylabel='w', title='weight of BP')
```
![bp_weight](https://github.com/PengTao-HUST/CBP/blob/master/figures/bp_weight.png?raw=true)

###### 5. training with CBP algorithm
```python
# define the chaotic loss function
def chaos_loss(out, z, I0=0.65):
    return -z * (I0 * torch.log(out) + (1 - I0) * torch.log(1 - out))

# training with CBP
net = SingleNet()
optimizer = torch.optim.SGD(net.parameters(), lr=1)

z = 9 # initial chaotic intensity
beta = 0.999 # anealing constant

loss_bp_list = []
loss_cbp_list = []
weight_list = []
bias_list = []
for i in range(1000):
    optimizer.zero_grad()
    out = net(inp)
    loss_bp = loss_func(out, tgt)
    loss_chaos = chaos_loss(out, z) # chaotic loss
    loss_cbp = loss_bp + loss_chaos # loss of CBP
    loss_cbp.backward()
    optimizer.step()
    z *= beta
    
    loss_bp_list.append(loss_bp.item())
    loss_cbp_list.append(loss_cbp.item())
    weight_list.append(net.layer.weight.item())
    bias_list.append(net.layer.bias.item())
``` 
###### 6. plot the learning curve of the weight for CBP
```python
cbp.plot_series(weight_list, ylabel='w', title='weight of CBP')
```
![cbp_weight](https://github.com/PengTao-HUST/CBP/blob/master/figures/cbp_weight.png?raw=true)

#### Example2. validate the global optimization ability of CBP on the XOR problem
###### 1. prepare the dataset and parameters
```python
# create the XOR dataset
trainloader = cbp.create_xor_dataloader()

inp, tgt = next(iter(trainloader))
print(inp, '\n', tgt)

# define params
loss_func = torch.nn.BCELoss() # loss function
lr = 0.2 # learning rate
max_epoch = 10000 # maximal training epoch
seed = 32 # random number seed
init_mode = 1 # initial weight interval
layer_list = [2, 2, 1] # layers for MLP
```

###### 2. training by BP
```python
cbp.set_random_seed(seed)
model = cbp.MLPS(layer_list, init_mode=init_mode, 
                 act_layer=torch.nn.Sigmoid(), active_last=True)

zs = None # chaotic intensity
cbp_epoch = 0
bp_l_list, bp_a_list, bp_w_list, bp_o_list = cbp.train_with_chaos(
    model=model,
    trainloader=trainloader,
    testloader=trainloader,
    loss_func=loss_func,
    zs=zs,
    record_weight=True,
    whole_weight=True,
    cbp_epoch=cbp_epoch,
    max_epoch=max_epoch,
    bp_lr=lr
)
```

###### 3. plot the trajectories of the weights for BP (first 2000 epochs)
```python
cbp.plot_xor_weight(bp_w_list[:2000])
```
Weights of BP

![xor_bp_weight](https://github.com/PengTao-HUST/CBP/blob/master/figures/bp_xor_weight_example.png?raw=true)

###### 4. training by BP from the same initial condition as CBP
```python
cbp.set_random_seed(seed)
model = cbp.MLPS(layer_list, init_mode=init_mode, 
                 act_layer=torch.nn.Sigmoid(), active_last=True)
zs = 12 # chaotic intensity
beta = 0.999 # annealing constant
cbp_epoch = max_epoch
cbp_l_list, cbp_a_list, cbp_w_list, cbp_o_list = cbp.train_with_chaos(
    model=model,
    trainloader=trainloader,
    testloader=trainloader,
    loss_func=loss_func,
    zs=zs,
    beta=beta,
    record_weight=True,
    whole_weight=True,
    cbp_epoch=cbp_epoch,
    max_epoch=max_epoch,
    cbp_lr=lr
)
```

###### 5. plot the trajectories of the weights in CBP (first 2000 epochs)
```
cbp.plot_xor_weight(cbp_w_list[:2000], suptitle='CBP')
```
Weights of CBP

![xor_bp_weight](https://github.com/PengTao-HUST/CBP/blob/master/figures/cbp_xor_weight_example.png?raw=true)

###### 6. compare the loss and accuracy of BP and CBP
```python
import numpy as np

loss_mat = np.array([bp_l_list, cbp_l_list]).T
acc_mat = np.array([bp_a_list, cbp_a_list]).T
cbp.plot_mul_loss_acc(loss_mat, acc_mat, alpha=1, ylabels=['loss', 'acc'])
```
![compare_loss](https://github.com/PengTao-HUST/CBP/blob/master/figures/comp_bp_cbp_loss_acc.png?raw=true)

#### Example3. choose the parameter z (initial chaotic intensity)
```python
cbp.set_random_seed(seed)
model = cbp.MLPS(layer_list, init_mode=init_mode, act_layer=torch.nn.Sigmoid(), active_last=True)

ws_lists = cbp.debug_chaos(model, trainloader, loss_func=loss_func)
le_list = cbp.cal_lyapunov_exponent(ws_lists)
cbp.plot_lyapunov_exponent_with_z(le_list)
```
![Lyapunov_exponent](https://github.com/PengTao-HUST/CBP/blob/master/figures/xor_w1_lyapunov_exponent_with_z.png?raw=true)

In this example, the Lyapunov exponent around interval [8, 11] is positive, which indicates chaotic dynamics.  
Then, z = 12 was chosen as the initial chaotic intensity.

### Reproduce the results in the paper
To reproduce the results in the paper, check the notebook files in [paper_example](https://github.com/PengTao-HUST/CBP/tree/master/paper_example).

### Components
- chaos_optim.py: implement the CBP algorithm in the form of optimizer.
- net.py: contains the neural network class, currently only support MLP.
- train.py: provide APIs to preform training with the Pytorch style.
- dataset.py: provide APIs to create the trainloader and testloader.
- utils.py: several auxiliary functions to analysis the training results.
- plot.py: several functions to show the training results.

### License
MIT License