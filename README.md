# Solving PDE by PINNs

By Jiaqi Yao, School of Computer and Mathematics Science, University of Adeaide

Lte's take a look at how to solve PDE by deep learning! Start by load some packages. The most important package in this file is Pytorch, which is also the famous package for deep learning.

## Our Problem

Our Problem is:

$$
u_{tt} - u_{xx} = 0.
$$

With IC & BC:

$$
u(x,0) = \sin(\frac{3\pi}{2e}x), u(x,\frac{4\pi}{5}) = \sin(\frac{3\pi}{2e}x)*\cos(\frac{6\pi}{5})
$$

and 

$$
u(0,t) = u_x(e,t) = 0
$$


It can be verfied that this problem has a analytic solution as $u(x,t) = a(x,t) + b(x,t)$, where, 

$$a(x,t) = \cos(\frac{3\pi t}{2e})*\sin(\frac{3\pi x}{2e})$$ 

And,

$$b(x,t)= \frac{2e}{5\pi}\sin(\frac{5\pi t}{2e})*\sin(\frac{5\pi x}{2e})$$


Now let's explore how to find the same solution by neural network!

## Solve the Problem by PINNs

```Python
import numpy as np
import torch              
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

Then, set some hyperparameters for the neural network.

```python
epochs = 10000    # Number of training iterations
h = 100    # Grid density for plotting
N = 1000    # Number of interior points
N1 = 100    # Number of boundary points

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Set the set so the each time we train the model it starts at the same initial stage.
setup_seed(777)
```

Set our sampling method according to the equation and the initial & boundary conditions.

```python

# Domain and Sampling
def interior(n=N):
    # interior point
    t = (torch.rand(n, 1)*torch.e*4/5)
    x = (torch.rand(n, 1)*torch.e)
    cond = (torch.zeros_like(x))
    return t.requires_grad_(True), x.requires_grad_(True), cond

def up(n=N1):
    # boundary condition at t=4e/5
    x = (torch.rand(n, 1)*torch.e)
    t = (torch.ones_like(x)*torch.e*4/5)
    cond = torch.sin(3*torch.pi/(2*torch.e)*x)*torch.cos(torch.tensor(6*torch.pi/5))
    return x.requires_grad_(True), t.requires_grad_(True), cond


def down(n=N1):
    # boundary condition at t=0
    x = (torch.rand(n, 1)*torch.e)
    t = (torch.zeros_like(x))
    cond = torch.sin(3*torch.pi/(2*torch.e)*x)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def left(n=N1):
    # boundary condition at x=0
    t = (torch.rand(n, 1)*torch.e*4/5)
    x = (torch.zeros_like(t))
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), t.requires_grad_(True), cond


def right_x(n=N1):
    # boundary condition at x=e
    t = (torch.rand(n, 1)*torch.e*4/5)
    x = (torch.ones_like(t)*torch.e)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), t.requires_grad_(True), cond
```


Set up the neural network model.

```python
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

Construct a function to calculate the derivative in order to train the model based on conditions.

```python
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
```

Now let's start training our model!

```python
# Training
Model = MLP()
optimizer = torch.optim.Adam(params=Model.parameters())
criterion = torch.nn.MSELoss()
```

We also need to define our global loss function, which is the sum of error of interior points as well as all the ICs & BCs.

```
# The following 7 losses are PDE losses
def l_interior(u):
    # Loss function L1
    x, t, cond = interior()
    uxt = u(torch.cat([x, t], dim=1))
    return criterion(gradients(uxt, t, 2) - gradients(uxt, x, 2), cond)

def l_up(u):
    # Loss function L2
    x, t, cond = up_t()
    uxt = u(torch.cat([x, t], dim=1))
    return criterion(uxt, cond)

def l_down(u):
    # Loss function L3
    x, t, cond = down()
    uxt = u(torch.cat([x, t], dim=1))
    return criterion(uxt, cond)

def l_left(u):
    # Loss function L4
    x, t, cond = left()
    uxt = u(torch.cat([x, t], dim=1))
    return criterion(uxt, cond)

def l_right_x(u):
    # Loss function L5
    x, t, cond = right_x()
    uxt = u(torch.cat([x, t], dim=1))
    return criterion(gradients(uxt, x, 1), cond)
```

Train the model based on forward process and backward process. Record the loss value in each iteration.

```python
loses = []


for i in range(epochs):
    optimizer.zero_grad()
    loss = l_interior(Model) + l_up(Model) + l_down(Model) + l_left(Model) + l_right_x(Model) 
    loss.backward()
    optimizer.step()
    
    loses.append(loss.item())
```

We can see that the loss value steadly decreases.

```
loses
```

## Plot the Result 

Compare our result with and true solution and visualize them.

```python
# Inference
xc = torch.linspace(0, torch.e, h)
tc = torch.linspace(0, torch.e*4/5, h)

xm, tm = torch.meshgrid(xc,tc)

xx = xm.reshape(-1, 1)
tt = tm.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)
u_pred = u(xt)
u_real = (torch.cos(3*torch.pi/(2*torch.e)*tt)*torch.sin(3*torch.pi/(2*torch.e)*xx) + (2*torch.e)/(5*torch.pi)*torch.sin(5*torch.pi/(2*torch.e)*tt)*torch.sin(5*torch.pi/(2*torch.e)*xx))
u_error = torch.abs(u_pred-u_real)
u_pred_fig = u_pred.reshape(h,h)
u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h,h)
```

First, plot the result given by PINNs.

```python
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(xm.detach().numpy(), tm.detach().numpy(), u_pred_fig.detach().numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

# Add a color bar
fig.colorbar(surf)

# Display the plot
plt.show()
```

Then, plot the analytic solution.

```python
# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制surface图
surf = ax.plot_surface(xm.detach().numpy(), tm.detach().numpy(), u_real_fig.detach().numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')


# 添加颜色条
fig.colorbar(surf)

# 显示图形
plt.show()
```

At the end, plot the error distribution.

```python
# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制surface图
surf = ax.plot_surface(xm.detach().numpy(), tm.detach().numpy(), u_error_fig.detach().numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')


# 添加颜色条
fig.colorbar(surf)

# 显示图形
plt.show()
```

Overall, we got good results!
