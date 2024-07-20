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


It can be verfied that this problem has a analytic solution as $u(x,t) = \cos(\frac{3\pi t}{2e})*\sin(\frac{3\pi x}{2e}) +\frac{2e}{5\pi}\sin(\frac{5\pi t}{2e})*\sin(\frac{5\pi x}{2e})$.


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
