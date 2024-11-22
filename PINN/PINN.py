import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Diffusion coefficient
D = 1.0

# Define Fick's second law as the governing equation
def ficks_second_law(x, C):
    dC_t = dde.grad.jacobian(C, x, i=0)  # ∂C/∂t
    dC_xx = dde.grad.hessian(C, x, component=0)  # ∂²C/∂x²
    return dC_t - D * dC_xx

# Define the spatial and time domain
geom = dde.geometry.Interval(0, 1)  # Spatial domain: x in [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)  # Time domain: t in [0, 1]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Boundary conditions (e.g., Dirichlet BCs where concentration is 0 at the boundaries)
def boundary(x, on_boundary):
    return on_boundary

bc = dde.DirichletBC(geomtime, lambda x: 0, boundary)

# Initial condition (e.g., initial concentration profile)
def initial_condition(x):
    return np.sin(np.pi * x[:, 0:1])  # Example initial condition

ic = dde.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

# Define the data object
data = dde.data.TimePDE(
    geomtime, ficks_second_law, [bc, ic], num_domain=2560, num_boundary=80, num_initial=160
)

# Define the neural network (use PyTorch backend)
net = dde.maps.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")

# Use PyTorch as the backend
model = dde.Model(data, net)

# Compile the model (using PyTorch optimizer)
model.compile("adam", lr=1e-3)

# Train the model
losshistory, train_state = model.train(iterations=5000)


# Save and plot the result
dde.saveplot(losshistory, train_state, issave=True, isplot=True)




