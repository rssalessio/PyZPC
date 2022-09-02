# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pyzpc import ZPC, Data, SystemZonotopes
from utils import generate_trajectories
from pyzonotope import Zonotope

# Define the loss function
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.array([[1,0,0,0]]*horizon)

    # Sum_t ||y_t - r_t||^2
    cost = 0
    for i in range(1, horizon):
        cost += 100*cp.norm(y[i,0] - 1)
    return  cost

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return []


# Plant
# In this example we consider the three-pulley 
# system analyzed in the original VRFT paper:
# 
# "Virtual reference feedback tuning: 
#      a direct method for the design offeedback controllers"
# -- Campi et al. 2003

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0. * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u, 3 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 1)))
V = Zonotope([0] * dim_x, 0.002 * np.ones((dim_x, 1)))
Y = Zonotope([1] * dim_x, np.diag(2*np.ones(dim_x)))
AV = V * sys.A
zonotopes = SystemZonotopes(X0, U, Y, W, V, AV)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon = 3

data = generate_trajectories(sys, X0, U, W, V, num_trajectories, num_steps_per_trajectory)

# Build DPC
zpc = ZPC(data)

x = X0.sample().flatten()

trajectory = [x]
problem = zpc.build_problem(zonotopes, horizon, loss_callback, constraints_callback)
for n in range(100):
    print(f'Solving step {n}')

    result, info = zpc.solve(x, verbose=True,warm_start=True)
    u = info['u_optimal']
    z = sys.A @ x +  np.squeeze(sys.B *u[0]) + W.sample()

    # We assume C = I
    x = (z + V.sample()).flatten()
    trajectory.append(x)

trajectory = np.array(trajectory)
for i in range(dim_x):
    plt.plot(trajectory[:,i], label=f'x{i}')

plt.grid()
plt.legend()
plt.show()
