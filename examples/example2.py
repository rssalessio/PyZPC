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



A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.ones((5, 1))
C = np.array([1, 0, 0, 0, 0])
D = np.array([0])

dim_x = A.shape[0]
dim_u = 1
dt = 0.05
A,B,C,D,_ = scipysig.cont2discrete(system=(A,B,C,D), dt = dt)

uref = np.array([8])
xref = (np.linalg.inv(np.eye(dim_x) - A) @ B @ uref).flatten()


# Define the loss function
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.array([[1,0,0,0]]*horizon)

    # Sum_t ||y_t - r_t||^2
    cost = 0
    for i in range(horizon):
        cost += 1e3 * cp.norm(y[i,:] - xref, p =2) + 1e-3 * cp.norm(u[i,:] - uref,p=2)
    return  cost

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return []


# Define zonotopes and generate data
X0 = Zonotope([-2, 4, 3, -2.5, 5.5], 1 * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u,  10 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 1)))
V = Zonotope([0] * dim_x, 0.002 * np.ones((dim_x, 1)))
Y = Zonotope([1] * dim_x, 15*np.diag(np.ones(dim_x)))
AV = V * A
zonotopes = SystemZonotopes(X0, U, Y, W, V, AV)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon = 2

data = generate_trajectories(scipysig.StateSpace(A,B,C,D), X0, U, W, V, num_trajectories, num_steps_per_trajectory)

# Build DPC
zpc = ZPC(data)

x = X0.sample().flatten()

trajectory = [x]
problem = zpc.build_problem(zonotopes, horizon, loss_callback, constraints_callback)
for n in range(100):
    print(f'Solving step {n}')

    result, info = zpc.solve(x, verbose=True,warm_start=True)
    u = info['u_optimal']
    z = A @ x +  np.squeeze(B *u[0]) + W.sample()

    # We assume C = I
    x = (z + V.sample()).flatten()
    trajectory.append(x)

trajectory = np.array(trajectory)

for i in range(dim_x):
    plt.plot(trajectory[:,i], label=f'x{i}')

plt.grid()
plt.legend()
plt.show()
