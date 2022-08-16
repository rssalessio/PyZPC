
import numpy as np

g1 = np.array([1,1,0])
g2 = np.array([1,-1,0])

G = np.hstack([g1[:, None], g2[:, None]])
n = 3

P = np.eye(n) - G @ np.linalg.inv(G.T @ G)@ G.T

w = np.array([3, 0,1])
c = np.array([0,0,0])

Z = P @ (w-c)
print(Z)

beta =  np.linalg.inv(G.T @ G)@ G.T @ (w-c)
print(beta)