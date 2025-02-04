import numpy as np

dim = 5
x = np.ones((6, 4, 4))
tiled = np.tile(x, (dim, dim))
print(tiled.shape)
