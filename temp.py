import os
import numpy as np

k = np.array([[1, 1, 1], [-1, 1, -1], [0, -1, -1]])
filter = np.array([1, 1, 1]).T

print(k)
print(np.where(k[:2, :2] == 1))
