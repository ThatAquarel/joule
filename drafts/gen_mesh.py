import numpy as np

res = 1000

# mesh = np.mgrid[0 : 1 : res * 1j, 0 : 1 : res * 1j]
# points = mesh.T.reshape((-1, 2))

a = np.empty((res - 1, 3 * res - 2), dtype=np.int32)
for i in range(2):
    i_iter = np.arange(i, res - 1 + i)
    j_iter = np.arange(res)

    a[:, i : 2 * res : 2] = i_iter[:, np.newaxis] * res + j_iter

i_iter = np.arange(1, res)
j_iter = np.arange(res - 2, 0, -1)
a[:, -(res - 2) :] = i_iter[:, np.newaxis] * res + j_iter

...

indices = []
for i in range(res - 1):
    offset = i * res
    for j in range(res):
        indices.extend([offset + j, offset + j + res])

    lines = []
    for j in range(1, res - 1):
        lines.append(offset + j + res)
    indices.extend(lines[::-1])


...
