import numpy as np


def normalize(vectors):
    vectors = np.copy(vectors)

    norms = np.linalg.norm(vectors, axis=1)
    non_zero = norms != 0
    vectors[non_zero] /= norms[non_zero, np.newaxis]

    return vectors


def magnitude(vectors):
    return np.linalg.norm(vectors, axis=1)


def column_wise(row_vectors):
    return row_vectors[:, np.newaxis]


def vec_cross(a, b):
    return np.cross(a, b, axis=1)


def vec_dot(a, b):
    return np.vecdot(a, b)[:, np.newaxis] * b


def get_basis():
    return np.eye(3)
