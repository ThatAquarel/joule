import numpy as np


def normalize(vectors, copy=True):
    """
    Normalizes magnitude of array of vectors to one

    :param vectors: Vectors of shape (n, k)
    :param copy: Create new instance of array in memory
    :return: Normalized vectors of shape (n, k)
    """

    if copy:
        vectors = np.copy(vectors)

    # array of magnitudes
    norms = np.linalg.norm(vectors, axis=1)

    # acquire a mask of non-zero magnitudes
    # to prevent zero division
    non_zero = norms != 0

    # normalize vectors with non-zero
    # magnitude
    vectors[non_zero] /= norms[non_zero, np.newaxis]

    return vectors


def magnitude(vectors):
    """
    Computes the magnitude of array of vectors

    :param vectors: Vectors of shape (n, k)
    :return: Magnitudes of vectors of shape (n,)
    """

    # compute magnitudes over array
    return np.linalg.norm(vectors, axis=1)


def column_wise(row_vectors):
    """
    Transform an array of row vectors into an
    array of column vectors

    :param row_vectors: Row vectors of shape (n,)
    :return: Column vectors of shape (n, 1)
    """

    # reshape as column vector
    return row_vectors[:, np.newaxis]


def vec_cross(a, b):
    """
    Vectorized cross product between two array of vectors

    :param a: Array of vectors a of shape (n, k)
    :param b: Array of vectors b of shape (n, k)
    :return: Array of vectors a cross b (n, k)
    """

    # cross product over array
    return np.cross(a, b, axis=1)


def vec_dot(a, b):
    """
    Vectorized dot product between two array of vectors,
    projecting vector a onto vector b

    :param a: Array of vectors a of shape (n, k)
    :param b: Array of vecotrs b of shape (n, k)
    :return: Array of vectors a projected onto b (n, k)
    """

    # scalar dot product over array
    dot = np.vecdot(a, b)[:, np.newaxis]

    # projection of scalar over b
    return dot * b
