import numpy as np
from OpenGL.GL import *

from joule.graphics.vbo import create_vao, draw_vao, update_vbo


class Surface:
    def __init__(self, initial_color, res=1024):
        """
        Surface: Surface render element for function plotting

        :param initial_color: Surface initial color
        :param res: Evaluation points per axis (total points of res*res)
        """

        # prebuffer mesh and indices
        self._point_mesh = self._build_point_mesh(res)
        self._mesh_index = self._build_indices(res)

        # preallocate data array with color
        self._n = len(self._mesh_index)
        self._data = np.ones((self._n, 9), dtype=np.float32)

        self._data[:, 3:6] = initial_color

        # build VAO
        self._vao, self._vbo = create_vao(
            self._data,
            return_vbo=True,
            store_normals=True,
        )

        self.ready = False

    def _build_point_mesh(self, res):
        """
        Build unit grid of points with resolution

        :param res: Resolution at which to sample
        :return: Mesh of points of shape (n, 2)
        """

        # mesh of points
        mesh = np.mgrid[0 : 1 : res * 1j, 0 : 1 : res * 1j].T
        samples = mesh.reshape((-1, 2))

        # return points of shape (n, 2)
        return samples

    def _build_indices(self, res):
        """
        Build draw indices for OpenGL triangle draw

        :param res: Resolution at which to sample
        :return: OpenGL triangle draw indices of shape (n,)
        """

        # allocate empty array for indices
        # I can explain this part in person, it'll probably be easier
        idx = np.empty((res - 1, 3 * res - 2), dtype=np.int32)
        for i in range(2):
            i_iter = np.arange(i, res - 1 + i)
            j_iter = np.arange(res)
            idx[:, i : 2 * res : 2] = i_iter[:, np.newaxis] * res + j_iter

        # numpy magic
        i_iter = np.arange(1, res)
        j_iter = np.arange(res - 2, 0, -1)
        idx[:, -(res - 2) :] = i_iter[:, np.newaxis] * res + j_iter

        return idx.flatten()

    def _point_mesh_scale(self, x_range, y_range):
        """
        Build a lambda function to scale a unit grid
        to a range

        :param x_range: New range of x values
        :param y_range: New range of y values

        :return: points scaling lambda
        """

        ranges = [x_range, y_range]

        # find the delta of the interval
        intervals = np.subtract.reduce(ranges, axis=1)
        intervals = np.abs(intervals)

        # find the minimum of the interval
        minimums = np.min(ranges, axis=1)

        return lambda vec: vec * intervals + minimums

    def get_point_mesh(self, x_range, y_range):
        """
        Build a scaled grid of points

        :param x_range: range of x values
        :param y_range: range of y values
        :return: Mesh of points of shape (n, 2)
        """

        # scale buffered unit point mesh to new range
        scale = self._point_mesh_scale(x_range, y_range)
        point_mesh = scale(self._point_mesh)

        # return point mesh
        return point_mesh

    def set_color(self, new_color):
        """
        Update surface' color

        :param new_color: New surface color
        """

        # change default buffer's color, so that subsequent
        # updates in self.update_function take this new color
        self._data[:, 3:6] = new_color
        update_vbo(self._vbo, self._data)

    def update_function(
        self,
        scaled_mesh,
        values,
        normals,
    ):
        """
        Update sampled points buffer

        :param scaled_mesh: Mesh of points of shape (n, 2)
        :param values: Array of f(x, y) function evaluations at points of shape (n,)
        :param normals: Array of normals evaluted at points of shape (n, 3)
        """

        # copy new function data into buffers
        self._data[:, :2] = scaled_mesh[self._mesh_index]
        self._data[:, 2] = values[self._mesh_index]
        self._data[:, -3:] = normals[self._mesh_index]

        update_vbo(self._vbo, self._data)
        self.ready = True

    def draw(self):
        """
        Draw surface
        """

        if not self.ready:
            return

        draw_vao(self._vao, GL_TRIANGLE_STRIP, self._n)
