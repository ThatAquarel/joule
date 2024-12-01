import numpy as np
from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import *


class Surface:
    def __init__(self, res=1024):
        self._point_mesh = self._build_point_mesh(res)
        self._mesh_index = self._build_indices(res)

        self._n = len(self._mesh_index)
        self._data = np.ones((self._n, 9), dtype=np.float32)
        self._vao, self._vbo = create_vbo(
            self._data,
            return_vbo=True,
            store_normals=True,
        )

        self.ready = False

    def _build_point_mesh(self, res):
        mesh = np.mgrid[0 : 1 : res * 1j, 0 : 1 : res * 1j].T
        samples = mesh.reshape((-1, 2))

        return samples

    def _build_indices(self, res):
        idx = np.empty((res - 1, 3 * res - 2), dtype=np.int32)
        for i in range(2):
            i_iter = np.arange(i, res - 1 + i)
            j_iter = np.arange(res)
            idx[:, i : 2 * res : 2] = i_iter[:, np.newaxis] * res + j_iter

        i_iter = np.arange(1, res)
        j_iter = np.arange(res - 2, 0, -1)
        idx[:, -(res - 2) :] = i_iter[:, np.newaxis] * res + j_iter

        return idx.flatten()

    def _point_mesh_scale(self, x_range, y_range):
        ranges = [x_range, y_range]

        intervals = np.subtract.reduce(ranges, axis=1)
        intervals = np.abs(intervals)
        minimums = np.min(ranges, axis=1)

        return lambda vec: vec * intervals + minimums

    def get_point_mesh(self, x_range, y_range):
        scale = self._point_mesh_scale(x_range, y_range)
        point_mesh = scale(self._point_mesh)
        return point_mesh

    def update_function(
        self,
        scaled_mesh,
        values,
        normals,
    ):
        self._data[:, :2] = scaled_mesh[self._mesh_index]
        self._data[:, 2] = values[self._mesh_index]
        self._data[:, -3:] = normals[self._mesh_index]

        update_vbo(self._vbo, self._data)
        self.ready = True

    def draw(self):
        if not self.ready:
            return

        draw_vbo(self._vao, GL_TRIANGLE_STRIP, self._n)
