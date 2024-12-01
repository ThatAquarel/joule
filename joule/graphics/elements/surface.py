import numpy as np
from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import *


class Surface:
    def __init__(self, res=1024):
        self._eval_mesh = self._build_mesh(res)
        self._mesh_index = self._build_indices(res)

        self._n = len(self._mesh_index)
        self._data = np.ones((self._n, 9), dtype=np.float32)
        self._vao, self._vbo = create_vbo(
            self._data,
            return_vbo=True,
            store_normals=True,
        )

        self.ready = False

    def _build_mesh(self, res):
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

    def _build_normals(
        self,
        d_dx,
        d_dy,
        eval_mesh,
        return_derivative_vector=False,
        normalize=True,
    ):
        d_dx_mesh = d_dx(*eval_mesh.T)  #  y constant, xz tan vec
        d_dx_vec = self._partial_derivative_tangent_vector(d_dx_mesh, 2, 0)

        d_dy_mesh = d_dy(*eval_mesh.T)  #  x constant, yz tan vec
        d_dy_vec = self._partial_derivative_tangent_vector(d_dy_mesh, 2, 1)

        normals = np.cross(d_dx_vec, d_dy_vec, axis=1)

        if normalize:
            normals = self._normalize(normals)

        if return_derivative_vector:
            # derivative_vector = self._normalize(d_dx_vec + d_dy_vec)
            derivative_vector = d_dx_mesh, d_dx_vec, d_dy_mesh, d_dy_vec
            # derivative_vector = self._derivative_tangent_vector(
            #     d_dx_vec, d_dy_mesh, normalize=normalize
            # )
            return normals, derivative_vector
        return normals

    def _partial_derivative_tangent_vector(self, d_ds_mesh, opp_idx, adj_idx):
        vec = np.zeros((*d_ds_mesh.shape, 3), dtype=np.float32)
        vec[:, adj_idx] = 1
        vec[:, opp_idx] = d_ds_mesh

        # norms = np.linalg.norm(vec, axis=1)
        # vec = vec / norms[:, np.newaxis]

        # return self._normalize(vec)
        return vec

    def _derivative_tangent_vector(self, d_dx_mesh, d_dy_mesh, normalize=True):
        tangent_vec = np.ones((len(d_dx_mesh), 3))
        tangent_vec[:, 0] = 1 / d_dx_mesh
        tangent_vec[:, 1] = 1 / d_dy_mesh
        tangent_vec[tangent_vec == np.inf] = 0

        if normalize:
            return self._normalize(tangent_vec)
        return tangent_vec

    def _eval_mesh_scale(self, x_range, y_range):
        ranges = [x_range, y_range]

        intervals = np.subtract.reduce(ranges, axis=1)
        intervals = np.abs(intervals)
        minimums = np.min(ranges, axis=1)

        return lambda vec: vec * intervals + minimums

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1)
        (non_zero,) = np.where(norms)
        vectors[non_zero] /= norms[non_zero, np.newaxis]

        return vectors

    def update_function(
        self,
        f_xy,
        d_dx,
        d_dy,
        x_range,
        y_range,
    ):
        scale = self._eval_mesh_scale(x_range, y_range)
        eval_mesh = scale(self._eval_mesh)

        z = f_xy(*eval_mesh.T)
        n = self._build_normals(d_dx, d_dy, eval_mesh)

        self._data[:, 2] = z[self._mesh_index]
        self._data[:, :2] = eval_mesh[self._mesh_index]
        self._data[:, -3:] = n[self._mesh_index]

        update_vbo(self._vbo, self._data)
        self.ready = True

    def draw(self):
        if not self.ready:
            return

        draw_vbo(self._vao, GL_TRIANGLE_STRIP, self._n)
