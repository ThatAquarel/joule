import numpy as np
from sympy import *

from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import GL_TRIANGLE_STRIP


class GraphEngine:
    def __init__(self, res=1024):
        self.ready = False

        self.idx = self._build_indices(res)

        self.eval_mesh = self._build_mesh(res)
        self.mesh = self.eval_mesh[self.idx]

        self.n = len(self.idx)

        self.data = np.ones((self.n, 9), dtype=np.float32)
        self.vao, self.vbo = create_vbo(self.data, return_vbo=True, store_normals=True)

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

    def _build_mesh(self, res):
        mesh = np.mgrid[0 : 1 : res * 1j, 0 : 1 : res * 1j].T
        samples = mesh.reshape((-1, 2))

        return samples

    def _build_normals(self, f_xy, x, y, eval_mesh):
        d_dx_mesh = self._d_ds_mesh(f_xy, x, y, x, eval_mesh)  #  y constant, xz tan vec
        d_dx_vec = self._d_ds_vec(d_dx_mesh, 2, 0)

        d_dy_mesh = self._d_ds_mesh(f_xy, x, y, y, eval_mesh)  #  x constant, yz tan vec
        d_dy_vec = self._d_ds_vec(d_dy_mesh, 2, 1)

        normals = np.cross(d_dx_vec, d_dy_vec, axis=1)

        return normals

    def _d_ds_mesh(self, f_xy, x, y, s, eval_mesh):
        d_ds = diff(f_xy, s)
        f_d_ds = lambdify([x, y], d_ds, "numpy")

        z = f_d_ds(*eval_mesh.T)
        # if d_ds.is_constant():
        if len(eval_mesh) != len(z):
            return np.ones(len(eval_mesh)) * z
        return z

    def _d_ds_vec(self, d_ds_mesh, opp_idx, adj_idx):
        vec = np.zeros((*d_ds_mesh.shape, 3), dtype=np.float32)
        vec[:, adj_idx] = 1
        vec[:, opp_idx] = d_ds_mesh

        norms = np.linalg.norm(vec, axis=1)
        vec = vec / norms[:, np.newaxis]

        return vec

    def _build_color(self, res): ...

    def update_function(self, eqn, x_min, x_max, y_min, y_max):
        x, y = symbols("x y")
        f_xy = sympify(eqn)

        scale = lambda v: v * [x_max - x_min, y_max - y_min] + [x_min, y_min]

        eval_mesh = scale(self.eval_mesh)
        f_xy_npy = lambdify([x, y], f_xy, "numpy")

        self.data[:, 2] = f_xy_npy(*eval_mesh.T)[self.idx]
        self.data[:, :2] = scale(self.mesh)
        self.data[:, -3:] = self._build_normals(f_xy, x, y, eval_mesh)[self.idx]

        update_vbo(self.vbo, self.data)
        self.ready = True

    def draw(self):
        if not self.ready:
            return

        draw_vbo(self.vao, GL_TRIANGLE_STRIP, self.n)
