import numpy as np
from sympy import *

from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import GL_TRIANGLE_STRIP


class GraphEngine:
    def __init__(self, res=512):
        self.ready = False

        self.mesh = self._build_mesh(res)
        self.n = len(self.mesh)

        self.data = np.ones((self.n, 6), dtype=np.float32)
        self.vao, self.vbo = create_vbo(self.data, return_vbo=True)

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

        return samples[self._build_indices(res)]

    def _build_color(self, res): ...

    def update_function(self, eqn, x_min, x_max, y_min, y_max):
        x, y = symbols("x y")
        f_xy = sympify(eqn)

        d_dx = diff(f_xy, x)
        d_dy = diff(f_xy, y)

        mesh = self.mesh * [x_max - x_min, y_max - y_min] + [x_min, y_min]
        f_xy_npy = lambdify([x, y], f_xy, "numpy")

        self.data[:, 2] = f_xy_npy(*mesh.T)
        self.data[:, :2] = mesh
        update_vbo(self.vbo, self.data)

        self.ready = True
        ...

    def draw(self):
        if not self.ready:
            return

        draw_vbo(self.vao, GL_TRIANGLE_STRIP, self.n)
