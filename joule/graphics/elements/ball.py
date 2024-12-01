from itertools import product

import numpy as np

from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import GL_TRIANGLE_STRIP


def generate_sphere_vertices_fast(radius, res):
    v = np.ones((res, res, 4, 3), dtype=np.float32) * radius

    latitude = np.linspace(np.pi / 2, -np.pi / 2, res + 1)
    longitude = np.linspace(0, 2 * np.pi, res + 1)

    long, lat = np.meshgrid(longitude, latitude)

    for i, j in product(range(2), range(2)):
        endpoint = lambda n: n if n != 0 else None
        ii = endpoint(i - 1)
        jj = endpoint(j - 1)

        l_i = long[i:ii, i:ii]
        l_j = lat[j:jj, j:jj]

        v[:, :, i + j * 2, 0] *= np.cos(l_j) * np.cos(l_i)
        v[:, :, i + j * 2, 1] *= np.cos(l_j) * np.sin(l_i)
        v[:, :, i + j * 2, 2] *= np.sin(l_j)

    return v.reshape((-1, 3))


class Ball:
    def __init__(self):
        vertices = generate_sphere_vertices_fast(1, 25)
        self.n = len(vertices)
        self.data = np.ones((len(vertices), 9), dtype=np.float32)
        # self.data[:, :3] = vertices * 0.125 + [1, 1, 1]
        self.data[:, :3] = vertices * 0.125
        self.data[:, 3:6] = [0.25, 0.25, 0.25]
        self.data[:, 6:9] = vertices

        self.vao, self.vbo = create_vbo(self.data, return_vbo=True, store_normals=True)

    def draw(self, s):
        data = np.copy(self.data)
        data[:, :3] += s

        update_vbo(self.vbo, data)

        draw_vbo(self.vao, GL_TRIANGLE_STRIP, self.n)
