
import numpy as np

from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo
from OpenGL.GL import GL_TRIANGLES, GL_TRIANGLE_STRIP


# def generate_sphere_vertices(radius, res):
#     v = np.ones((res, (res + 1) * 2 ,3), dtype=np.float32) * radius

#     latitude = np.linspace(np.pi/2, -np.pi/2, res + 1)
#     longitude = np.linspace(0, 2*np.pi, res, endpoint=False)

#     lat, long = np.meshgrid(latitude[:-1], longitude)
#     v[:, :-2:2, 0] *= np.cos(lat) * np.cos(long)
#     v[:, :-2:2, 1] *= np.sin(lat)
#     v[:, :-2:2, 2] *= np.cos(lat) * np.sin(long)

#     lat, long = np.meshgrid(latitude[1:], longitude)
#     v[:, 1:-2:2, 0] *= np.cos(lat) * np.cos(long)
#     v[:, 1:-2:2, 1] *= np.sin(lat)
#     v[:, 1:-2:2, 2] *= np.cos(lat) * np.sin(long)

#     v[:, -2:]= v[:, :2]

#     return v.reshape((-1, 3))


def generate_sphere_vertices(radius, res):
    vertices = []

    stacks = res
    slices = res

    # Generates all the points of any given spherical object (planets, black holes, etc)
    for i in range(stacks):
        # Gets the latitudinal sections of each vertice
        lat0 = np.pi * (-0.5 + i / stacks)
        lat1 = np.pi * (-0.5 + (i + 1) / stacks)
        sin_lat0, cos_lat0 = np.sin(lat0), np.cos(lat0)
        sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)

        for j in range(slices):
            # Gets the longitudinal sections of each vertice
            lon0 = 2 * np.pi * (j / slices)
            lon1 = 2 * np.pi * ((j + 1) / slices)
            sin_lon0, cos_lon0 = np.sin(lon0), np.cos(lon0)
            sin_lon1, cos_lon1 = np.sin(lon1), np.cos(lon1)

            # Combines the latitudinal and longitudinal components of the sphere to form it's vertices

            v0 = [
                radius * cos_lat0 * cos_lon0,
                radius * cos_lat0 * sin_lon0,
                radius * sin_lat0,
            ]
            v1 = [
                radius * cos_lat1 * cos_lon0,
                radius * cos_lat1 * sin_lon0,
                radius * sin_lat1,
            ]
            v2 = [
                radius * cos_lat1 * cos_lon1,
                radius * cos_lat1 * sin_lon1,
                radius * sin_lat1,
            ]
            v3 = [
                radius * cos_lat0 * cos_lon1,
                radius * cos_lat0 * sin_lon1,
                radius * sin_lat0,
            ]

            vertices.extend(v0)
            vertices.extend(v1)
            vertices.extend(v3)
            vertices.extend(v2)

    return np.array(vertices, dtype=np.float32).reshape((-1, 3))





class Ball:
    def __init__(self):
        vertices = generate_sphere_vertices(1, 10)
        self.n = len(vertices)
        self.data = np.zeros((len(vertices), 6), dtype=np.float32)
        self.data[:, :3] = vertices
        self.data[:, 3:6] = np.random.random((len(vertices), 3))

        self.vao, self.vbo = create_vbo(self.data, return_vbo=True)

    def draw(self, s):
        # data = self.data[:, :3] + s
        # update_vbo(self.vbo, data)

        draw_vbo(self.vao, GL_TRIANGLE_STRIP, self.n)

