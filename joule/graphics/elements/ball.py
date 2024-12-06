from itertools import product

import numpy as np
from OpenGL.GL import GL_TRIANGLE_STRIP

from joule.compute.calculus import CalculusEngine
from joule.compute.linalg import column_wise
from joule.graphics.vbo import create_vao, draw_vao, update_vbo


def generate_sphere_vertices_fast(radius, res):
    """
    Generate vertices of a sphere for OpenGL draw

    :param radius: radius of sphere
    :param res: resolution of vertices
    """

    # preallocate an array to contain the vertices
    # that takes the value of radius
    v = np.ones((res, res, 4, 3), dtype=np.float32) * radius

    # compute the set of latitudes and longitudes
    # of every point to form a sphere
    latitude = np.linspace(np.pi / 2, -np.pi / 2, res + 1)
    longitude = np.linspace(0, 2 * np.pi, res + 1)

    # compute possible combinations of long and lat
    # for every vertex that lies tangent to unit sphere
    long, lat = np.meshgrid(longitude, latitude)

    # every four vertices form a quadrilaterial
    # (ie: two triangles for OpenGL)
    for i, j in product(range(2), range(2)):
        # makes sure that index rollsover properly
        endpoint = lambda n: n if n != 0 else None
        ii = endpoint(i - 1)
        jj = endpoint(j - 1)

        # get current long and lat of quadrilateral
        l_i = long[i:ii, i:ii]
        l_j = lat[j:jj, j:jj]

        # project spherical coordinates back into
        # cartesian
        # formulas from: https://en.wikipedia.org/wiki/Spherical_coordinate_system
        v[:, :, i + j * 2, 0] *= np.cos(l_j) * np.cos(l_i)
        v[:, :, i + j * 2, 1] *= np.cos(l_j) * np.sin(l_i)
        v[:, :, i + j * 2, 2] *= np.sin(l_j)

    # return cartesian vertices of shape (n, 3)
    return v.reshape((-1, 3))


class Ball:
    def __init__(self, initial_color, res=25):
        """
        Ball: Balls render element

        :param initial_color: Balls initial color
        :param res: Balls vertices resolution
        """

        # generate a unit sphere coordinates
        vertices = generate_sphere_vertices_fast(1, res)
        self.n = len(vertices)

        # preallocate data buffer, containing
        # position, color and normals
        self.data = np.ones((len(vertices), 9), dtype=np.float32)
        self.data[:, :3] = vertices
        self.data[:, 3:6] = initial_color
        self.data[:, 6:9] = vertices

        # build VAO and VBO for OpenGL
        self.vao, self.vbo = create_vao(self.data, return_vbo=True, store_normals=True)

    def _draw_ball(self, r, s):
        """
        Draw a ball of radius r at position s

        :param r: Scalar radius of ball
        :param s: Position of ball as vector of shape (3,)
        """

        # TODO: Remove this redundant memcopy
        # this is very slow, but it'll require a refactor of the
        # whole rendering architecture, which I didn't have time to do

        # make a copy of the default buffer
        data = np.copy(self.data)
        # resize ball to new resize and translate
        data[:, :3] = data[:, :3] * r + s

        # update VBO and draw this ball
        update_vbo(self.vbo, data)
        draw_vao(self.vao, GL_TRIANGLE_STRIP, self.n)

    def set_color(self, new_color):
        """
        Update all balls' color

        :param new_color: New balls color
        """

        # change default buffer's color, so that subsequent
        # copies in self._draw_ball take this new color
        self.data[:, 3:6] = new_color
        update_vbo(self.vbo, self.data)

    def draw(self, positions, masses, calculus_engine: CalculusEngine):
        """
        Draw all balls

        :param positions: Array of positions to draw ball of shape (n, 3)
        :param masses: Array of ball masses of shape (n,)
        :param calculus_engine: joule.compute.CalculusEngine instance
        """

        # skip draw if no balls
        if not len(positions):
            return

        # evaluate normals at the position of the balls
        point_mesh = positions[:, :2]
        normals = calculus_engine.build_normals(point_mesh)

        # calculate radius based on uniform density
        # V=4pi r^3/3
        radii = np.cbrt(3 * masses / (4 * np.pi)) * 0.08

        # using the normals, place each ball tangential to the surface
        positions += normals * column_wise(radii)

        # draw each ball with its distinct radius
        for pos, radius in zip(positions, radii):
            self._draw_ball(radius, pos)
