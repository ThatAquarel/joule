import numpy as np

from joule.graphics.vbo import create_vao, draw_vao

from OpenGL.GL import *


class Axes:
    def __init__(
        self,
        initial_x_domain,
        initial_y_domain,
    ):
        ranges = self.compute_ranges(initial_x_domain, initial_y_domain)
        self.update_domain(*ranges)

    def compute_ranges(self, function_x_domain, function_y_domain):
        rounded = np.round(np.abs([function_x_domain, function_y_domain]))

        range_max = np.max(rounded)
        domain = (-range_max, range_max)
        divs = (2 * range_max, 2 * range_max)
        return domain, domain, domain, divs

    def update_domain(self, x_range, y_range, z_range, divs):
        x_div, y_div = divs

        offset = [np.min(x_range), np.min(y_range), 0]
        self.x_vbo, self.x_n = self._build_gridline_vbo(
            [1, 0, 2], offset, x_div, *x_range
        )
        self.y_vbo, self.y_n = self._build_gridline_vbo(
            [0, 1, 2], offset, y_div, *y_range
        )
        self.axes_vbo, self.axes_n = self._build_axes_vbo(*x_range, *y_range, *z_range)

    def _build_axes_vbo(self, x_min, x_max, y_min, y_max, z_min, z_max):
        data = np.array(
            [
                [x_min, 0, 0],
                [1, 0, 0],
                [x_max, 0, 0],
                [1, 0, 0],
                [0, y_min, 0],
                [0, 1, 0],
                [0, y_max, 0],
                [0, 1, 0],
                [0, 0, z_min],
                [0, 0, 1],
                [0, 0, z_max],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return create_vao(data), 6

    def _build_scaled_gridlines(self, s_div, s_min, s_max):
        grid = np.mgrid[0:2, 0 : 1 : (s_div + 1) * 1j, 0:1].T

        lines = grid.squeeze(0)
        lines = np.delete(lines, len(lines) // 2, axis=0)

        vertices = lines.reshape((-1, 3))
        return vertices * (s_max - s_min)

    def _build_gridline_color(self, vertices, color=[0.6, 0.6, 0.6]):
        colors = np.ones(vertices.shape, dtype=np.float32) * color
        return np.hstack((vertices, colors)).astype(np.float32)

    def _build_gridline_vbo(self, axis_index, offset, s_div, s_min, s_max):
        axes_s = self._build_scaled_gridlines(s_div, s_min, s_max)[:, axis_index]
        axes_s += offset
        vbo_data = self._build_gridline_color(axes_s)

        return create_vao(vbo_data), len(axes_s)

    def draw(self):
        glLineWidth(0.25)
        draw_vao(self.x_vbo, GL_LINES, self.x_n)
        draw_vao(self.y_vbo, GL_LINES, self.y_n)

        glLineWidth(2.0)
        draw_vao(self.axes_vbo, GL_LINES, self.axes_n)
