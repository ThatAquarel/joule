import numpy as np

from joule.graphics.vbo import create_vbo, draw_vbo, update_vbo

from OpenGL.GL import *


class Test:
    def __init__(self):
        self.data = np.array([[0,0,0],[1,1,1]], dtype=np.float32)
        self.vbo = create_vbo(self.data)
        self.n = 1

    def draw(self, s):
        self.data[0] = s
        update_vbo(self.vbo, self.data)

        glPointSize(100.0)
        draw_vbo(self.vbo, GL_POINTS, self.n)
