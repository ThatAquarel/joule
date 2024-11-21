import importlib.resources

from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.shaders import compileShader, compileProgram

import joule.graphics.shaders as s


def load_shader(package, file, type):
    source = importlib.resources.read_text(package, file)
    return compileShader(source, type)


def get_main_shader():
    v = load_shader(s, "vertex.glsl", GL_VERTEX_SHADER)
    f = load_shader(s, "fragment.glsl", GL_FRAGMENT_SHADER)

    return compileProgram(v, f)
