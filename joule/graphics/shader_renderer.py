import importlib.resources

import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

import joule.graphics.shaders


class ShaderRenderer:
    def __init__(self, background_color=(0.86, 0.87, 0.87), **kwargs):
        super().__init__(**kwargs)

        self._background_color = background_color
        self._transform = glm.mat4x4(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self._view_vec = glm.vec4(0, 0, 10, 1)

    def get_right_handed(self):
        return self._transform

    def _load_shader_source(self, file, type):
        source = importlib.resources.read_text(joule.graphics.shaders, file)
        return compileShader(source, type)

    def _load_shader(self):
        v_shader = self._load_shader_source("vertex.glsl", GL_VERTEX_SHADER)
        f_shader = self._load_shader_source("fragment.glsl", GL_FRAGMENT_SHADER)

        return compileProgram(v_shader, f_shader)

    def _uniform_float(self, name, value):
        location = glGetUniformLocation(self._shader, name)
        glUniform1f(location, value)

    def _uniform_vec3(self, name, glm_vec3):
        location = glGetUniformLocation(self._shader, name)
        glUniform3fv(location, 1, glm.value_ptr(glm_vec3))

    def _uniform_mat4(self, name, glm_mat4):
        location = glGetUniformLocation(self._shader, name)
        glUniformMatrix4fv(location, 1, GL_TRUE, glm.value_ptr(glm_mat4))

    def render_setup(self):
        self._shader = self._load_shader()

        glEnable(GL_DEPTH_TEST)

        # enable antialiasing (smooth lines)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_POINT_SMOOTH)

        # enable opacity
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

    def frame_setup(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self._background_color, 1.0)
        glUseProgram(self._shader)

    def set_matrix_uniforms(
        self,
        cam_projection,
        cam_transform,
    ):
        self._uniform_mat4("world_transform", self.get_right_handed())
        self._uniform_mat4("cam_projection", cam_projection)
        self._uniform_mat4("cam_transform", cam_transform)

        cam_transform_inv = glm.inverse(cam_transform)
        view_pos = glm.vec3(cam_transform_inv * self._view_vec)

        self._uniform_vec3("view_pos", view_pos)
        self._uniform_vec3("light_pos", view_pos)

    def set_lighting_uniforms(
        self,
        light_color,
        ambient_strength=0.2,
        diffuse_strength=0.2,
        diffuse_base=0.3,
        specular_strength=0.1,
        specular_reflection=64.0,
    ):
        self._uniform_vec3("light_color", light_color)

        self._uniform_float("ambient_strength", ambient_strength)
        self._uniform_float("diffuse_strength", diffuse_strength)
        self._uniform_float("diffuse_base", diffuse_base)
        self._uniform_float("specular_strength", specular_strength)
        self._uniform_float("specular_reflection", specular_reflection)
