import time

import glfw
import sympy
import numpy as np
from numpy import *

import glm
import imgui

from joule.graphics.elements.axes import Axes
from joule.graphics.elements.ball import Ball
from joule.graphics.elements.test import Test
from joule.graphics.orbit_controls import CameraOrbitControls
from joule.graphics.shader_renderer import ShaderRenderer

from imgui.integrations.glfw import GlfwRenderer

from OpenGL.GL import *
from OpenGL.GLU import *

from joule.mechanics.graph import GraphEngine


# main class for the simulation and usage of it
class App(CameraOrbitControls, ShaderRenderer):
    def __init__(
        self,
        window_size,
        name,
        *orbit_control_args,
    ):
        super().__init__(*orbit_control_args)

        # Creates window and buttons
        self.window = self.window_init(window_size, name)
        self.imgui_impl = self.init_imgui(self.window)

        self.axes = Axes()
        self.graph_engine = GraphEngine()
        self.graph_engine.update_function(
            "-sin(1/(sqrt(x**2 + y**2)))", -np.pi, np.pi, -np.pi, np.pi
        )

        self.rendering_loop(self.window, self.imgui_impl)

    def window_init(self, window_size, name):
        # Raises an exception if GLFW couldn't be initiated
        if not glfw.init():
            raise Exception("GLFW could not be initialized.")

        # Creates window
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(*window_size, name, None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window could not be created.")

        # Gets and uses information needed to maintain and update the window
        glfw.make_context_current(window)

        # Attach functions to callback
        glfw.set_cursor_pos_callback(window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)
        glfw.set_framebuffer_size_callback(window, self.resize_callback)

        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

        # Resizes window
        self.resize_callback(window, *window_size)

        return window

    def init_imgui(self, window):
        # Creates imgui context and renderer
        imgui.create_context()
        return GlfwRenderer(window, attach_callbacks=False)

    def imgui_want_mouse(self):
        return imgui.get_io().want_capture_mouse

    def imgui_want_keyboard(self):
        return imgui.get_io().want_capture_keyboard

    def key_callback(self, *args):
        if self.imgui_want_keyboard():
            self.imgui_impl.keyboard_callback(*args)

    def char_callback(self, *args):
        if self.imgui_want_keyboard():
            self.imgui_impl.char_callback(*args)

    def mouse_button_callback(self, window, button, action, mods):
        # Forward imgui mouse events
        if self.imgui_want_mouse():
            return

        super().mouse_button_callback(window, button, action, mods)

        if button == glfw.MOUSE_BUTTON_LEFT:
            xpos, ypos = glfw.get_cursor_pos(window)

            win_x, win_y = glfw.get_window_size(window)
            ypos = win_y - ypos

            viewport = glm.vec4(0, 0, win_x, win_y)

            pos = self.get_camera_transform()
            proj = self.get_camera_projection()
            rh = self.get_right_handed()

            # modelview = glm.inverse(pos * rh * proj)
            # modelview = glm.inverse(pos * rh)
            modelview = pos * rh

            depth = glReadPixels(xpos, ypos, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
            print(depth)

            # x_ndc = (2.0 * xpos / win_x) - 1.0
            # y_ndc = 1.0 - (2.0 * ypos / win_y)

            self.pos_3d = glm.unProject(
                glm.vec3(xpos, ypos, depth), modelview, proj, viewport
            )
            print(self.pos_3d)
            # glm.unProject(window_pos, )

    def cursor_pos_callback(self, window, xpos, ypos):
        # Forward imgui mouse events
        if self.imgui_want_mouse():
            return

        super().cursor_pos_callback(window, xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        # Forward imgui mouse events
        if self.imgui_want_mouse():
            return

        super().scroll_callback(window, xoffset, yoffset)

    def resize_callback(self, window, width, height):
        # Properly sizes the viewport window to the correct ratio
        glViewport(0, 0, width, height)
        super().resize_callback(window, width, height)

    def window_should_close(self, window):
        # Returns if the window should close
        return glfw.window_should_close(window)

    def terminate(self):
        # Terminates the window
        glfw.terminate()

    def rendering_loop(
        self,
        window,
        imgui_impl,
    ):
        self.render_setup()

        text = ""

        ball = Ball()

        start = time.time()
        dt = 0

        while not self.window_should_close(window):
            self.frame_setup()

            self.set_matrix_uniforms(
                self.get_camera_projection(),
                self.get_camera_transform(),
            )

            self.set_lighting_uniforms(glm.vec3(1, 1, 1))
            self.graph_engine.draw()

            self.set_lighting_uniforms(
                glm.vec3(1, 1, 1),
                diffuse_strength=0.6,
                diffuse_base=0.5,
                specular_strength=1.0,
                specular_reflection=16,
            )
            ball.draw([5, 5, 5])

            glPointSize(20.0)
            glBegin(GL_POINTS)
            # glVertex3f(1, 0, 0)
            # glVertex3f(0, 2, 0)
            # glVertex3f(0, 0, 3)

            if hasattr(self, "pos_3d"):
                glVertex3f(*self.pos_3d.to_list())

            glEnd()

            imgui.new_frame()
            imgui.begin("Test")

            if dt:
                imgui.text(f"{1/dt:.2f} fps")

            changed, text = imgui.input_text("Expression", text, 256)

            if imgui.button("evaluate"):
                self.graph_engine.update_function(text, -np.pi, np.pi, -np.pi, np.pi)

            imgui.end()

            imgui.render()
            imgui_impl.process_inputs()
            imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
            glfw.poll_events()

            current = time.time()
            dt = current - start
            start = current

        self.terminate()


# run the app
def run():
    App((1280, 720), "The Force Awakens")
