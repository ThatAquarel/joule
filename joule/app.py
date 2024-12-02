import time

import glfw
import sympy as sp
import numpy as np
from numpy import *

import glm
import imgui

from joule.compute.mecanics import MecanicsEngine
from joule.graphics.elements.axes import Axes
from joule.graphics.elements.ball import Ball
from joule.graphics.elements.surface import Surface
from joule.graphics.elements.test import Test
from joule.graphics.orbit_controls import CameraOrbitControls
from joule.graphics.shader_renderer import ShaderRenderer

from imgui.integrations.glfw import GlfwRenderer

from OpenGL.GL import *
from OpenGL.GLU import *

from joule.compute.calculus import CalculusEngine


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
        self.balls = Ball()
        self.calculus_engine = CalculusEngine()
        self.mecanics_engine = MecanicsEngine()

        self.surface = Surface(res=1024)
        self.update_function("sin(x + y)", [-np.pi, np.pi], [-np.pi, np.pi])

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
        self.camera_resize_callback(window, *window_size)

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

        self.camera_mouse_button_callback(window, button, action, mods)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.on_add(window)

    def cursor_pos_callback(self, window, xpos, ypos):
        # Forward imgui mouse events
        if self.imgui_want_mouse():
            return

        self.camera_cursor_pos_callback(window, xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        # Forward imgui mouse events
        if self.imgui_want_mouse():
            return

        self.camera_scroll_callback(window, xoffset, yoffset)

    def resize_callback(self, window, width, height):
        # Properly sizes the viewport window to the correct ratio
        glViewport(0, 0, width, height)
        self.camera_resize_callback(window, width, height)

    def window_should_close(self, window):
        # Returns if the window should close
        return glfw.window_should_close(window)

    def terminate(self):
        # Terminates the window
        glfw.terminate()

    def _ui_space(self):
        for _ in range(5):
            imgui.spacing()

    def rendering_loop(
        self,
        window,
        imgui_impl,
    ):
        self.render_setup()

        text = ""
        parser_message = ""

        start = time.time()
        dt = 0

        while not self.window_should_close(window):
            self.mecanics_engine.update(dt, self.calculus_engine, z_correction=True)

            self.on_frame(dt)

            imgui.new_frame()
            imgui.begin("Joule")

            imgui.text("Graphics")
            imgui.separator()
            if dt:
                imgui.text(f"{1/dt:.2f} fps")
            n_bodies = self.mecanics_engine.get_render_n()
            max_bodies = self.mecanics_engine.get_render_max()
            imgui.text(f"{n_bodies}/{max_bodies} bodies")

            self._ui_space()
            imgui.text("Expression")
            imgui.separator()

            changed, text = imgui.input_text_multiline(
                "",
                text,
                1024,
                200,
                100,
                imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
            )

            if imgui.button("Evaluate"):
                parser_message = self.update_function(
                    text, [-np.pi, np.pi], [-np.pi, np.pi]
                )
                text = ""

            imgui.text(parser_message)

            self._ui_space()
            imgui.separator()
            imgui.text("Functions")

            for name, expression in self.function_texts.items():
                imgui.spacing()
                imgui.text(f"{name}\n{expression}")

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

    def on_frame(self, dt):
        self.frame_setup()

        self.set_matrix_uniforms(
            self.get_camera_projection(),
            self.get_camera_transform(),
        )

        self.set_lighting_uniforms(glm.vec3(1, 1, 1))
        self.surface.draw()

        self.set_lighting_uniforms(
            glm.vec3(1, 1, 1),
            diffuse_strength=0.6,
            diffuse_base=0.5,
            specular_strength=1.0,
            specular_reflection=16,
        )

        positions = self.mecanics_engine.get_render_positions()
        self.balls.draw(positions, self.calculus_engine)

    def on_add(self, window):
        rh = self.get_right_handed()
        x, y, _ = self.get_click_point(window, rh)

        point_mesh = np.array([[x, y]])
        (z,) = self.calculus_engine.build_values(point_mesh)

        self.mecanics_engine.add_ball((x, y, z), 10)

    def update_function(self, text, x_domain, y_domain):
        parser_message = self.calculus_engine.update_function(text)
        self.mecanics_engine.clear()

        point_mesh = self.surface.get_point_mesh(x_domain, y_domain)
        self.surface.update_function(
            point_mesh,
            self.calculus_engine.build_values(point_mesh),
            self.calculus_engine.build_normals(point_mesh),
        )

        f = self.calculus_engine.get_function(symbolic=True)
        x, y = self.calculus_engine.x, self.calculus_engine.y
        fx = self.calculus_engine.get_partial(x, 1, symbolic=True)
        fxx = self.calculus_engine.get_partial(x, 2, symbolic=True)
        fy = self.calculus_engine.get_partial(y, 1, symbolic=True)
        fyy = self.calculus_engine.get_partial(y, 2, symbolic=True)
        fxy = self.calculus_engine.get_mixed_partial(symbolic=True)

        p = self.calculus_engine.pretty_print

        self.function_texts = {
            "f(x,y) =": p(f),
            "df/dx =": p(fx),
            "df/dy =": p(fy),
            "d2f/dx2 =": p(fxx),
            "d2f/dy2 =": p(fyy),
            "d2f/dxdy =": p(fxy),
        }

        return parser_message


# run the app
def run():
    App((1280, 720), "Joule")
