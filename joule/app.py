import time

import glfw
import numpy as np

import glm
import imgui

from joule.graphics.orbit_controls import CameraOrbitControls
from joule.graphics.parameter_interface import ParameterInterface
from joule.graphics.shader_renderer import ShaderRenderer

from joule.graphics.elements.axes import Axes
from joule.graphics.elements.ball import Ball
from joule.graphics.elements.surface import Surface

from joule.compute.mechanics import MechanicsEngine
from joule.compute.calculus import CalculusEngine


class App(CameraOrbitControls, ShaderRenderer):
    def __init__(
        self,
        window_size,
        name,
        *orbit_control_args,
    ):
        super().__init__(*orbit_control_args)

        self.window = self.window_init(window_size, name)
        self.ui = ParameterInterface(
            self.window,
            self.on_evaluate,
            self.on_change_ball_color,
            self.on_change_surface_color,
        )

        self.axes = Axes(
            self.ui.x_domain_slider,
            self.ui.y_domain_slider,
        )
        self.balls = Ball(
            initial_color=self.ui.ball_color,
            res=25,
        )
        self.surface = Surface(
            initial_color=self.ui.surface_color,
            res=1024,
        )

        self.calculus_engine = CalculusEngine()
        self.mechanics_engine = MechanicsEngine(
            initial_gravity=self.ui.gravity_slider,
            initial_friction=self.ui.friction_slider,
        )

        self.on_evaluate(
            self.ui.expression_textbox, self.ui.x_domain_slider, self.ui.y_domain_slider
        )

        self.rendering_loop()

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

    def key_callback(self, *args):
        if self.ui.want_mouse:
            self.ui.impl.keyboard_callback(*args)

    def char_callback(self, *args):
        if self.ui.want_keyboard:
            self.ui.impl.char_callback(*args)

    def mouse_button_callback(self, window, button, action, mods):
        # Forward imgui mouse events
        if self.ui.want_mouse:
            return

        self.camera_mouse_button_callback(window, button, action, mods)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.on_add(window)

    def cursor_pos_callback(self, window, xpos, ypos):
        # Forward imgui mouse events
        if self.ui.want_mouse:
            return

        self.camera_cursor_pos_callback(window, xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        # Forward imgui mouse events
        if self.ui.want_mouse:
            return

        self.camera_scroll_callback(window, xoffset, yoffset)

    def resize_callback(self, window, width, height):
        # Properly sizes the viewport window to the correct ratio
        self.camera_resize_callback(window, width, height)

    def window_should_close(self):
        # Returns if the window should close
        return glfw.window_should_close(self.window)

    def rendering_loop(self):
        self.render_setup()

        start = time.time()
        dt = 0

        while not self.window_should_close():
            n_bodies = self.mechanics_engine.get_render_n()
            buffer_size = self.mechanics_engine.get_render_max()

            self.ui.update_status(dt, n_bodies, buffer_size)
            self.mechanics_engine.set_gravity(self.ui.gravity_slider)
            self.mechanics_engine.set_friction(self.ui.friction_slider)

            self.mechanics_engine.update(
                dt, self.calculus_engine, z_correction=self.ui.z_correction
            )

            self.on_render_frame()
            self.ui.on_render_ui()

            self.ui.impl.process_inputs()
            self.ui.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            current = time.time()
            dt = current - start
            start = current

        glfw.terminate()

    def on_render_frame(self):
        self.frame_setup(self.ui.background_color)

        self.set_matrix_uniforms(
            self.get_camera_projection(),
            self.get_camera_transform(),
        )

        self.surface.draw()

        self.set_lighting_uniforms(
            glm.vec3(*self.ui.light_color),
            ambient_strength=self.ui.ambient_strength,
            diffuse_strength=self.ui.diffuse_strength,
            diffuse_base=self.ui.diffuse_base,
            specular_strength=self.ui.specular_strength,
            specular_reflection=self.ui.specular_reflection,
        )

        positions = self.mechanics_engine.get_render_positions()
        masses = self.mechanics_engine.get_render_masses()
        self.balls.draw(positions, masses, self.calculus_engine)

        if self.ui.show_axes:
            self.axes.draw()

    def on_add(self, window):
        rh = self.get_right_handed()
        x, y, _ = self.get_click_point(window, rh)

        point_mesh = np.array([[x, y]])
        (z,) = self.calculus_engine.build_values(point_mesh)

        self.mechanics_engine.add_ball((x, y, z), self.ui.mass_slider)

    def on_evaluate(self, expression, x_domain, y_domain):
        parser_message = self.calculus_engine.update_function(expression)
        self.mechanics_engine.clear()

        ranges = self.axes.compute_ranges(x_domain, y_domain)
        self.axes.update_domain(*ranges)

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

        self.ui.update_differentiation(
            parser_message,
            {
                "f(x,y) =": p(f),
                "df/dx =": p(fx),
                "df/dy =": p(fy),
                "d2f/dx2 =": p(fxx),
                "d2f/dy2 =": p(fyy),
                "d2f/dxdy =": p(fxy),
            },
        )

    def on_change_ball_color(self, color):
        self.balls.set_color(color)

    def on_change_surface_color(self, color):
        self.surface.set_color(color)


# run the app
def run():
    App((1280, 720), "Joule")
