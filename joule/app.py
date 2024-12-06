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
        """
        Joule App: Main class for application

        Extends: CameraOrbitControls, ShaderRenderer

        :param window_size: Initial window size (width, height)
        :param name: Initial window name
        """

        # init camera orbit controls and shader renderer
        super().__init__(*orbit_control_args)

        # initialize window
        self.window = self.window_init(window_size, name)

        # initialize ui
        self.ui = ParameterInterface(
            self.window,
            self.on_evaluate,
            self.on_change_ball_color,
            self.on_change_surface_color,
        )

        # initialize rendering objects
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

        # initialize computation engines
        self.calculus_engine = CalculusEngine()
        self.mechanics_engine = MechanicsEngine(
            initial_gravity=self.ui.gravity_slider,
            initial_friction=self.ui.friction_slider,
        )

        # evaluate initial function to display
        self.on_evaluate(
            self.ui.expression_textbox, self.ui.x_domain_slider, self.ui.y_domain_slider
        )

        # fall into rendering loop
        self.rendering_loop()

    def window_init(self, window_size, name):
        # throw exception if glfw failed to init
        if not glfw.init():
            raise Exception("GLFW could not be initialized.")

        # enable multisampling (antialiasing) on glfw window
        glfw.window_hint(glfw.SAMPLES, 4)

        # create window and context
        window = glfw.create_window(*window_size, name, None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window could not be created.")
        glfw.make_context_current(window)

        # wrapper callback functions to dispatch events to ui and camera
        glfw.set_cursor_pos_callback(window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)
        glfw.set_framebuffer_size_callback(window, self.resize_callback)

        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)

        # initially call window resize to rescale frame
        self.camera_resize_callback(window, *window_size)

        return window

    def key_callback(self, *args):
        # forward ui keyboard callbacks
        if self.ui.want_keyboard:
            self.ui.impl.keyboard_callback(*args)

    def char_callback(self, *args):
        # forward ui keyboard callbacks
        if self.ui.want_keyboard:
            self.ui.impl.char_callback(*args)

    def mouse_button_callback(self, window, button, action, mods):
        # forward ui mouse callbacks
        if self.ui.want_mouse:
            return

        # forward camera mouse callbacks
        self.camera_mouse_button_callback(window, button, action, mods)

        # add ball: left click
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.on_add(window)

    def cursor_pos_callback(self, window, xpos, ypos):
        # forward ui mouse callbacks
        if self.ui.want_mouse:
            return

        # forward camera mouse callbacks
        self.camera_cursor_pos_callback(window, xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        # forward ui mouse callbacks
        if self.ui.want_mouse:
            return

        # forward camera mouse callbacks
        self.camera_scroll_callback(window, xoffset, yoffset)

    def resize_callback(self, window, width, height):
        # forward camera window callback
        self.camera_resize_callback(window, width, height)

    def window_should_close(self):
        return glfw.window_should_close(self.window)

    def rendering_loop(self):
        """
        Main rendering loop for application
        """

        self.render_setup()

        start = time.time()
        dt = 0

        # main rendering loop until user quits
        while not self.window_should_close():

            # update engines and ui
            n_bodies = self.mechanics_engine.get_render_n()
            buffer_size = self.mechanics_engine.get_render_max()

            self.ui.update_status(dt, n_bodies, buffer_size)
            self.mechanics_engine.set_gravity(self.ui.gravity_slider)
            self.mechanics_engine.set_friction(self.ui.friction_slider)

            self.mechanics_engine.update(
                dt, self.calculus_engine, z_correction=self.ui.z_correction
            )

            # call rendering
            self.on_render_frame()
            self.ui.on_render_ui()

            self.ui.impl.process_inputs()
            self.ui.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            # compute dt for integration
            current = time.time()
            dt = current - start
            start = current

        glfw.terminate()

    def on_render_frame(self):
        """
        Render frame event callback
        """

        # setup frame rendering with OpenGL calls
        self.frame_setup(self.ui.background_color)

        # shader: update camera matrices
        self.set_matrix_uniforms(
            self.get_camera_projection(),
            self.get_camera_transform(),
        )

        # shader: update lighting
        self.set_lighting_uniforms(
            glm.vec3(*self.ui.light_color),
            ambient_strength=self.ui.ambient_strength,
            diffuse_strength=self.ui.diffuse_strength,
            diffuse_base=self.ui.diffuse_base,
            specular_strength=self.ui.specular_strength,
            specular_reflection=self.ui.specular_reflection,
        )

        # draw elements
        self.surface.draw()

        positions = self.mechanics_engine.get_render_positions()
        masses = self.mechanics_engine.get_render_masses()
        self.balls.draw(positions, masses, self.calculus_engine)

        if self.ui.show_axes:
            self.axes.draw()

    def on_add(self, window):
        """
        Add ball event callback

        :param window: glfw window
        """

        # get 3D click coordinates
        rh = self.get_right_handed()
        x, y, _ = self.get_click_point(window, rh)

        # evaluate function at click point
        point_mesh = np.array([[x, y]])
        (z,) = self.calculus_engine.build_values(point_mesh)

        # add ball at coordinates
        self.mechanics_engine.add_ball((x, y, z), self.ui.mass_slider)

    def on_evaluate(self, expression, x_domain, y_domain):
        # update calculus engine with new function
        parser_message = self.calculus_engine.update_function(expression)
        self.mechanics_engine.clear()

        # update axes
        ranges = self.axes.compute_ranges(x_domain, y_domain)
        self.axes.update_domain(*ranges)

        # update surface
        point_mesh = self.surface.get_point_mesh(x_domain, y_domain)
        self.surface.update_function(
            point_mesh,
            self.calculus_engine.build_values(point_mesh),
            self.calculus_engine.build_normals(point_mesh),
        )

        # acquire all functions for ui
        f = self.calculus_engine.get_function(symbolic=True)
        x, y = self.calculus_engine.x, self.calculus_engine.y
        fx = self.calculus_engine.get_partial(x, 1, symbolic=True)
        fxx = self.calculus_engine.get_partial(x, 2, symbolic=True)
        fy = self.calculus_engine.get_partial(y, 1, symbolic=True)
        fyy = self.calculus_engine.get_partial(y, 2, symbolic=True)
        fxy = self.calculus_engine.get_mixed_partial(symbolic=True)

        p = self.calculus_engine.pretty_print

        # update ui
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
        """
        Change balls color

        :param color: New balls color vector as shape (3,)
        """

        self.balls.set_color(color)

    def on_change_surface_color(self, color):
        """
        Change surface color

        :param color: New surface color vector as shape (3,)
        """

        self.surface.set_color(color)


def run():
    # run the app
    App((1280, 720), "Joule")
