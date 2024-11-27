import time

import glfw
import sympy
import numpy as np
from numpy import *

import glm
import imgui

from joule.graphics.elements.axes import Axes
from joule.graphics.elements.test import Test
import joule.graphics.shaders.load as shader
import joule.graphics.vbo as vbo

from imgui.integrations.glfw import GlfwRenderer

from OpenGL.GL import *
from OpenGL.GLU import *

from joule.mechanics.graph import GraphEngine

# Drawing transformation array to transform OpenGL coordinates to right-handed physics coordinate system
T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# Sets the axes for the simulation
_axes_y = np.mgrid[0:2, 0:1:11j, 0:1].T.reshape((-1, 3)) - [0.5, 0.5, 0.0]
_axes_x = _axes_y[:, [1, 0, 2]]


# main class for the simulation and usage of it
class App:
    def __init__(
        self,
        window_size,
        name,
        zoom_sensitivity=0.1,
        pan_sensitvity=0.001,
        orbit_sensitivity=0.0025,
        start_zoom=1,
    ):
        # Setting attributes of the class and starting conditions
        self.zoom_sensitivity = zoom_sensitivity
        self.pan_sensitvity = pan_sensitvity
        self.orbit_sensitivity = orbit_sensitivity

        # Camera movement conditions
        self.angle_x, self.angle_y = np.pi / 4, np.pi / 4
        self.pan_x, self.pan_y = 0.0, 0.0
        self.last_x, self.last_y = 0.0, 0.0
        self.dragging, self.panning = False, False
        self.zoom_level = start_zoom

        self.view_left, self.view_right = 0, 0

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

        # Most recent position of cursor
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

        # Resizes window
        self.resize_callback(window, *window_size)

        return window

    def init_imgui(self, window):
        # Creates imgui context and renderer
        imgui.create_context()
        return GlfwRenderer(window, attach_callbacks=False)

    def key_callback(self, *args):
        if self.imgui_impl != None and imgui.get_io().want_capture_keyboard:
            self.imgui_impl.keyboard_callback(*args)

    def char_callback(self, *args):
        if self.imgui_impl != None and imgui.get_io().want_capture_keyboard:
            self.imgui_impl.char_callback(*args)

    def mouse_button_callback(self, window, button, action, mods):
        # Forward imgui mouse events
        if self.imgui_impl != None and imgui.get_io().want_capture_mouse:
            return

        press = action == glfw.PRESS

        # If a given button is pressed, the screen inside the window is panned or rotated
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = press
            shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            self.panning = shift
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.dragging = press
            self.panning = press

    def cursor_pos_callback(self, window, xpos, ypos):
        # Forward imgui mouse events
        if self.imgui_impl != None and imgui.get_io().want_capture_mouse:
            return

        # If the cursor is dragging, updates the position of the cursor in the program
        if self.dragging:
            dx = xpos - self.last_x
            dy = ypos - self.last_y
            if self.panning:
                self.pan_x += dx * self.pan_sensitvity * self.zoom_level
                self.pan_y -= dy * self.pan_sensitvity * self.zoom_level
            else:
                self.angle_x += dy * self.orbit_sensitivity
                self.angle_y += dx * self.orbit_sensitivity

        self.last_x, self.last_y = xpos, ypos

    def scroll_callback(self, window, xoffset, yoffset):
        # Forward imgui mouse events
        if self.imgui_impl != None and imgui.get_io().want_capture_mouse:
            return

        # Zooms in and out
        if yoffset > 0:
            self.zoom_level /= 1 + self.zoom_sensitivity
        elif yoffset < 0:
            self.zoom_level *= 1 + self.zoom_sensitivity

    def resize_callback(self, window, width, height):
        # Properly sizes the viewport window to the correct ratio
        glViewport(0, 0, width, height)

        # Ensure aspect ratio is always the same as window;
        # makes sure rendered objects aren't stretched
        aspect_ratio = width / height if height > 0 else 1.0
        self.view_left = -aspect_ratio
        self.view_right = aspect_ratio

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
        shader_prog = shader.get_main_shader()

        # enable depth and occlusion
        glEnable(GL_DEPTH_TEST)

        # enable antialiasing (smooth lines)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_POINT_SMOOTH)

        # enable opacity
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        text = ""

        while not self.window_should_close(window):
            # Updates the introdution
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.86, 0.87, 0.87, 1.0)

            # Updates the window, background, and axes
            # self.update()
            # self.draw_axes()

            glUseProgram(shader_prog)

            proj = glm.ortho(
                self.view_left * self.zoom_level,
                self.view_right * self.zoom_level,
                -self.zoom_level,
                self.zoom_level,
                -32,
                32,
            )

            proj_loc = glGetUniformLocation(shader_prog, "cam_projection")
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, glm.value_ptr(proj))

            pos = glm.translate(glm.vec3(self.pan_x, self.pan_y, 0.0))
            pos @= glm.rotate(self.angle_x, (1.0, 0.0, 0.0))
            pos @= glm.rotate(self.angle_y, (0.0, 1.0, 0.0))

            pos_loc = glGetUniformLocation(shader_prog, "cam_position")
            glUniformMatrix4fv(pos_loc, 1, GL_TRUE, glm.value_ptr(pos))

            light_loc = glGetUniformLocation(shader_prog, "light_pos_u")
            vec = glm.vec4(0, 0, 10, 1)

            # vec = glm.vec3(100*np.cos(time.time()), 100, 100*np.sin(time.time()))
            glUniform4fv(light_loc, 1, glm.value_ptr(vec))

            view_loc = glGetUniformLocation(shader_prog, "view_pos")
            view_vec = glm.vec4(0, 0, 10, 1)
            # view_pos = view_vec * glm.inverse(pos)
            # view_pos = view_vec * glm.inverse(pos)
            view_pos = view_vec * pos
            view_pos = glm.vec3(view_pos)
            glUniform3fv(view_loc, 1, glm.value_ptr(view_pos))


            # self.axes.draw()
            self.graph_engine.draw()

            imgui.new_frame()
            imgui.begin("Test")

            changed, text = imgui.input_text("Expression", text, 256)

            if imgui.button("evaluate"):
                self.graph_engine.update_function(text, -np.pi, np.pi, -np.pi, np.pi)

            imgui.end()

            imgui.render()
            imgui_impl.process_inputs()
            imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
            glfw.poll_events()

        self.terminate()


# run the app
def run():
    App((1280, 720), "The Force Awakens")
