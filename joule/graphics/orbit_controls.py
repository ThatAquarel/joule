import glm
import glfw
import numpy as np


from OpenGL.GL import glViewport, glReadPixels, GL_DEPTH_COMPONENT, GL_FLOAT


class CameraOrbitControls:
    def __init__(
        self,
        zoom_sensitivity=0.1,
        pan_sensitvity=0.001,
        orbit_sensitivity=0.0025,
        initial_zoom=5,
        initial_view_angle=(np.pi / 6, np.pi / 4),
        clipping=[-32, 32],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._zoom_sensitivity = zoom_sensitivity
        self._pan_sensitivity = pan_sensitvity
        self._orbit_sensitivity = orbit_sensitivity

        self._zoom_level = initial_zoom
        self._clipping = clipping

        self._view_angle = np.array(initial_view_angle)
        self._view_pan = np.zeros(2)
        self._view_box = np.zeros(2)
        self._prev_mouse_pos = np.zeros(2)

        self._dragging, self._panning = False, False

    def camera_mouse_button_callback(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_RIGHT:
            return

        self._dragging = action == glfw.PRESS
        self._panning = glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS

    def camera_cursor_pos_callback(self, window, x_pos, y_pos):
        mouse_pos = [x_pos, y_pos]

        if self._dragging:
            ds = mouse_pos - self._prev_mouse_pos

            if self._panning:
                zoomed_pan = self._pan_sensitivity * self._zoom_level
                self._view_pan += ds * [1, -1] * zoomed_pan
            else:
                self._view_angle += ds[::-1] * self._orbit_sensitivity

        self._prev_mouse_pos[:] = mouse_pos

    def camera_scroll_callback(self, window, x_offset, y_offset):
        if y_offset > 0:
            self._zoom_level /= 1 + self._zoom_sensitivity
        elif y_offset < 0:
            self._zoom_level *= 1 + self._zoom_sensitivity

    def camera_resize_callback(self, window, width, height):
        glViewport(0, 0, width, height)

        aspect_ratio = width / height if height > 0 else 1.0
        self._view_box[:] = [-aspect_ratio, aspect_ratio]

    def get_camera_projection(self):
        p = glm.ortho(
            *self._view_box * self._zoom_level,
            -self._zoom_level,
            self._zoom_level,
            *self._clipping,
        )

        return p

    def get_camera_transform(self):
        t = glm.translate(glm.vec3(*self._view_pan, 0.0))
        t = glm.rotate(t, self._view_angle[0], (1.0, 0.0, 0.0))
        t = glm.rotate(t, self._view_angle[1], (0.0, 1.0, 0.0))

        return t

    def get_click_point(self, window, world_transform):
        xpos, ypos = glfw.get_cursor_pos(window)
        win_x, win_y = glfw.get_window_size(window)

        ypos = win_y - ypos

        depth = glReadPixels(xpos, ypos, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        click = glm.vec3(xpos, ypos, depth)

        pos = self.get_camera_transform()
        modelview = pos * world_transform

        proj = self.get_camera_projection()

        viewport = glm.vec4(0, 0, win_x, win_y)

        return glm.unProject(click, modelview, proj, viewport)
