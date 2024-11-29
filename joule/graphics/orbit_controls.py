import glm
import glfw
import numpy as np


class CameraOrbitControls:
    def __init__(
        self,
        zoom_sensitivity=0.1,
        pan_sensitvity=0.001,
        orbit_sensitivity=0.0025,
        initial_zoom=1,
        clipping=[-32, 32],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._zoom_sensitivity = zoom_sensitivity
        self._pan_sensitivity = pan_sensitvity
        self._orbit_sensitivity = orbit_sensitivity

        self._zoom_level = initial_zoom
        self._clipping = clipping

        self._view_angle = np.array([np.pi / 4, np.pi / 4])
        self._view_pan = np.zeros(2)
        self._view_box = np.zeros(2)
        self._prev_mouse_pos = np.zeros(2)

        self._dragging, self._panning = False, False

    def mouse_button_callback(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_RIGHT:
            return

        self._dragging = action == glfw.PRESS
        self._panning = glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS

    def cursor_pos_callback(self, window, x_pos, y_pos):
        mouse_pos = [x_pos, y_pos]

        if self._dragging:
            ds = mouse_pos - self._prev_mouse_pos

            if self._panning:
                zoomed_pan = self._pan_sensitivity * self._zoom_level
                self._view_pan += ds * [1, -1] * zoomed_pan
            else:
                self._view_angle += ds[::-1] * self._orbit_sensitivity

        self._prev_mouse_pos[:] = mouse_pos

    def scroll_callback(self, window, x_offset, y_offset):
        if y_offset > 0:
            self._zoom_level /= 1 + self._zoom_sensitivity
        elif y_offset < 0:
            self._zoom_level *= 1 + self._zoom_sensitivity

    def resize_callback(self, window, width, height):
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
