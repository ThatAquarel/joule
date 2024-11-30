import numpy as np

from OpenGL.GL import *


class MecanicsEngine:
    def __init__(self, buffer_size=16):
        self._compute_state = np.zeros(buffer_size, dtype=bool)
        self._m, self._s, self._v, self._a = np.zeros((4, buffer_size, 3))

        self._gravity = np.array([0, 0, -9.81])
        self._friction = 0.25

    def get_gravity(self):
        return self._gravity

    def set_gravity(self, gravity):
        self._gravity[:] = gravity

    def get_friction(self):
        return self._friction

    def set_friction(self, friction):
        self._friction = friction

    def _get_available_compute_spot(self):
        i = np.argmin(self._compute_state)
        if self._compute_state[i]:
            distances_to_origin = np.linalg.norm(self._s, axis=1)
            i = np.argmax(distances_to_origin)
            print(f"mecanics buffer full: overwrite {i} (furthest)")
        return i

    def add_ball(self, select_position, f_xy, mass):
        x, y, _ = select_position
        ball_position = [x, y, f_xy(x, y)]

        i = self._get_available_compute_spot()
        self._m[i], self._s[i], self._v[i], self._a[i] = [mass, ball_position, 0, 0]
        self._compute_state[i] = True

    def remove_ball(self, select_position):
        distances_to_select = np.linalg.norm(select_position - self._s, axis=1)
        i = np.argmin(distances_to_select)
        self._compute_state[i] = False

    def clear(self):
        self._compute_state[:] = False

    def update(self, dt, f_xy, d_dx, d_dy, surface):
        if self._compute_state.sum() == 0:
            return

        m = self._m[self._compute_state]
        v = self._v[self._compute_state]
        xy = self._s[self._compute_state, :2]

        normal, tangent = surface._build_normals(
            d_dx, d_dy, xy, return_derivative_vector=True
        )

        # d_dx_mesh = d_dx(*xy.T)  #  y constant, xz tan vec
        # d_dx_vec = surface._partial_derivative_tangent_vector(d_dx_mesh, 2, 0)

        # d_dy_mesh = d_dy(*xy.T)  #  x constant, yz tan vec
        # d_dy_vec = surface._partial_derivative_tangent_vector(d_dy_mesh, 2, 1)

        # tangent_vec = np.ones(d_dx_vec.shape)
        # tangent_vec[:, 0] = 1 / d_dx_mesh
        # tangent_vec[:, 1] = 1 / d_dy_mesh

        sign = -np.sign(tangent[:, -1])
        tangent_downward = tangent * sign[:, np.newaxis]

        Fg = self.get_gravity()
        Fg_y = np.vecdot(Fg, normal)[:, np.newaxis] * normal
        Fg_x = np.vecdot(Fg, tangent_downward)[:, np.newaxis] * tangent_downward
        N = np.linalg.norm(Fg_y)
        fk_dir = surface._normalize(-self._v[self._compute_state])
        fk_x = self.get_friction() * N * fk_dir

        glBegin(GL_LINES)
        glVertex3f(*self._s[0])
        glVertex3f(*Fg_y[0] + self._s[0])
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(*self._s[0])
        glVertex3f(*fk_x[0] + self._s[0])
        glEnd()

        Fnet_x = Fg_x + fk_x
        # glBegin(GL_LINES)
        # glVertex3f(*self._s[0])
        # glVertex3f(*Fnet_x[0] + self._s[0])
        # glEnd()

        a_x = Fnet_x / m

        # self._a[self._compute_state] = a_x
        self._v[self._compute_state] += a_x * dt
        self._s[self._compute_state] += self._v[self._compute_state] * dt

        self._s[self._compute_state, 2] = f_xy(*self._s[self._compute_state, :2].T)

    def get_render_positions(self):
        return self._s[self._compute_state]
