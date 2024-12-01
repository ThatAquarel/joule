import sympy as sp
import numpy as np

from OpenGL.GL import *


class MecanicsEngine:
    def __init__(self, buffer_size=16):
        self._compute_state = np.zeros(buffer_size, dtype=bool)
        self._m, self._s, self._v, self._a = np.zeros((4, buffer_size, 3))

        # self._gravity = np.array([0, 0, -9.81])
        self._gravity = np.array([0, 0, -25])
        self._friction = 0.05

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

    def update(self, calculus_engine, dt, f_xy, d_dx, d_dy, d2_dx2, d2_dy2, surface):
        if self._compute_state.sum() == 0:
            return

        m = self._m[self._compute_state]
        v = self._v[self._compute_state]
        xy = self._s[self._compute_state, :2]

        normal, tangent = surface._build_normals(
            d_dx, d_dy, xy, return_derivative_vector=True
        )

        sign = -np.sign(tangent[:, -1])
        tangent_downward = tangent * sign[:, np.newaxis]

        Fg = self.get_gravity()
        # Fg_y = np.vecdot(Fg, normal)[:, np.newaxis] * normal
        Fg_x = np.vecdot(Fg, tangent_downward)[:, np.newaxis] * tangent_downward
        # N = np.linalg.norm(Fg_y)
        # fk_dir = surface._normalize(-self._v[self._compute_state])
        # fk_x = self.get_friction() * N * fk_dir

        ################################
        ################################
        ################################

        (x, y), f = calculus_engine.x_y, calculus_engine.f_xy
        fx, fy = sp.diff(f, x), sp.diff(f, y)
        fxx, fyy = sp.diff(fx, x), sp.diff(fy, y)
        fxy = sp.diff(fx, y)

        curvature = []
        for u, (x0, y0) in zip(surface._normalize(v), xy):
            u = u[:2]
            replace = {x: x0, y: y0}

            gradient0 = np.array([fx.subs(replace), fy.subs(replace)])

            slope0 = np.dot(gradient0, u)
            slope1 = (
                fxx.subs(replace) * u[0] ** 2
                + 2 * fxy.subs(replace) * u[0] * u[1]
                + fyy.subs(replace) * u[1] ** 2
            )

            k = slope1 / sp.Pow(1 + slope0**2, sp.Rational(3 / 2))
            curvature.append(k)

        curvature = np.nan_to_num(np.abs(curvature).astype(np.float64))

        Fnet_y = (
            (np.linalg.norm(v, axis=1) ** 2 * curvature)[:, np.newaxis] * normal * m
        )

        N_y = Fnet_y - np.vecdot(Fg, normal)[:, np.newaxis] * normal
        N = np.linalg.norm(N_y, axis=1)
        fk_dir = surface._normalize(-self._v[self._compute_state])
        fk_x = self.get_friction() * N * fk_dir
        Fnet_x = Fg_x + fk_x

        a_x = Fnet_x / m
        a_y = Fnet_y / m

        glBegin(GL_LINES)
        glVertex3f(*self._s[0])
        glVertex3f(*self._s[0] + a_x[0])
        glVertex3f(*self._s[0])
        glVertex3f(*self._s[0] + a_y[0])
        glEnd()

        a = a_x + a_y

        self._v[self._compute_state] += a * dt
        self._s[self._compute_state] += self._v[self._compute_state] * dt

        self._s[self._compute_state, 2] = f_xy(*self._s[self._compute_state, :2].T)

    def get_render_positions(self):
        return self._s[self._compute_state]
