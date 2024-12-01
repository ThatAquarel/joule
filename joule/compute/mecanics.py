import sympy as sp
import numpy as np

from OpenGL.GL import *

from joule.compute.linalg import (
    column_wise,
    get_basis,
    magnitude,
    normalize,
    vec_cross,
    vec_dot,
)


class MecanicsEngine:
    def __init__(self, buffer_size=16):
        self._compute_state = np.zeros(buffer_size, dtype=bool)
        self._m, self._s, self._v, self._a = np.zeros((4, buffer_size, 3))

        # self._gravity = np.array([0, 0, -9.81])
        self._gravity = np.array([0, 0, -25])
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

    def update(
        self,
        pos_3d,
        calculus_engine,
        dt,
        f_xy,
        d_dx,
        d_dy,
        d2_dx2,
        d2_dy2,
        surface,
    ):
        if self._compute_state.sum() == 0:
            return

        (x, y), f = calculus_engine.x_y, calculus_engine.f_xy
        fx, fy = sp.diff(f, x), sp.diff(f, y)
        fxx, fyy = sp.diff(fx, x), sp.diff(fy, y)
        fxy = sp.diff(fx, y)

        # s = np.array([pos_3d])

        s = self._s[self._compute_state]
        xy = s[:, :2]

        normal, (d_dx_mesh, d_dx_vec, d_dy_mesh, d_dy_vec) = surface._build_normals(
            d_dx, d_dy, xy, return_derivative_vector=True
        )

        Fg_net = self.get_gravity()

        Z = normalize(normal)
        Fg_z = vec_dot(Fg_net, Z)
        Fg_x = Fg_net - Fg_z
        X = normalize(Fg_x)
        Y = vec_cross(Z, X)

        curvatures = []
        for i, n in enumerate(np.where(self._compute_state)[0]):
            I, J, K = get_basis()
            try:
                T = np.linalg.solve([I, J, K], [X[i], Y[i], Z[i]])
                J_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                curvatures.append(0)
                continue
                T = np.eye(3)
                J_inv = np.eye(3)

            # u = np.array([1.0, 1.0, 1.0])
            u = self._v[n]
            # u = normal[i]

            replace = {x: xy[i, 0], y: xy[i, 1]}

            gradient0 = np.array([fx.subs(replace), fy.subs(replace)])
            # u = u @ T
            d = np.linalg.norm(u)
            if d == 0:
                u = np.zeros(3)
            else:
                u = u / d

            u = u[:2]

            # hessian = np.array(
            #     [
            #         [fxx.subs(replace), fxy.subs(replace)],
            #         [fxy.subs(replace), fyy.subs(replace)],
            #     ]
            # )
            # gradient1 = np.dot(hessian, u_[:2])

            gradient1 = np.array(
                [
                    fxx.subs(replace) * u[0] + fxy.subs(replace) * u[1],
                    fyy.subs(replace) * u[1] + fxy.subs(replace) * u[0],
                ]
            )

            # slope0 = np.dot([*gradient0, 0], u_)
            # slope1 = np.dot([*gradient1, 0], u_)

            slope0 = np.dot(gradient0, u)
            slope1 = np.dot(gradient1, u)

            # k = slope1 / sp.Pow(1 + slope0**2, sp.Rational(3 / 2))
            k = slope1 / (1 + slope0 * slope0) ** (3 / 2)

            # curvatures.append(k.evalf())
            # curvatures.append(np.abs(k))
            curvatures.append(k)
        curvatures = np.array(curvatures, dtype=np.float64)

        m = self._m[self._compute_state]
        vel_xy = self._v[self._compute_state]
        vel_dir = normalize(self._v[self._compute_state])

        Fnet_z = Z * m * column_wise(curvatures * (magnitude(vel_xy) ** 2))
        N_z = Fnet_z - Fg_z
        N = magnitude(N_z)

        # flying = np.isclose(N, 0)
        # N[flying] = 0
        # N_z[flying] = 0
        # Fnet_z = N_z + Fg_z

        fk_xy = -vel_dir * column_wise(self.get_friction() * N)

        Fnet_xy = Fg_x + fk_xy

        a_z = Fnet_z / m
        a_xy = Fnet_xy / m

        a_net = a_z + a_xy

        self._v[self._compute_state] = a_net * dt + self._v[self._compute_state]
        v_net = self._v[self._compute_state]
        self._s[self._compute_state] = v_net * dt + self._s[self._compute_state]
        self._s[self._compute_state, 2] = f_xy(*xy.T)

        s = self._s[self._compute_state]

        # glBegin(GL_LINES)
        # for i, n in enumerate(np.where(self._compute_state)[0]):
        #     glVertex3f(*s[i])
        #     # glVertex3f(*s[i] + (Fnet_xy + Fnet_z)[i])
        #     glVertex3f(*s[i] + N_z[i])

        #     # glVertex3f(*s[i])
        #     # glVertex3f(*s[i] + Y[i])

        #     # glVertex3f(*s[i])
        #     # glVertex3f(*s[i] + Z[i])

        # glEnd()

        # glBegin(GL_LINES)
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + normal[0])
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + Fg_x[0])
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + [*gradient0, 0])
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + [*gradient1, 0])
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + R[0])
        # glEnd()

        # s = np.array([[5, 5, 5]])
        # glBegin(GL_LINES)
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + Fg_z[0] @ J_inv)
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + Fg_x[0] @ J_inv)
        # glVertex3f(*s[0])
        # glVertex3f(*s[0] + Y[0] @ J_inv)
        # glEnd()

    def get_render_positions(self):
        return self._s[self._compute_state]
