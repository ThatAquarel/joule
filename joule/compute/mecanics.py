import numpy as np

from joule.compute.calculus import CalculusEngine
from joule.compute.linalg import (
    column_wise,
    magnitude,
    normalize,
    vec_dot,
)


class MecanicsEngine:
    def __init__(self, initial_gravity, initial_friction, buffer_size=32):
        self._compute_state = np.zeros(buffer_size, dtype=bool)
        self._s, self._v = np.zeros((2, buffer_size, 3))
        self._m = np.zeros(buffer_size)

        self.set_gravity(initial_gravity)
        self.set_friction(initial_friction)

    def get_gravity(self):
        return self._gravity

    def set_gravity(self, gravity):
        self._gravity = np.array([0, 0, -gravity])

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

    def add_ball(self, position, mass):
        i = self._get_available_compute_spot()
        self._s[i] = position
        self._v[i] = 0
        self._m[i] = mass
        self._compute_state[i] = True

    def remove_ball(self, select_position):
        distances_to_select = np.linalg.norm(select_position - self._s, axis=1)
        i = np.argmin(distances_to_select)
        self._compute_state[i] = False

    def clear(self):
        self._compute_state[:] = False

    def update(self, dt, calculus_engine: CalculusEngine, z_correction=True):
        if not self._compute_state.sum():
            return

        pos = self._s[self._compute_state]
        vel = self._v[self._compute_state]

        point_mesh = pos[:, :2]
        normal = calculus_engine.build_normals(point_mesh)

        Fg_net = self.get_gravity()
        Z = normalize(normal)
        Fg_z = vec_dot(Fg_net, Z)
        Fg_x = Fg_net - Fg_z
        X = normalize(Fg_x)
        # Y = vec_cross(Z, X)

        vel[np.isinf(vel)] = 0
        vel_dir = normalize(vel)

        grad_mask = magnitude(vel_dir) != 0
        curvature = np.zeros(len(grad_mask))

        if grad_mask.sum():
            point_mesh_vel = vel_dir[grad_mask, :2]
            grad_1 = calculus_engine.build_gradient_first(point_mesh_vel)
            grad_2 = calculus_engine.build_gradient_second(point_mesh_vel)

            slope_1 = np.vecdot(grad_1, point_mesh_vel)
            slope_2 = np.vecdot(grad_2, point_mesh_vel)

            curvature[grad_mask] = np.abs(slope_2) / (1 + slope_1**2) ** (3 / 2)

        mass = self._m[self._compute_state]
        Fnet_z = column_wise(curvature * (magnitude(vel) ** 2) * mass) * Z
        N_z = Fnet_z - Fg_z
        N = magnitude(N_z)
        fk_xy = -vel_dir * column_wise(self.get_friction() * N)

        Fnet_xy = Fg_x + fk_xy

        a_z = Fnet_z / column_wise(mass)
        a_xy = Fnet_xy / column_wise(mass)
        a_net = a_z + a_xy

        self._v[self._compute_state] = a_net * dt + vel
        v_net = self._v[self._compute_state]
        self._s[self._compute_state] = v_net * dt + pos

        if z_correction:
            z = calculus_engine.build_values(point_mesh)
            self._s[self._compute_state, 2] = z

    def get_render_positions(self):
        return self._s[self._compute_state]

    def get_render_masses(self):
        return self._m[self._compute_state]

    def get_render_n(self):
        return self._compute_state.sum()

    def get_render_max(self):
        return len(self._compute_state)
