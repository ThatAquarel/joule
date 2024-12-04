import numpy as np

from joule.compute.calculus import CalculusEngine
from joule.compute.linalg import (
    column_wise,
    magnitude,
    normalize,
    vec_dot,
)


class MechanicsEngine:
    def __init__(self, initial_gravity, initial_friction, buffer_size=32):
        """
        Mechanics Engine: Handling all physics computations
        of application, and Euler integration for ball positions

        :param initial_gravity: Initial gravity (m/s^2)
        :param initial_friction: Initial friction (kinetic)
        :param buffer_size: Physics computation buffer size

        :return: MechanicsEngine instance
        """
        # preallocate buffers for physics computation
        # to optimize and vectorize all calculations
        self._buffer_increment = buffer_size

        # boolean mask of indices where computation is needed
        # True: calculates physics, False: does not
        self._compute_state = np.zeros(buffer_size, dtype=bool)

        # s: position buffer (m)
        # v: velocity buffer (m/s)
        self._s, self._v = np.zeros((2, buffer_size, 3))

        # m: masses (kg)
        self._m = np.zeros(buffer_size)

        self._gravity = initial_gravity
        self._friction = initial_friction

    def get_gravity(self):
        """
        Returns gravity

        :return: gravity (m/s^2)
        """

        return self._gravity

    def set_gravity(self, gravity):
        """
        Sets internal gravity vector

        :param: gravity (m/s^2)
        """

        # convert to internal vec3
        self._gravity = np.array([0, 0, -gravity])

    def get_friction(self):
        """
        Returns kinetic friction

        :return: friction
        """(0,0 -g)

        return self._friction

    def set_friction(self, friction):
        """
        Sets internal friction constant

        :param: friction (kinetic)
        """

        self._friction = friction

    def _get_available_compute_spot(self):
        """
        Acquire index of the first free location in
        computation buffer to accelerate computation

        Reallocates buffer if full to prevent overflow

        :return: index < buffer_size
        """

        # get first location where buffer is False
        # ie, not currently used
        i = np.argmin(self._compute_state)

        # if location is True
        # implies: every spot in buffer is used because
        # argmin could not find a minimum
        if self._compute_state[i]:
            # increment new buffer size
            old_size = len(self._compute_state)
            new_size = old_size + self._buffer_increment

            print(f"mechanics: reallocate, from {old_size} to {new_size}")

            # reallocate compute buffer state
            compute_state = np.zeros(new_size, dtype=bool)
            # copy old values into new buffer
            compute_state[:old_size] = self._compute_state
            self._compute_state = compute_state

            # reallocate position, velocity and masses
            (s, v), m = np.zeros((2, new_size, 3)), np.zeros(new_size)
            # copy old values into new buffer
            s[:old_size], v[:old_size], m[:old_size] = self._s, self._v, self._m
            self._s, self._v, self._m = s, v, m

            # returns the first available spot
            # which is the one after the last index
            # of the previous buffer
            i = old_size

        return i

    def add_ball(self, position, mass):
        """
        Adds ball with mass at given position into
        compute buffer

        :param position: Vector of position (m)
        :param mass: Scalar of mass (kg)
        """

        i = self._get_available_compute_spot()

        # (re)set parameters
        self._s[i] = position
        self._v[i] = 0
        self._m[i] = mass

        # turn on computation at index
        self._compute_state[i] = True

    def remove_ball(self, select_position):
        """
        Removes ball closest to given position
        from compute buffer

        :param select_position: Vector of position (m)
        """

        # compute distances from select position
        # each of the balls
        distances_to_select = np.linalg.norm(select_position - self._s, axis=1)

        # find minimum distance (closest)
        i = np.argmin(distances_to_select)

        # deactivate computation at index
        self._compute_state[i] = False


    def clear(self):
        """
        Frees all of compute buffer
        """

        # stop computation for all indices
        self._compute_state[:] = False

    def update(self, dt, calculus_engine: CalculusEngine, z_correction=True):
        """
        Step through Euler integration for dt

        :param dt: Time delta to integrate
        :param calculus_engine: Instance of joule.calculus.CalculusEngine
        :param z_correction: Correct for vertical deviation over time
        """

        # sum returns the number of True values
        # in compute state
        # if no computation is required, skip
        if not self._compute_state.sum():
            return

        # acquire position and velocity of indices
        # that need to be computed
        pos = self._s[self._compute_state]
        vel = self._v[self._compute_state]

        # isolate x and y of position
        point_mesh = pos[:, :2]
        # build normals at x and y
        normal = calculus_engine.build_normals(point_mesh)

        # build reference frame of the ball
        Z = normalize(normal)
        # X = normalize(Fg_x)
        # Y = vec_cross(Z, X)
        
        # project vertical component of gravity
        Fg_net = self.get_gravity()
        Fg_z = vec_dot(Fg_net, Z)

        # acquire horizontal component of gravity
        Fg_x = Fg_net - Fg_z

        # bugfix: safen velocity computation by
        # setting infinite velocities back to zero
        # for numerical stability
        vel[np.isinf(vel)] = 0

        # find direction of velocity
        vel_dir = normalize(vel)

        # mask for gradient computation if velocity
        # is non-zero for numerical stability
        grad_mask = magnitude(vel_dir) != 0

        # preallocate buffer for curvature computation
        curvature = np.zeros(len(grad_mask))

        # if gradient is needed to be computed
        if grad_mask.sum():
            # isolate x and y of velocity
            point_mesh_vel = vel_dir[grad_mask, :2]

            # compute first order gradient of surface
            grad_1 = calculus_engine.build_gradient_first(point_mesh_vel)

            # compute second order gradient of surface
            grad_2 = calculus_engine.build_gradient_second(point_mesh_vel)

            # project first and second order gradients
            # onto velocity direction, effectively
            # calculating the directional gradient that
            # is aligned to velocity
            slope_1 = np.vecdot(grad_1, point_mesh_vel)
            slope_2 = np.vecdot(grad_2, point_mesh_vel)

            # calculate curvature according to directional
            # derivatives
            curvature[grad_mask] = np.abs(slope_2) / (1 + slope_1**2) ** (3 / 2)

        # acquire masses of indices
        # that need to be computed
        mass = self._m[self._compute_state]

        # calculates radial net force
        # curvature: k = 1/r
        # radial acceleration: a = V^2/r
        #                        = V^2 * k
        # radial net force: F*a
        Fnet_z = column_wise(curvature * (magnitude(vel) ** 2) * mass) * Z

        # calculates normal force of surface
        N_z = Fnet_z - Fg_z
        N = magnitude(N_z)

        # using normal force, calculate friction vector
        # with direction opposite to velocity
        fk_xy = -vel_dir * column_wise(self.get_friction() * N)

        # sum of forces horizontal
        Fnet_xy = Fg_x + fk_xy

        a_z = Fnet_z / column_wise(mass)
        a_xy = Fnet_xy / column_wise(mass)

        # sum of accelerations
        a_net = a_z + a_xy

        # integrate acceleration with respect to time
        # to get velocity
        self._v[self._compute_state] = a_net * dt + vel
        v_net = self._v[self._compute_state]

        # integrate velocity with respect to time
        # to get position
        self._s[self._compute_state] = v_net * dt + pos

        # if vertical integration correction is activated
        if z_correction:
            # computes actual z position of balls
            z = calculus_engine.build_values(point_mesh)

            # sets z position of balls to surface
            self._s[self._compute_state, 2] = z

    def get_render_positions(self):
        """
        Returns positions where physics is computed

        :return: Position vectors of shape (n, 3)
        """

        return self._s[self._compute_state]

    def get_render_masses(self):
        """
        Returns masses where physics is computed

        :return: Masses of shape (n,)
        """

        return self._m[self._compute_state]

    def get_render_n(self):
        """
        Returns current number of balls for which
        computation is ran on

        :return: Number of balls
        """

        return self._compute_state.sum()

    def get_render_max(self):
        """
        Returns length of computation buffer:
        maximum number of balls until reallocation

        :return: Buffer size
        """

        return len(self._compute_state)
