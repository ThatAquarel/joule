import numpy as np
import sympy as sp

from joule.graphics.elements.surface import Surface

from joule.mechanics.parse import parse_function


class GraphEngine:
    def __init__(self, res=1024):
        self.surface = Surface(res=res)

    def _function_lambda(self, variables, function):
        lambified = sp.lambdify(variables, function, "numpy")

        def constant_safe(*values):
            evaluated, _ = np.broadcast_arrays(lambified(*values), values[0])
            return evaluated

        return constant_safe

    def _partial_derivative_lambda(
        self,
        variables,
        function,
        to_differentiate,
    ):
        d_ds = sp.diff(function, to_differentiate)
        d_ds_lambda = self._function_lambda(variables, d_ds)

        return d_ds_lambda

    def update_function(self, equation):
        x_y, f_xy = parse_function(equation)

        self._f_xy = self._function_lambda(x_y, f_xy)
        x, y = x_y
        self._d_dx = self._partial_derivative_lambda(x_y, f_xy, x)
        self._d_dy = self._partial_derivative_lambda(x_y, f_xy, y)

        self.surface.update_function(
            self._f_xy,
            self._d_dx,
            self._d_dy,
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        )

    def draw(self):
        self.surface.draw()
