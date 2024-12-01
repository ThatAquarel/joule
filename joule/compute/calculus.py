import numpy as np
import sympy as sp

from joule.graphics.elements.surface import Surface


class CalculusEngine:
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

        return d_ds_lambda, d_ds

    def get_f_xy(self):
        return self._f_xy

    def get_d_dx(self):
        return self._d_dx

    def get_d2_dx2(self):
        return self._d2_dx2

    def get_d_dy(self):
        return self._d_dy

    def get_d2_dy2(self):
        return self._d2_dy2

    def _parse_function(self, equation):
        x, y = sp.symbols("x y")

        allowed = {
            "x": x,
            "y": y,
            "sin": sp.sin,
            "asin": sp.asin,
            "cos": sp.cos,
            "acos": sp.acos,
            "sinh": sp.sinh,
            "asinh": sp.asinh,
            "cosh": sp.cosh,
            "acosh": sp.acosh,
            "tanh": sp.tanh,
            "atanh": sp.atanh,
            "pow": sp.Pow,
            "sqrt": sp.sqrt,
            "exp": sp.exp,
            "ln": sp.ln,
            "log": sp.log,
            "ceil": sp.ceiling,
            "floor": sp.floor,
        }

        f_xy = sp.sympify(
            equation,
            locals=allowed,
            rational=True,
        )

        return (x, y), f_xy

    def update_function(self, equation):
        x_y, f_xy = self._parse_function(equation)

        self.x_y, self.f_xy = x_y, f_xy

        self._f_xy = self._function_lambda(x_y, f_xy)
        x, y = x_y
        self._d_dx, d_dx = self._partial_derivative_lambda(x_y, f_xy, x)
        self._d2_dx2, _ = self._partial_derivative_lambda(x_y, d_dx, x)

        self._d_dy, d_dy = self._partial_derivative_lambda(x_y, f_xy, y)
        self._d2_dy2, _ = self._partial_derivative_lambda(x_y, d_dy, y)

        self.surface.update_function(
            self.get_f_xy(),
            self.get_d_dx(),
            self.get_d_dy(),
            [-4 * np.pi, 4 * np.pi],
            [-4 * np.pi, 4 * np.pi],
        )

    def draw(self):
        self.surface.draw()
