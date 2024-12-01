import numpy as np
import sympy as sp

from joule.compute.linalg import normalize
from joule.graphics.elements.surface import Surface


class CalculusEngine:
    def __init__(self):
        self._x, self._y = sp.symbols("x y")

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
        d_ds = sp.diff(function, *to_differentiate)
        d_ds = sp.simplify(d_ds)
        d_ds_lambda = self._function_lambda(variables, d_ds)

        return d_ds_lambda, d_ds

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def get_function(self, symbolic=False):
        if symbolic:
            return self._f
        return self._f_l

    def get_partial(self, variable, order, symbolic=False):
        if order not in [1, 2]:
            raise NotImplementedError(f"Partial {order}th derivative not computed")

        if symbolic:
            partials_map = {
                self.x: [self._fx, self._fxx],
                self.y: [self._fy, self._fyy],
            }
        else:
            partials_map = {
                self.x: [self._fx_l, self._fxx_l],
                self.y: [self._fy_l, self._fyy_l],
            }

        if (partials := partials_map.get(variable)) is None:
            raise ValueError(f"Unknown variable: {variable}")

        return partials[order - 1]

    def get_mixed_partial(self, symbolic=False):
        if symbolic:
            return self._fxy
        return self._fxy_l

    def _tangent_vec(self, derivative_values, axis):
        vec = np.zeros((*derivative_values.shape, 3))
        vec[:, 2] = derivative_values
        vec[:, axis] = 1

        return vec

    def build_normals(self, point_mesh):
        fx_val, fy_val = self._fx_l(*point_mesh.T), self._fy_l(*point_mesh.T)
        fx_vec, fy_vec = self._tangent_vec(fx_val, 0), self._tangent_vec(fy_val, 1)

        normals = np.cross(fx_vec, fy_vec)

        return normalize(normals)

    def build_values(self, point_mesh):
        return self._f_l(*point_mesh.T)

    def _build_hessian(self, point_mesh):
        fxx_val, fyy_val = self._fxx_l(*point_mesh.T), self._fyy_l(*point_mesh.T)
        fxy_val = self._fxy_l(*point_mesh.T)

        hessian = np.array([[fxx_val, fxy_val], [fxy_val, fyy_val]])

        return hessian.T

    def build_gradient_first(self, point_mesh):
        fx_val, fy_val = self._fx_l(*point_mesh.T), self._fy_l(*point_mesh.T)
        gradient = np.array([fx_val, fy_val])

        return gradient.T

    def build_gradient_second(self, point_mesh):
        hessian = self._build_hessian(point_mesh)

        return np.matmul(hessian, point_mesh[:, :, np.newaxis]).squeeze(-1)

    def _parse_function(self, text):
        allowed = {
            "x": self.x,
            "y": self.y,
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

        return sp.sympify(
            text,
            locals=allowed,
            rational=True,
        )

    def update_function(self, equation):
        symbols = self.x, self.y

        self._f = self._parse_function(equation)
        self._f_l = self._function_lambda(symbols, self._f)

        self._fx_l, self._fx = self._partial_derivative_lambda(
            symbols,
            self._f,
            [self.x],
        )
        self._fxx_l, self._fxx = self._partial_derivative_lambda(
            symbols,
            self._fx,
            [self.x],
        )

        self._fy_l, self._fy = self._partial_derivative_lambda(
            symbols,
            self._f,
            [self.y],
        )
        self._fyy_l, self._fyy = self._partial_derivative_lambda(
            symbols,
            self._fy,
            [self.y],
        )

        self._fxy_l, self._fxy = self._partial_derivative_lambda(
            symbols, self._fy, [self.x, self.y]
        )
