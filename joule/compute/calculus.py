import numpy as np
import sympy as sp

from joule.compute.linalg import normalize


class CalculusEngine:
    def __init__(self):
        """
        Calculus Engine: Handling all math computations
        of application, and differentiation of functions

        :return: CalculusEngine instance
        """

        # create sympy symbols x and y for functions
        self._x, self._y = sp.symbols("x y")

    def _function_lambda(self, variables, function):
        """
        Turn sympy symbolic representation of function
        into executable lambda that can be evaluated at
        different points

        :param variables: Symbols implicated in function
        :param function: Sympy symbolic function
        :return: Lambda representation of function
        """

        # turn into lambda compatible with numpy
        lambified = sp.lambdify(variables, function, "numpy")

        # lambdify is not safe when function is a constant
        # that does not involve variables
        def constant_safe(*values):
            """
            Constant-safe lambdify wrapper
            :param *values: Variables of functions
            """

            # broadcast result over initial shape of values
            evaluated, _ = np.broadcast_arrays(lambified(*values), values[0])
            return evaluated

        # return wrapper function
        return constant_safe

    def _partial_derivative_lambda(
        self,
        variables,
        function,
        to_differentiate,
    ):
        """
        Compute symbolic partial derivative of function
        and turn into executable lambda that can be
        evaluated at different points

        :param variables: Symbols implicated in function
        :param function: Sympy symbolic function
        :param to_differentiate: Array of variables to differentiate
        :return: Lambda representation of partial derivative
        """

        # differentiate symbolically with respect to variables
        d_ds = sp.diff(function, *to_differentiate)

        # symbolically simplify expression
        d_ds = sp.simplify(d_ds)

        # turn into executable lambda
        d_ds_lambda = self._function_lambda(variables, d_ds)

        return d_ds_lambda, d_ds

    @property
    def x(self):
        """
        Returns x symbol

        :return: Symbolic variable x
        """

        return self._x

    @property
    def y(self):
        """
        Returns y symbol

        :return: Symbolic variable y
        """

        return self._y

    def get_function(self, symbolic=False):
        """
        Returns base function

        :param symbolic: Symbolic or lambda representation
        :return: Internal math function
        """

        if symbolic:
            return self._f
        return self._f_l

    def get_partial(self, variable, order, symbolic=False):
        """
        Returns partial derivative function

        :param variable: Variable differentiated with respect to
        :param order: Order of partial derivative
        :param symbolic: Symbolic or lambda representation
        :return: Internal math function partial derivative
        """

        # bound the order of derivatives
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

        # bounds the varialbe
        if (partials := partials_map.get(variable)) is None:
            raise ValueError(f"Unknown variable: {variable}")

        return partials[order - 1]

    def get_mixed_partial(self, symbolic=False):
        """
        Returns mixed partial derivative function

        :param symbolic: Symbolic or lambda representation
        :return: Internal math function mixed derivative
        """

        if symbolic:
            return self._fxy
        return self._fxy_l

    def _tangent_vec(self, derivative_values, axis):
        """
        Computes tangent vectors to surface given values
        of derivatives and axis of tangent

        :param derivative_values: Computed values of derivatives of shape (n,)
        :param axis: 0 is x, 1 is y
        :return: Tangent vectors of shape (n, 3)
        """

        # preallocate tangent vectors
        vec = np.zeros((*derivative_values.shape, 3))

        # derivative is slope=rise/run
        # rise = d/ds
        vec[:, 2] = derivative_values
        # run = 1
        vec[:, axis] = 1

        return vec

    def build_normals(self, point_mesh):
        """
        Computes normal vectors to surface given points
        at which normals are evaluated

        :param point_mesh: Array of points of shape (n, 2)
        :return: Normal vectors of shape (n, 3)
        """

        # computes derivative values at points
        fx_val, fy_val = self._fx_l(*point_mesh.T), self._fy_l(*point_mesh.T)

        # build vectors tangent to surface with
        # respect to x and y
        fx_vec, fy_vec = self._tangent_vec(fx_val, 0), self._tangent_vec(fy_val, 1)

        # normal is orthogonal to both tangent vectors
        # therefore, cross product by right hand rule
        normals = np.cross(fx_vec, fy_vec)

        # normalized unitary normals
        return normalize(normals)

    def build_values(self, point_mesh):
        """
        Evaluates base function at given points

        :param point_mesh: Array of points of shape (n, 2)
        :return: Value of function of shape (n,)
        """

        # compute z = f(x, y) for all points
        return self._f_l(*point_mesh.T)

    def _build_hessian(self, point_mesh):
        """
        Computes Hessian matrices at given points

        :param point_mesh: Array of points of shape (n, 2)
        :return: Hessian matrices at points of shape (n, 2, 2)
        """
        # the formula from
        # https://en.wikipedia.org/wiki/Hessian_matrix
        # and my understanding built from
        # https://math.stackexchange.com/questions/4750978/second-order-directional-derivative-better-understending

        # calculates partial derivatives at points
        fxx_val, fyy_val = self._fxx_l(*point_mesh.T), self._fyy_l(*point_mesh.T)
        fxy_val = self._fxy_l(*point_mesh.T)

        # build hessian matrices of shape (2, 2, n)
        hessian = np.array([[fxx_val, fxy_val], [fxy_val, fyy_val]])

        # transpose matrix to have shape (n, 2, 2)
        return hessian.T

    def build_gradient_first(self, point_mesh):
        """
        Computes first order gradient vectors at given points

        :param point_mesh: Array of points of shape (n, 2)
        :return: Array of gradients at points of shape (n, 2)
        """

        # evaluates first order derivates at given points
        fx_val, fy_val = self._fx_l(*point_mesh.T), self._fy_l(*point_mesh.T)

        # build gradient vectors of shape (2, n)
        gradient = np.array([fx_val, fy_val])

        # transpose matrix to have shape (n, 2)
        return gradient.T

    def build_gradient_second(self, point_mesh):
        """
        Computes second order gradient vectors at given points

        :param point_mesh: Array of points of shape (n, 2)
        :return: Array of gradients at points of shape (n, 2)
        """

        # evaluates second order hessian matrices
        # for gradient computation
        hessian = self._build_hessian(point_mesh)

        # dots the points with their respective hessians
        # acquire second order gradient vectors
        return np.matmul(hessian, point_mesh[:, :, np.newaxis]).squeeze(-1)

    def _parse_function(self, text):
        """
        Parse a function given as text into its symbolic
        equivalent

        :param text: Textual expression of function
        :return: Symbolic function
        """

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
        }

        return sp.sympify(
            text,
            locals=allowed,
            rational=True,
        )

    def update_function(self, equation):
        """
        Updates internal base function and derivatives

        :param equation: Textual expression of function
        :return: Parser message
        """

        symbols = self.x, self.y

        # catch all possible exceptions thrown by parser
        # not best practice, but I didn't have time to
        # look through the documentation to find which
        # specific ones sp.sympify can throw
        try:
            self._f = self._parse_function(equation)
        except Exception as e:
            return f"Parsing failed:\n{str(e)}"

        # compute symbolic and lambda equivalent
        # of base function and its derivatives

        try:
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
        except Exception as e:
            return f"Derivation failed:\n{str(e)}"

        return "Parsed sucessfully"

    def pretty_print(self, function):
        """
        Returns a string of a symbolic function

        :param function: Symbolic function
        :return: Pretty string
        """

        return sp.pretty(function, use_unicode=False)
