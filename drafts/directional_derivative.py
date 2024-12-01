import sympy as sp
import numpy as np


x, y, z = sp.symbols("x y z")

f = -sp.sqrt(1 - x * x - y * y)

fx, fy = sp.diff(f, x), sp.diff(f, y)
fxx, fyy = sp.diff(fx, x), sp.diff(fy, y)
fxy = sp.diff(fx, y)
# fyx = sp.diff(fy, x)
# fxy_ = sp.diff(f, x, y)
# fyx_ = sp.diff(f, y, x)

...


def test(x0, y0, v):
    u = v / sp.sqrt(v[0] ** 2 + v[1] ** 2)

    replace = {x: x0, y: y0}

    gradient0 = np.array([fx.subs(replace), fy.subs(replace)])
    gradient1 = np.array(
        [
            fxx.subs(replace) * u[0] + fxy.subs(replace) * u[1],
            fyy.subs(replace) * u[1] + fxy.subs(replace) * u[0],
        ]
    )

    hessian = np.array(
        [
            [fxx.subs(replace), fxy.subs(replace)],
            [fxy.subs(replace), fyy.subs(replace)],
        ]
    )
    gradient1 = np.dot(hessian, u)

    slope0 = np.dot(gradient0, u)
    slope1 = np.dot(gradient1, u)
    # slope1 = np.dot(, u)

    slope1_ = (
        fxx.subs(replace) * u[0] ** 2
        + 2 * fxy.subs(replace) * u[0] * u[1]
        + fyy.subs(replace) * u[1] ** 2
    )

    print()
    print()
    print(f"test at point {(x0, y0)} with direction {v}")

    print()
    print(f"slope0: {slope0}")
    print(f"slope1: {slope1}")
    print(f"slope1 with expansion: {slope1_}")

    k = slope1 / sp.Pow(1 + slope0**2, sp.Rational(3 / 2))
    print(f"curvature {k}")
    print(f"radius {1/k}")
    print()


test(
    sp.sqrt(2) / 4,
    sp.sqrt(2) / 4,
    np.array([sp.Rational(1, 1), sp.Rational(1, 1)]),
)

test(
    -sp.sqrt(2) / 4,
    -sp.sqrt(2) / 4,
    np.array([sp.Rational(-1, 1), sp.Rational(-1, 1)]),
)

test(
    sp.Rational(1 / 2),
    0,
    np.array([sp.Rational(1, 1), sp.Rational(0, 1)]),
)

test(
    sp.Rational(-1 / 2),
    0,
    np.array([sp.Rational(-1, 1), sp.Rational(0, 1)]),
)

test(
    0,
    sp.Rational(1 / 2),
    np.array([sp.Rational(0, 1), sp.Rational(1, 1)]),
)

test(
    0,
    sp.Rational(-1 / 2),
    np.array([sp.Rational(0, 1), sp.Rational(-1, 1)]),
)


# https://en.wikipedia.org/wiki/Directional_derivative
# https://en.wikipedia.org/wiki/Radius_of_curvature
# https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/directional-derivative-introduction
# https://math.stackexchange.com/questions/4750978/second-order-directional-derivative-better-understending
