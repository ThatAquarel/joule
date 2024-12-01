import sympy as sp
import numpy as np


x, y, z = sp.symbols("x y z")

#####################################
# 2D Curve, solve for curvature k of f(x)
# fx = -sp.sqrt(1 - x * x)
fx = -x * x
d_dx_1 = sp.diff(fx)
d2_dx2_1 = sp.diff(d_dx_1)

k_1 = d2_dx2_1 / sp.Pow(1 + sp.Pow(d_dx_1, 2), sp.Rational(3, 2))

...
#####################################
# Tangential sphere method
# fxy = -sp.sqrt(1 - x * x - y * y)
f = x * x + y * y
fx, fy = sp.diff(f, x), sp.diff(f, y)
fxx, fyy = sp.diff(fx, x), sp.diff(fy, y)
fxy = sp.diff(f, x, y)

k_x = fxx / sp.Pow(1 + sp.Pow(fx, 2), sp.Rational(3, 2))
k_y = fyy / sp.Pow(1 + sp.Pow(fy, 2), sp.Rational(3, 2))

...

#####################################
# ChatGPT formula
# k = (d2_dx2 * d2_dy2 - sp.Pow(d2_dxdy, 2)) / sp.Abs(
#     sp.Pow(1 + sp.Pow(d_dx, 2) + sp.Pow(d_dy, 2), sp.Rational(3, 2))
# )

#####################################
# Plugging in nubmers method

# s = np.array([1.0, 0.0, 0.0])
# v = np.array([0.0, 0.0, -1.0])
# n = np.array([-1.0, 0.0, 0.0])

# s = np.array([np.sqrt(2), np.sqrt(2), 0.0])
# v = np.array([0.0, 0.0, -1.0])
# n = np.array([-np.sqrt(2), -np.sqrt(2), 0.0])

# s = np.array([1.23995972, -1.14563215, -0.08867139])
# v = np.array([0.64945313, -0.60486052, 0.46081926])
# n = np.array([-0.49152452, 0.5277616, 0.69272745])

# n0 = np.cross(v, n)
# plane_eqn = sp.Eq(
#     n0[0] * x + n0[1] * y + n0[2] * z, n0[0] * s[0] + n0[1] * s[1] + n0[2] * s[2]
# )

##########################################
##### DIRECTIONAL DERIVATIVES METHOD #####
##########################################

f = sp.sin(x * y)

x0, y0 = sp.pi / 3, sp.Rational(1, 2)
v = np.array([sp.Rational(2, 1), sp.Rational(3, 1)])

u = v / sp.sqrt(v[0] ** 2 + v[1] ** 2)

fx, fy = sp.diff(f, x), sp.diff(f, y)
replace = {x: x0, y: y0}
gradient = np.array([fx.subs(replace), fy.subs(replace)])

slope = np.dot(gradient, u)

...

f = -sp.sqrt(1 - x * x - y * y)

x0, y0 = sp.Rational(1, 2), sp.Rational(1, 2)
v = np.array([sp.Rational(1, 1), sp.Rational(1, 1)])

u = v / sp.sqrt(v[0] ** 2 + v[1] ** 2)

fx, fy = sp.diff(f, x), sp.diff(f, y)
replace = {x: x0, y: y0}
gradient = np.array([fx.subs(replace), fy.subs(replace)])

slope = np.dot(gradient, u)

...

f = -sp.sqrt(1 - x * x - y * y)

# x0, y0 = sp.Rational(1, 2), sp.Rational(1, 2)
# v = np.array([sp.Rational(1, 1), sp.Rational(1, 1)])

# x0, y0 = 0, 0
# v = np.array([sp.Rational(1, 1), sp.Rational(1, 1)])

x0, y0 = sp.sqrt(2) / 4, sp.sqrt(2) / 4
v = np.array([sp.Rational(1, 1), sp.Rational(1, 1)])

# x0, y0 = sp.Rational(1 / 2), 0
# v = np.array([sp.Rational(1, 1), sp.Rational(0, 1)])

# x0, y0 = sp.sqrt(3) / 2, 0
# v = np.array([sp.Rational(1, 1), sp.Rational(0, 1)])

u = v / sp.sqrt(v[0] ** 2 + v[1] ** 2)
# print(u)

fx, fy = sp.diff(f, x), sp.diff(f, y)
fxx, fyy = sp.diff(fx, x), sp.diff(fy, y)
replace = {x: x0, y: y0}
gradient0 = np.array([fx.subs(replace), fy.subs(replace)])
gradient1 = np.array([fxx.subs(replace), fyy.subs(replace)])

slope0 = np.dot(gradient0, u)
slope1 = np.dot(gradient1, u)

print(slope0)
print(slope1)

k = slope1 / sp.Pow(1 + slope0**2, sp.Rational(3 / 2))
print(k)

# https://en.wikipedia.org/wiki/Directional_derivative
# https://en.wikipedia.org/wiki/Radius_of_curvature
# https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/directional-derivative-introduction
# https://math.stackexchange.com/questions/4750978/second-order-directional-derivative-better-understending
