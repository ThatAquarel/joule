import sympy as sp


def parse_function(equation):
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
