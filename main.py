import math
from typing import Tuple
from function_to_optimize import FunctionToOptimize
from plotter import (
    plot_bisection,
    plot_chord,
    plot_golden_section,
    plot_newton,
    plot_square_approx,
)


def func(x: float) -> float:
    return 0.25 * x**4 + x**2 - 8 * x + 12


def func_derivative(x: float) -> float:
    return x**3 + 2 * x - 8


def func_second_derivative(x: float) -> float:
    return 3 * x**2 + 2


func_range: Tuple[float, float] = (0, 2)
epsilon: float = 0.05


if __name__ == "__main__":
    optimizer = FunctionToOptimize(func, func_range, True)
    optimizer.plot_path = "output/"
    df_left, df_right = func_derivative(func_range[0]), func_derivative(func_range[1])

    if df_left * df_right >= 0:
        if optimizer.debug:
            print("Error: Derivative does not change sign")
        raise ValueError("Производная не меняет знак на интервале")

    try:
        optimizer.bisection(epsilon=epsilon)
        plot_bisection(optimizer)
    except ValueError as e:
        print(e)
    try:

        optimizer.chord(epsilon=epsilon, df=func_derivative)
        plot_chord(optimizer)

    except ValueError as e:
        print(e)
    try:

        mn, it = optimizer.golden_section(epsilon=epsilon)
        print("min = ", mn)
        plot_golden_section(optimizer)

    except ValueError as e:
        print(e)
    try:

        optimizer.newton(
            epsilon=epsilon, x0=1, df=func_derivative, d2f=func_second_derivative
        )
        plot_newton(optimizer)

    except ValueError as e:
        print(e)
    try:

        optimizer.square_approx(
            (func_range[0] + func_range[1]) / 2,
            (func_range[0] + func_range[1]) / 4,
            epsilon=epsilon,
        )
        plot_square_approx(optimizer)

    except ValueError as e:
        print(e)
