from typing import Tuple
from function_to_optimize import FunctionToOptimize


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

    optimizer.bisection(epsilon=epsilon, df=func_derivative)
    optimizer.chord(epsilon=epsilon, df=func_derivative)
    optimizer.golden_section(epsilon=epsilon)
    optimizer.newton(
        epsilon=epsilon, x0=1, df=func_derivative, d2f=func_second_derivative
    )
    optimizer.square_approx(
        (func_range[0] + func_range[1]) / 2, (func_range[0] + func_range[1]) / 4
    )


# пук пук среньк пук пук среньк пук пук среньк скибиди доп доп доп ес ес скибиди дабл ю дип дип скибиди доп доп доп ес ес скибиди скибиди скибиди скибиди
