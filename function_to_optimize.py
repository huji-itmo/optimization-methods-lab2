from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np


class FunctionToOptimize:
    debug: bool = False  # Флаг отладки по умолчанию

    def __init__(
        self,
        func: Callable[[float], float],
        range: Tuple[float, float],
        debug: bool = False,
    ):
        self.func = func
        self.optimization_range = range
        self.debug = debug  # Позволяем устанавливать debug при создании объекта

        # Data storage for visualization
        self.golden_data = []
        self.bisection_data = []
        self.chord_data = []
        self.newton_data = []
        self.square_approx_data = []

    def _numerical_derivative(self, x: float, h: float = 1e-6) -> float:
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def _numerical_second_derivative(self, x: float, h: float = 1e-6) -> float:
        return (self.func(x + h) - 2 * self.func(x) + self.func(x - h)) / (h**2)

    def golden_section(
        self, epsilon: float = 1e-6, max_iter: int = 100
    ) -> Tuple[float, int]:
        if self.debug:
            print(f"\n=== Golden Section (ε={epsilon}, max_iter={max_iter}) ===")
            print(f"Start range: {self.optimization_range}")

        left, right = self.optimization_range

        x1 = left + 0.382 * (right - left)
        x2 = left + 0.618 * (right - left)
        f_x1, f_x2 = self.func(x1), self.func(x2)

        for it in range(max_iter):
            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  Left: {left:.6f}, Right: {right:.6f}")
                print(f"  x1: {x1:.6f} (f={f_x1:.6f}), x2: {x2:.6f} (f={f_x2:.6f})")
                self.golden_data.append(
                    {
                        "left": left,
                        "right": right,
                        "x1": x1,
                        "f_x1": f_x1,
                        "x2": x2,
                        "f_x2": f_x2,
                    }
                )

            if f_x1 < f_x2:
                right, x2, f_x2 = x2, x1, f_x1
                x1 = left + 0.382 * (right - left)
                f_x1 = self.func(x1)
            else:
                left, x1, f_x1 = x1, x2, f_x2
                x2 = left + 0.618 * (right - left)
                f_x2 = self.func(x2)

            if abs(right - left) < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return (left + right) / 2, it + 1

        if self.debug:
            print(f"\nReached max iterations ({max_iter})")
        return (left + right) / 2, max_iter

    def bisection(
        self,
        epsilon: float = 1e-6,
        max_iter: int = 100,
    ) -> Tuple[float, int]:
        if self.debug:
            print(f"\n=== Bisection (ε={epsilon}, max_iter={max_iter}) ===")
            print(f"Start range: {self.optimization_range}")

        left, right = self.optimization_range

        for it in range(max_iter):
            x1 = (left + right - epsilon) / 2
            x2 = (left + right + epsilon) / 2

            y1 = self.func(x1)
            y2 = self.func(x2)

            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  x1: {x1:.6f}, y1: {y1:.6f}")
                print(f"  x2: {x2:.6f}, y2: {y2:.6f}")
                print(f"  Current interval before update: [{left:.6f}, {right:.6f}]")
                self.bisection_data.append(
                    {
                        "left": left,
                        "right": right,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

            # Обновляем границы интервала
            if y1 > y2:
                left = x1
            else:
                right = x2

            if self.debug:
                print(f"  Updated interval: [{left:.6f}, {right:.6f}]")

            # Проверка условия остановки после обновления границ
            if (right - left) <= 2 * epsilon:
                xm = (left + right) / 2
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                    print(f"Final interval: [{left:.6f}, {right:.6f}], xm: {xm:.6f}")
                return xm, it + 1

        # Достигнуто максимальное количество итераций
        xm = (left + right) / 2
        if self.debug:
            print(f"\nReached max iterations ({max_iter})")
            print(f"Final interval: [{left:.6f}, {right:.6f}], xm: {xm:.6f}")
        return xm, max_iter

    def chord(
        self,
        epsilon: float = 1e-6,
        max_iter: int = 100,
        df: Optional[Callable[[float], float]] = None,
    ) -> Tuple[float, int]:
        if self.debug:
            print(f"\n=== Chord (ε={epsilon}, max_iter={max_iter}) ===")
            print(f"Start range: {self.optimization_range}")

        left, right = self.optimization_range
        df = df or self._numerical_derivative

        df_left = df(left)
        df_right = df(right)

        if self.debug:
            print(f"Initial df(left): {df_left:.6f}, df(right): {df_right:.6f}")

        if df_left * df_right >= 0:
            if self.debug:
                print("Error: Derivative does not change sign")
            raise ValueError("Производная не меняет знак на интервале")

        for it in range(max_iter):
            mid = left - df_left * (right - left) / (df_right - df_left)
            df_mid = df(mid)

            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  Left: {left:.6f} (df={df_left:.6f})")
                print(f"  Right: {right:.6f} (df={df_right:.6f})")
                print(f"  New point: {mid:.6f} (df={df_mid:.6f})")
                self.chord_data.append(
                    {
                        "left": left,
                        "df_left": df_left,
                        "right": right,
                        "df_right": df_right,
                        "mid": mid,
                        "df_mid": df_mid,
                    }
                )

            if abs(df_mid) < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return mid, it + 1

            if df_left * df_mid < 0:
                right, df_right = mid, df_mid
            else:
                left, df_left = mid, df_mid

        if self.debug:
            print(f"\nReached max iterations ({max_iter})")
        return mid, max_iter

    def newton(
        self,
        epsilon: float = 1e-6,
        max_iter: int = 100,
        x0: Optional[float] = None,
        df: Optional[Callable[[float], float]] = None,
        d2f: Optional[Callable[[float], float]] = None,
    ) -> Tuple[float, int]:
        if self.debug:
            print(f"\n=== Newton (ε={epsilon}, max_iter={max_iter}) ===")
            print(f"Start range: {self.optimization_range}")

        left, right = self.optimization_range
        x = x0 if x0 else (left + right) / 2
        df = df or self._numerical_derivative
        d2f = d2f or self._numerical_second_derivative

        df_left = df(left)
        df_right = df(right)

        if df_left * df_right >= 0:
            if self.debug:
                print("Error: Derivative does not change sign")
            raise ValueError("Производная не меняет знак на интервале")

        for it in range(max_iter):
            dx = df(x)
            d2x = d2f(x)

            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  x: {x:.6f}")
                print(f"  f'(x): {dx:.6f}, f''(x): {d2x:.6f}")

            if abs(dx) < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return x, it + 1

            if d2x == 0:
                if self.debug:
                    print("Error: Second derivative is zero")
                raise ValueError("Вторая производная равна нулю")

            x_new = x - dx / d2x
            x_new = max(min(x_new, right), left)  # Clamping to the range

            if self.debug:
                print(f"  New x: {x_new:.6f}")
                self.newton_data.append({"x": x, "dx": dx, "d2x": d2x, "x_new": x_new})

            if abs(x_new - x) < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return x_new, it + 1

            x = x_new

        if self.debug:
            print(f"\nReached max iterations ({max_iter})")
        return x, max_iter

    def square_approx(self, x_parabola, h, epsilon: float = 1e-6, max_iter: int = 10):

        print(f"\n=== square approximation, x_0={x_parabola}, h={h} ===")
        f = self.func

        f_0 = f(x_parabola)
        f_0_plus_h = f(x_parabola + h)

        print("  f(x_0) = ", f_0)
        print("  f(x_0 + h) = ", f_0_plus_h)

        x_1, x_2, x_3 = 0, 0, 0
        if f_0 < f_0_plus_h:
            x_1 = x_parabola - h
            x_2 = x_parabola
            x_3 = x_parabola + h
        else:
            x_1 = x_parabola
            x_2 = x_parabola + h
            x_3 = x_parabola + 2 * h

        for it in range(max_iter):
            print(f"\nIteration {it}:")

            print("  x_1 = ", x_1, " (f(x_1) = ", f(x_1), ")")
            print("  x_2 = ", x_2, " (f(x_2) = ", f(x_2), ")")
            print("  x_3 = ", x_3, " (f(x_3) = ", f(x_3), ")")

            F_min = min(f(x_1), f(x_2), f(x_3))
            print("  F_min = ", F_min)

            x_parabola = 0.5 * (x_1 + x_2) + 0.5 * (x_3 - x_1) * (x_3 - x_2) * (
                f(x_2) - f(x_1)
            ) / (f(x_1) * (x_2 - x_3) + f(x_2) * (x_3 - x_1) + f(x_3) * (x_1 - x_2))

            print("  f(x_parabola) = ", f(x_parabola))

            func = {f(x_1): x_1, f(x_2): x_2, f(x_3): x_3, f(x_parabola): x_parabola}
            sorted_x_mins = [
                func[x] for x in sorted([f(x_1), f(x_2), f(x_3), f(x_parabola)])
            ]
            print("  sorted_x_mins = ", sorted_x_mins)

            f_stop_criteria = abs((F_min - f(x_parabola)) / f(x_parabola))
            x_stop_criteria = abs((sorted_x_mins[0] - x_parabola) / x_parabola)

            print(
                "  abs((F_min - f_min) / f_min) = ",
                f_stop_criteria,
            )
            print(
                "  abs((x_2 - minimum_x_0) / minimum_x_0) = ",
                x_stop_criteria,
            )

            print("  x* = ", x_parabola)

            if self.debug:
                self.square_approx_data.append(
                    {
                        "x_1": x_1,
                        "x_2": x_2,
                        "x_3": x_3,
                        "f_1": f(x_1),
                        "f_2": f(x_2),
                        "f_3": f(x_3),
                        "x_min": x_parabola,
                        "f_min": f(x_parabola),
                    }
                )

            if f_stop_criteria < epsilon and x_stop_criteria < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return x_parabola, it + 1
            else:
                if x_parabola >= x_1 and x_parabola <= x_3:
                    best_x = sorted_x_mins[0]
                    sorted_points = sorted([x_1, x_2, x_3])

                    index = 0
                    for i in range(len(sorted_points)):
                        if best_x > sorted_points[i]:
                            index = i

                    if index >= 1:
                        sorted_points = sorted([best_x, x_2, x_3])
                    else:
                        sorted_points = sorted([best_x, x_1, x_2])

                    x_1, x_2, x_3 = (
                        sorted_points[0],
                        sorted_points[1],
                        sorted_points[2],
                    )

                else:
                    x_1, x_2, x_3 = x_2, x_2 + h, 0
                    f_0 = f(x_1)
                    f_0_plus_h = f(x_2)
                    if f_0 > f_0_plus_h:
                        x_3 = x_parabola + 2 * h
                    else:
                        x_3 = x_parabola - h

                    sorted_points = sorted([x_1, x_2, x_3])
                    x_1, x_2, x_3 = sorted_points[0], sorted_points[1], sorted_points[2]

        return x_parabola, max_iter
