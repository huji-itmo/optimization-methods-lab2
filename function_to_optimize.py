from typing import Tuple, Callable, Optional


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
        phi = (1 + 5**0.5) / 2

        c = right - (right - left) / phi
        d = left + (right - left) / phi
        fc, fd = self.func(c), self.func(d)

        for it in range(max_iter):
            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  Left: {left:.6f}, Right: {right:.6f}")
                print(f"  c: {c:.6f} (f={fc:.6f}), d: {d:.6f} (f={fd:.6f})")

            if fc < fd:
                right, d, fd = d, c, fc
                c = right - (right - left) / phi
                fc = self.func(c)
            else:
                left, c, fc = c, d, fd
                d = left + (right - left) / phi
                fd = self.func(d)

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
        df: Optional[Callable[[float], float]] = None,
    ) -> Tuple[float, int]:
        if self.debug:
            print(f"\n=== Bisection (ε={epsilon}, max_iter={max_iter}) ===")
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
            mid = (left + right) / 2
            df_mid = df(mid)

            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  Left: {left:.6f} (df={df_left:.6f})")
                print(f"  Right: {right:.6f} (df={df_right:.6f})")
                print(f"  Mid: {mid:.6f} (df={df_mid:.6f})")

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
        return (left + right) / 2, max_iter

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

            if abs(x_new - x) < epsilon:
                if self.debug:
                    print(f"\nConverged after {it+1} iterations")
                return x_new, it + 1

            x = x_new

        if self.debug:
            print(f"\nReached max iterations ({max_iter})")
        return x, max_iter
