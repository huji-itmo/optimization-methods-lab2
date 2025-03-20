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
        self.square_data = {}

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
        df = self._numerical_derivative
        df_left = df(left)
        df_right = df(right)

        if df_left * df_right >= 0:
            if self.debug:
                print("Error: Derivative does not change sign")
            raise ValueError("Производная не меняет знак на интервале")

        phi = (1 + 5**0.5) / 2

        c = right - (right - left) / phi
        d = left + (right - left) / phi
        fc, fd = self.func(c), self.func(d)

        for it in range(max_iter):
            if self.debug:
                print(f"\nIteration {it}:")
                print(f"  Left: {left:.6f}, Right: {right:.6f}")
                print(f"  c: {c:.6f} (f={fc:.6f}), d: {d:.6f} (f={fd:.6f})")
                self.golden_data.append(
                    {"left": left, "right": right, "c": c, "fc": fc, "d": d, "fd": fd}
                )

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
                self.bisection_data.append(
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

    def square_approx(self, initial_x_0, h):
        print(f"\n=== square approximation, x_0={initial_x_0}, h={h} ===")
        f = self.func
        f_0 = f(initial_x_0)
        f_0_plus_h = f(initial_x_0 + h)

        x_1, x_2, x_3 = 0, 0, 0
        if f_0 < f_0_plus_h:
            x_1 = initial_x_0 - h
            x_2 = initial_x_0
            x_3 = initial_x_0 + h
        else:
            x_1 = initial_x_0 - 2 * h
            x_2 = initial_x_0
            x_3 = initial_x_0 + 2 * h

        x_min = (x_1 + x_2) * 0.5 + 0.5 * (x_3 - x_1) * (x_3 - x_2) * (
            f(x_2) - f(x_1)
        ) / (f(x_1) * (x_2 - x_3) + f(x_2) * (x_3 - x_1) + f(x_3) * (x_1 - x_2))
        print(f"\nresult: {x_min}")

        if self.debug:
            self.square_approx_data = {
                "x_1": x_1,
                "x_2": x_2,
                "x_3": x_3,
                "f_1": f(x_1),
                "f_2": f(x_2),
                "f_3": f(x_3),
                "x_min": x_min,
                "f_min": f(x_min),
            }

        return x_min

    # Existing methods (_numerical_derivative, golden_section, bisection, chord, newton, square_approx)
    # are retained with data collection added in debug mode as shown in the thought process.

    def plot_golden_section(self):
        if not self.golden_data:
            print("Run golden_section with debug=True first.")
            return

        left, right = self.optimization_range
        x = np.linspace(left, right, 1000)
        y = [self.func(xi) for xi in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Function")
        for i, d in enumerate(self.golden_data):
            plt.axvspan(d["left"], d["right"], alpha=0.1, color="blue")
            plt.scatter([d["c"], d["d"]], [d["fc"], d["fd"]], color="red", s=20)
        final_x = (self.golden_data[-1]["left"] + self.golden_data[-1]["right"]) / 2
        plt.scatter(
            final_x,
            self.func(final_x),
            color="green",
            marker="*",
            s=100,
            label="Result",
        )
        plt.title("Golden Section Method")
        plt.legend()
        plt.savefig(self.plot_path + "golden_section.pdf")
        plt.savefig(self.plot_path + "golden_section.png")
        plt.close()

    def plot_bisection(self):
        if not self.bisection_data:
            print("Run bisection with debug=True first.")
            return

        x = np.linspace(*self.optimization_range, 1000)
        dy = [self._numerical_derivative(xi) for xi in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, dy, label="Derivative")
        plt.axhline(0, color="black", linestyle="--")
        for d in self.bisection_data:
            plt.axvspan(d["left"], d["right"], alpha=0.1, color="blue")
            plt.scatter(d["mid"], d["df_mid"], color="red", s=20)
        final_x = (
            self.bisection_data[-1]["left"] + self.bisection_data[-1]["right"]
        ) / 2
        plt.scatter(final_x, 0, color="green", marker="*", s=100, label="Result")
        plt.title("Bisection Method")
        plt.legend()
        plt.savefig(self.plot_path + "bisection.pdf")
        plt.savefig(self.plot_path + "bisection.png")
        plt.close()

    def plot_chord(self):
        if not self.chord_data:
            print("Run chord method with debug=True first.")
            return

        plt.figure(figsize=(12, 7))

        # Plot derivative function
        x = np.linspace(*self.optimization_range, 1000)
        dy = [self._numerical_derivative(xi) for xi in x]
        plt.plot(x, dy, label="Function's Derivative", color="navy")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)

        # Plot chord lines and iterations
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.chord_data)))
        for i, (d, color) in enumerate(zip(self.chord_data, colors)):
            # Draw chord line between current endpoints
            plt.plot(
                [d["left"], d["right"]],
                [d["df_left"], d["df_right"]],
                "--",
                color=color,
                alpha=0.7,
                label=f"Chord {i+1}" if i < 3 else None,
            )

            # Draw intersection with zero line
            plt.plot(
                [d["mid"], d["mid"]],
                [0, d["df_mid"]],
                color=color,
                linestyle=":",
                alpha=0.5,
            )

            # Annotate iterations
            plt.annotate(
                f"Iter {i+1}\nx={d['mid']:.3f}\nf'={d['df_mid']:.3f}",
                xy=(d["mid"], 0),
                xytext=(10, 20 * (i % 2 - 0.5)),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=color),
                color=color,
            )

        # Highlight final result
        final_x = self.chord_data[-1]["mid"]
        plt.scatter(
            final_x, 0, color="red", marker="*", s=200, zorder=5, label="Final Result"
        )

        # Formatting
        plt.title("Chord Method Visualization\n(Zero Finding via Secant Method)")
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)

        # Save if path exists
        if hasattr(self, "plot_path"):
            plt.savefig(self.plot_path + "chord.pdf", bbox_inches="tight")
            plt.savefig(self.plot_path + "chord.png", bbox_inches="tight")

        plt.close()

    def plot_newton(self):
        if not self.newton_data:
            print("Run newton method with debug=True first.")
            return

        x = np.linspace(*self.optimization_range, 1000)
        y = [self.func(xi) for xi in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Function")
        for d in self.newton_data:
            plt.scatter(d["x"], self.func(d["x"]), color="red", s=20)
            plt.plot(
                [d["x"], d["x_new"]],
                [self.func(d["x"]), self.func(d["x_new"])],
                "--",
                color="blue",
                alpha=0.5,
            )
        final_x = self.newton_data[-1]["x_new"]
        plt.scatter(
            final_x,
            self.func(final_x),
            color="green",
            marker="*",
            s=100,
            label="Result",
        )
        plt.title("Newton's Method")
        plt.legend()
        plt.savefig(self.plot_path + "newton.pdf")
        plt.savefig(self.plot_path + "newton.png")
        plt.close()

    def plot_square_approx(self):
        if not self.square_approx_data:
            print("Run square_approx method first.")
            return

        d = self.square_approx_data
        x_points = np.array([d["x_1"], d["x_2"], d["x_3"]])
        y_points = np.array([d["f_1"], d["f_2"], d["f_3"]])

        # Create fit data
        coeffs = np.polyfit(x_points, y_points, 2)
        x_fit = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
        y_fit = np.polyval(coeffs, x_fit)

        # Create original function data
        x_orig = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
        y_orig = [self.func(xi) for xi in x_orig]

        plt.figure(figsize=(10, 6))

        # Plot original function
        plt.plot(x_orig, y_orig, label="Original Function", color="blue")

        # Plot quadratic approximation
        plt.plot(x_fit, y_fit, "--", label="Quadratic Fit", color="orange")

        # Plot initial points and minimum
        plt.scatter(
            x_points, y_points, color="red", s=80, label="Initial Points", zorder=3
        )
        plt.scatter(
            d["x_min"],
            d["f_min"],
            color="green",
            marker="*",
            s=200,
            label="Estimated Minimum",
            zorder=4,
        )

        plt.title("Square Approximation Method")
        plt.legend()
        plt.grid(True)

        # Only save if plot_path exists in the object
        if hasattr(self, "plot_path"):
            plt.savefig(self.plot_path + "square_approx.pdf")
            plt.savefig(self.plot_path + "square_approx.png")
        plt.close()
