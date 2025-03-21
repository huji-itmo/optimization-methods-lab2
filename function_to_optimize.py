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

        left, right = self.optimization_range
        x = np.linspace(left, right, 1000)
        y = [self.func(xi) for xi in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Function")
        for i, d in enumerate(self.bisection_data):
            plt.axvspan(d["left"], d["right"], alpha=0.1, color="blue")
            plt.scatter([d["x1"], d["x2"]], [d["y1"], d["y2"]], color="red", s=20)
        final_x = (
            self.bisection_data[-1]["left"] + self.bisection_data[-1]["right"]
        ) / 2
        plt.scatter(
            final_x,
            self.func(final_x),
            color="green",
            marker="*",
            s=100,
            label="Result",
        )

        # Plot the final result
        final_x = (
            self.bisection_data[-1]["left"] + self.bisection_data[-1]["right"]
        ) / 2
        plt.scatter(final_x, 0, color="green", marker="*", s=100, label="Result")

        # Add labels, legend, and save the plot
        plt.title("Bisection Method Visualization")
        plt.xlabel("x")
        plt.ylabel("Derivative Value")
        plt.legend()
        plt.grid(True)
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

    import matplotlib.pyplot as plt

    def plot_newton(self) -> None:
        if not hasattr(self, "newton_data") or len(self.newton_data) == 0:
            raise ValueError(
                "No Newton iteration data found. Run newton() with debug=True first."
            )

        left, right = self.optimization_range
        x = np.linspace(left, right, 400)

        # Determine if we can plot the original function
        has_original_function = hasattr(self, "f") and callable(self.f)

        # Create figure and subplots
        if has_original_function:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 4))
            ax1 = None  # Placeholder for logic below

        # Plot original function if available
        if has_original_function:
            try:
                f_values = [self.f(xi) for xi in x]
                ax1.plot(x, f_values, label="$f(x)$", color="blue", linewidth=2)
                final_x = self.newton_data[-1]["x_new"]
                ax1.scatter(
                    final_x,
                    self.f(final_x),
                    color="red",
                    s=100,
                    zorder=5,
                    label="Optimal $x$",
                )
                ax1.set_title("Original Function $f(x)$ and Optimal Point", pad=20)
                ax1.legend(loc="upper right")
                ax1.grid(True, linestyle="--", alpha=0.7)
            except Exception as e:
                print(f"Warning: Failed to plot original function: {str(e)}")
                has_original_function = False

        # Plot derivative function
        df = getattr(self, "df", self._numerical_derivative)
        df_values = [df(xi) for xi in x]

        # Create second axis if needed
        if not has_original_function:
            ax2 = plt.gca()

        ax2.plot(x, df_values, label="$f'(x)$", color="green", linewidth=2)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
        ax2.set_title("Newton Iterations on Derivative $f'(x)$", pad=20)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Plot iteration steps
        for data in self.newton_data:
            x_curr = data["x"]
            dx = data["dx"]
            x_next = data["x_new"]

            # Current point on derivative curve
            ax2.scatter(x_curr, dx, color="blue", s=50, zorder=5, label="Current $x$")

            # Tangent line approximation
            tangent_x = np.array(
                [x_curr - 0.5, x_curr + 0.5]
            )  # Local region around x_curr
            tangent_y = dx + self._numerical_second_derivative(x_curr) * (
                tangent_x - x_curr
            )
            ax2.plot(
                tangent_x, tangent_y, "r--", alpha=0.7, linewidth=1.5, label="Tangent"
            )

            # Next approximation on x-axis
            ax2.scatter(
                x_next, 0, color="green", s=50, zorder=5, alpha=0.8, label="Next $x$"
            )

        # Final root marker
        final_x = self.newton_data[-1]["x_new"]
        ax2.scatter(
            final_x,
            0,
            color="red",
            s=100,
            zorder=10,
            label="Root $x^*$",
            edgecolor="black",
        )

        # Add legend with unique entries
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc="upper right")

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3 if has_original_function else 0.1)

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
