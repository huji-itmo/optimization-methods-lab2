from matplotlib import pyplot as plt
import numpy as np

from function_to_optimize import FunctionToOptimize


def plot_golden_section(optimizer: FunctionToOptimize):
    if not optimizer.golden_data:
        print("Run golden_section with debug=True first.")
        return

    left, right = optimizer.optimization_range
    x = np.linspace(left, right, 1000)
    y = [optimizer.func(xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Function")
    for i, d in enumerate(optimizer.golden_data):
        plt.axvspan(d["left"], d["right"], alpha=0.1, color="blue")
        plt.scatter([d["c"], d["d"]], [d["fc"], d["fd"]], color="red", s=20)
    final_x = (
        optimizer.golden_data[-1]["left"] + optimizer.golden_data[-1]["right"]
    ) / 2
    plt.scatter(
        final_x,
        optimizer.func(final_x),
        color="green",
        marker="*",
        s=100,
        label="Result",
    )
    plt.title("Golden Section Method")
    plt.legend()
    plt.savefig(optimizer.plot_path + "golden_section.pdf")
    plt.savefig(optimizer.plot_path + "golden_section.png")
    plt.close()


def plot_bisection(optimizer: FunctionToOptimize):
    if not optimizer.bisection_data:
        print("Run bisection with debug=True first.")
        return

    left, right = optimizer.optimization_range
    x = np.linspace(left, right, 1000)
    y = [optimizer.func(xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Function")
    for i, d in enumerate(optimizer.bisection_data):
        plt.axvspan(d["left"], d["right"], alpha=0.1, color="blue")
        plt.scatter([d["x1"], d["x2"]], [d["y1"], d["y2"]], color="red", s=20)
    final_x = (
        optimizer.bisection_data[-1]["left"] + optimizer.bisection_data[-1]["right"]
    ) / 2
    plt.scatter(
        final_x,
        optimizer.func(final_x),
        color="green",
        marker="*",
        s=100,
        label="Result",
    )

    # Plot the final result
    final_x = (
        optimizer.bisection_data[-1]["left"] + optimizer.bisection_data[-1]["right"]
    ) / 2
    plt.scatter(final_x, 0, color="green", marker="*", s=100, label="Result")

    # Add labels, legend, and save the plot
    plt.title("Bisection Method Visualization")
    plt.xlabel("x")
    plt.ylabel("Derivative Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(optimizer.plot_path + "bisection.pdf")
    plt.savefig(optimizer.plot_path + "bisection.png")
    plt.close()


def plot_chord(optimizer: FunctionToOptimize):
    if not optimizer.chord_data:
        print("Run chord method with debug=True first.")
        return

    plt.figure(figsize=(12, 7))

    # Plot derivative function
    x = np.linspace(*optimizer.optimization_range, 1000)
    dy = [optimizer._numerical_derivative(xi) for xi in x]
    plt.plot(x, dy, label="Function's Derivative", color="navy")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)

    # Plot chord lines and iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(optimizer.chord_data)))
    for i, (d, color) in enumerate(zip(optimizer.chord_data, colors)):
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
    final_x = optimizer.chord_data[-1]["mid"]
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
    if hasattr(optimizer, "plot_path"):
        plt.savefig(optimizer.plot_path + "chord.pdf", bbox_inches="tight")
        plt.savefig(optimizer.plot_path + "chord.png", bbox_inches="tight")

    plt.close()


def plot_newton(optimizer: FunctionToOptimize) -> None:
    if not hasattr(optimizer, "newton_data") or len(optimizer.newton_data) == 0:
        raise ValueError(
            "No Newton iteration data found. Run newton() with debug=True first."
        )

    left, right = optimizer.optimization_range
    x = np.linspace(left, right, 400)

    # Determine if we can plot the original function
    has_original_function = hasattr(optimizer, "f") and callable(optimizer.f)

    # Create figure and subplots with adjusted dimensions
    fig = None
    if has_original_function:
        try:
            # Test if we can actually compute f values
            _ = [optimizer.f(xi) for xi in x]
            # If successful, create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        except Exception as e:
            print(f"Warning: Original function cannot be plotted: {str(e)}")
            has_original_function = False
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            ax1 = None
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        ax1 = None

    # Plot original function if available
    if has_original_function:
        try:
            f_values = [optimizer.f(xi) for xi in x]
            ax1.plot(x, f_values, label="$f(x)$", color="blue", linewidth=2)
            final_x = optimizer.newton_data[-1]["x_new"]
            ax1.scatter(
                final_x,
                optimizer.f(final_x),
                color="red",
                s=100,
                zorder=5,
                label="Optimal $x$",
            )
            ax1.set_title("Original Function $f(x)$ and Optimal Point", pad=20)
            ax1.legend(loc="upper right")
            ax1.grid(True, linestyle="--", alpha=0.7)
        except Exception as e:
            print(f"Error: Failed to plot original function: {str(e)}")
            has_original_function = False
            # Recreate figure if original function failed after subplot creation
            plt.close(fig)
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            ax1 = None

    # Plot derivative function
    df = getattr(optimizer, "df", optimizer._numerical_derivative)
    df_values = [df(xi) for xi in x]

    # Plot to correct axis
    ax = ax2 if has_original_function else plt.gca()

    ax.plot(x, df_values, label="$f'(x)$", color="green", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_title("Newton Iterations on Derivative $f'(x)$", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Plot iteration steps
    for data in optimizer.newton_data:
        x_curr = data["x"]
        dx = data["dx"]
        x_next = data["x_new"]

        # Current point on derivative curve
        ax.scatter(x_curr, dx, color="blue", s=50, zorder=5, label="Current $x$")

        # Tangent line approximation
        tangent_x = np.array([x_curr - 0.5, x_curr + 0.5])
        tangent_y = dx + optimizer._numerical_second_derivative(x_curr) * (
            tangent_x - x_curr
        )
        ax.plot(tangent_x, tangent_y, "r--", alpha=0.7, linewidth=1.5, label="Tangent")

        # Next approximation on x-axis
        ax.scatter(
            x_next, 0, color="green", s=50, zorder=5, alpha=0.8, label="Next $x$"
        )

    # Final root marker
    final_x = optimizer.newton_data[-1]["x_new"]
    ax.scatter(
        final_x,
        0,
        color="red",
        s=100,
        zorder=10,
        label="Root $x^*$",
        edgecolor="black",
    )

    # Add legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Adjust layout
    plt.tight_layout()
    if has_original_function:
        plt.subplots_adjust(hspace=0.3)

    plt.savefig(optimizer.plot_path + "newton.pdf")
    plt.savefig(optimizer.plot_path + "newton.png")
    plt.close()


def plot_square_approx(optimizer: FunctionToOptimize):
    if not optimizer.square_approx_data:
        print("Run square_approx method first.")
        return

    data_list = optimizer.square_approx_data
    n_iters = len(data_list)

    # Determine subplot grid dimensions
    ncols = 1 if n_iters == 1 else 2  # Adjust columns based on iteration count
    nrows = int(np.ceil(n_iters / ncols))

    plt.figure(figsize=(10 * ncols, 6 * nrows))

    for i, data in enumerate(data_list, 1):
        plt.subplot(nrows, ncols, i)

        x_1 = data["x_1"]
        x_2 = data["x_2"]
        x_3 = data["x_3"]
        f_1 = data["f_1"]
        f_2 = data["f_2"]
        f_3 = data["f_3"]
        x_min = data["x_min"]
        f_min = data["f_min"]

        x_points = np.array([x_1, x_2, x_3])
        y_points = np.array([f_1, f_2, f_3])

        # Quadratic fit
        coeffs = np.polyfit(x_points, y_points, 2)
        x_fit = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
        y_fit = np.polyval(coeffs, x_fit)

        # Original function
        x_orig = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
        y_orig = [optimizer.func(x) for x in x_orig]

        plt.plot(x_orig, y_orig, label="Original Function", color="blue")
        plt.plot(x_fit, y_fit, "--", label="Quadratic Fit", color="orange")
        plt.scatter(x_points, y_points, color="red", s=80, label="Points", zorder=3)
        plt.scatter(
            x_min,
            f_min,
            color="green",
            marker="*",
            s=200,
            label="Estimated Min",
            zorder=4,
        )

        plt.title(f"Iteration {i}")
        plt.legend()
        plt.grid(True)

    # Hide any empty subplots
    for j in range(i + 1, nrows * ncols + 1):
        plt.subplot(nrows, ncols, j)
        plt.axis("off")

    plt.tight_layout()

    if hasattr(optimizer, "plot_path"):
        plt.savefig(optimizer.plot_path + "square_approx_iterations.pdf")
        plt.savefig(optimizer.plot_path + "square_approx_iterations.png")
    plt.close()
