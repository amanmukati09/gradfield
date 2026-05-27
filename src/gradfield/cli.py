"""
cli.py — Command-line interface for gradfield.
"""

import argparse
import sys
import numpy as np


NAMED_FUNCTIONS = {
    "rosenbrock": (
        lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2,
        (-2.0, 2.0), (-1.0, 3.0),
    ),
    "himmelblau": (
        lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
        (-5.0, 5.0), (-5.0, 5.0),
    ),
    "rastrigin": (
        lambda x, y: 20 + x**2 - 10 * np.cos(2 * np.pi * x)
                       + y**2 - 10 * np.cos(2 * np.pi * y),
        (-5.12, 5.12), (-5.12, 5.12),
    ),
    "sphere": (
        lambda x, y: x**2 + y**2,
        (-3.0, 3.0), (-3.0, 3.0),
    ),
    "beale": (
        lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2
                     + (2.625 - x + x*y**3)**2,
        (-4.5, 4.5), (-4.5, 4.5),
    ),
    "booth": (
        lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2,
        (-10.0, 10.0), (-10.0, 10.0),
    ),
    "matyas": (
        lambda x, y: 0.26 * (x**2 + y**2) - 0.48 * x * y,
        (-10.0, 10.0), (-10.0, 10.0),
    ),
}


def parse_expr(expr: str):
    """Safely parse a math expression string into a callable f(x, y)."""
    allowed = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi, "e": np.e,
    }
    def fn(x, y):
        return eval(expr, {"__builtins__": {}}, {**allowed, "x": x, "y": y})
    return fn


def cmd_plot(args):
    from gradfield import sample_grid, render

    expr = args.function.strip().lower()

    if expr in NAMED_FUNCTIONS:
        fn, default_x, default_y = NAMED_FUNCTIONS[expr]
        x_range = tuple(args.xrange) if args.xrange else default_x
        y_range = tuple(args.yrange) if args.yrange else default_y
        title = expr.capitalize()
    else:
        fn = parse_expr(args.function)
        x_range = tuple(args.xrange) if args.xrange else (-3.0, 3.0)
        y_range = tuple(args.yrange) if args.yrange else (-3.0, 3.0)
        title = args.function

    print(f"[gradfield] Sampling '{title}' at {args.resolution}x{args.resolution}...")
    field = sample_grid(fn, x_range=x_range, y_range=y_range, resolution=args.resolution)

    print("[gradfield] Rendering...")
    render(
        field,
        title=title,
        output=args.output,
        colorscale=args.colorscale,
        show_gradients=not args.no_gradients,
        open_browser=not args.no_browser,
    )


def cmd_list(args):
    print("\nBuilt-in functions you can plot:\n")
    for name in NAMED_FUNCTIONS:
        print(f"  gradfield plot {name}")
    print("\nOr pass any math expression:")
    print('  gradfield plot "x**2 + y**2"')
    print('  gradfield plot "sin(x) * cos(y)"\n')


def main():
    parser = argparse.ArgumentParser(
        prog="gradfield",
        description="Gradient field & loss landscape visualizer",
    )
    subparsers = parser.add_subparsers(dest="command")

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot a function or named landscape")
    plot_parser.add_argument("function", help='Function name or math expression e.g. "rosenbrock" or "x**2+y**2"')
    plot_parser.add_argument("--resolution", "-r", type=int, default=60, help="Grid resolution (default: 60)")
    plot_parser.add_argument("--xrange", "-x", type=float, nargs=2, metavar=("MIN", "MAX"), help="X axis range")
    plot_parser.add_argument("--yrange", "-y", type=float, nargs=2, metavar=("MIN", "MAX"), help="Y axis range")
    plot_parser.add_argument("--output", "-o", default="gradfield_output.html", help="Output HTML file")
    plot_parser.add_argument("--colorscale", "-c", default="Viridis", help="Plotly colorscale (default: Viridis)")
    plot_parser.add_argument("--no-gradients", action="store_true", help="Hide gradient arrows")
    plot_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # list subcommand
    subparsers.add_parser("list", help="List all built-in named functions")

    args = parser.parse_args()

    if args.command == "plot":
        cmd_plot(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()