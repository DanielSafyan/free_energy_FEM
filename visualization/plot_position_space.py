import argparse
import os
from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _read_series(h5_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read ball and paddle Y-series from HDF5.

    Returns (ball_y, paddle_y, dt)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ball = f["game/ball_pos"]  # (T,2)
        plat = f["game/platform_pos"]  # (T,)
        T = int(min(ball.shape[0], plat.shape[0]))
        if T < 1:
            raise ValueError("HDF5 contains no timesteps.")
        by = ball[:T, 1].astype(float)
        py = plat[:T].astype(float)
        dt = float(f.attrs.get("dt", 0.0))
    return by, py, dt


def _compute_indices(n: int, start_frac: float, until_frac: float) -> np.ndarray:
    """
    Map fractional [start, until] to index range [start_idx:end_idx) and return arange indices.
    Follows the same semantics as visualization/pong_replay.py
    """
    if not (0.0 <= start_frac <= 1.0) or not (0.0 <= until_frac <= 1.0):
        raise ValueError("--from and --until must be between 0 and 1 (inclusive)")
    if start_frac >= until_frac:
        raise ValueError("--from must be less than --until")

    start_idx = int(np.floor(start_frac * n))
    end_idx = int(np.ceil(until_frac * n))
    start_idx = max(0, min(start_idx, n))
    end_idx = max(0, min(end_idx, n))
    if end_idx - start_idx <= 0:
        raise ValueError("Slicing range is empty after applying --from/--until")

    return np.arange(start_idx, end_idx, dtype=int)


def plot_position_space(h5_path: str,
                        start_frac: float = 0.0,
                        until_frac: float = 1.0,
                        output: str | None = os.path.join("output", "pong_position_space.png"),
                        dpi: int = 200,
                        show: bool = False,
                        paddle_height: float = 200.0,
                        color_by_xpos: bool = False) -> None:
    """
    Plot paddle_y vs ball_y for the selected time range (fraction of the full series).
    """
    by, py, dt = _read_series(h5_path)
    # Convert saved top-of-paddle Y to center Y using half the paddle height
    py_center = py + float(paddle_height) / 2.0

    n = int(min(len(by), len(py)))
    indices = _compute_indices(n, start_frac, until_frac)

    # Extract selected range
    sel_by = by[indices]
    sel_py = py_center[indices]

    # Build scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    # Choose color mapping: by time (default) or by ball X position when requested
    if color_by_xpos:
        with h5py.File(h5_path, "r") as f:
            bx_full = f["game/ball_pos"][:n, 0].astype(float)
        colors = bx_full[indices]
        cb_label = "Ball X [px]"
    else:
        colors = np.linspace(0.0, 1.0, num=indices.size)
        cb_label = "Relative time in selection"
    sc = ax.scatter(sel_py, sel_by, c=colors, s=8, cmap="viridis", alpha=0.6, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, label=cb_label)

    # y = x reference line (same coordinate frame)
    minv = float(min(sel_py.min(), sel_by.min()))
    maxv = float(max(sel_py.max(), sel_by.max()))
    ax.plot([minv, maxv], [minv, maxv], "k--", lw=1, alpha=0.6, label="y = x")

    ax.set_xlabel("Paddle Y (center) [px]")
    ax.set_ylabel("Ball Y [px]")
    ax.set_title("Position-space: paddle_y vs ball_y")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    fig.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}")

    if show:
        try:
            plt.show()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Plot paddle_y vs ball_y from Pong HDF5 log.")
    parser.add_argument("-i", "--h5", default=os.path.join("output", "pong_simulation.h5"),
                        help="Path to HDF5 file (default: output/pong_simulation.h5)")
    parser.add_argument("--from", dest="start_frac", type=float, default=0.0,
                        help="Start fraction in [0,1] of the series (default: 0.0)")
    parser.add_argument("--until", dest="until_frac", type=float, default=1.0,
                        help="End fraction in [0,1] of the series (default: 1.0)")
    parser.add_argument("--output", default=os.path.join("output", "pong_position_space.png"),
                        help="Output PNG filename (default: output/pong_position_space.png)")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI (default: 200)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument("--paddle-height", type=float, default=200.0,
                        help="Paddle height in pixels; used to convert top to center (default: 200)")
    parser.add_argument("--xpos", action="store_true",
                        help="Color scatter points by ball X coordinate instead of relative time")

    args = parser.parse_args()

    plot_position_space(
        h5_path=args.h5,
        start_frac=args.start_frac,
        until_frac=args.until_frac,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
        paddle_height=args.paddle_height,
        color_by_xpos=args.xpos,
    )


if __name__ == "__main__":
    main()
