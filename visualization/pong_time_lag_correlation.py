import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import h5py

# We avoid importing PongH5Reader to prevent pygame dependency in headless environments.


def compute_time_lag_correlation(h5_path: str, horizon: int = 200, use_npen: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute time-lag Pearson correlation between paddle Y and ball Y.

    Returns (lags, corr, dt)
    - lags: integer lags in steps, from -horizon to +horizon
    - corr: correlation coefficient for each lag
    - dt: timestep size in seconds (0.0 if unavailable)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ball_ds = f["game/ball_pos"]  # shape (T, 2)
        plat_ds = f["game/platform_pos"]  # shape (T,)
        T = int(min(ball_ds.shape[0], plat_ds.shape[0]))
        if T < 2:
            raise ValueError("Not enough timesteps to compute correlation (need >= 2).")
        by = ball_ds[:T, 1].astype(float)
        py = plat_ds[:T].astype(float)
        dt = float(f.attrs.get("dt", 0.0))

    H = int(max(0, horizon))
    H = min(H, T - 1)
    lags = np.arange(-H, H + 1, dtype=int)
    corr = np.zeros_like(lags, dtype=float)

    for idx, k in enumerate(lags):
        if k >= 0:
            a = py[0:T - k]
            b = by[k:T]
        else:
            a = py[-k:T]
            b = by[0:T + k]
        # Pearson correlation on overlapping window
        a_mean = a.mean()
        b_mean = b.mean()
        a_std = a.std()
        b_std = b.std()
        if a_std == 0.0 or b_std == 0.0:
            corr[idx] = 0.0
        else:
            corr[idx] = float(np.dot(a - a_mean, b - b_mean) / (len(a) * a_std * b_std))

    return lags, corr, dt


def load_series(h5_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load full paddle Y and ball Y series from the HDF5 file, trimmed to equal length.
    Returns (py, by, dt).
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ball_ds = f["game/ball_pos"]  # shape (T, 2)
        plat_ds = f["game/platform_pos"]  # shape (T,)
        T = int(min(ball_ds.shape[0], plat_ds.shape[0]))
        if T < 2:
            raise ValueError("Not enough timesteps to compute correlation (need >= 2).")
        by = ball_ds[:T, 1].astype(float)
        py = plat_ds[:T].astype(float)
        dt = float(f.attrs.get("dt", 0.0))

    return py, by, dt


def compute_corr_series(py: np.ndarray, by: np.ndarray, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lagged Pearson correlation for given 1D series py, by and horizon H.
    Returns (lags, corr).
    """
    T = int(min(len(py), len(by)))
    if T < 1:
        raise ValueError("Empty time series provided.")
    H = int(max(0, min(H, T - 1)))

    lags = np.arange(-H, H + 1, dtype=int)
    corr = np.zeros_like(lags, dtype=float)

    for idx, k in enumerate(lags):
        if k >= 0:
            a = py[0:T - k]
            b = by[k:T]
        else:
            a = py[-k:T]
            b = by[0:T + k]
        a_mean = a.mean()
        b_mean = b.mean()
        a_std = a.std()
        b_std = b.std()
        if a_std == 0.0 or b_std == 0.0:
            corr[idx] = 0.0
        else:
            corr[idx] = float(np.dot(a - a_mean, b - b_mean) / (len(a) * a_std * b_std))

    return lags, corr


def plot_time_lag_correlation(lags: np.ndarray, corr: np.ndarray, dt: float, output: str | None = None):
    """
    Plot correlation vs lag and optionally save to file.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, corr, 'b-', lw=2)
    ax.axvline(0, color='k', lw=1, alpha=0.6)

    # Annotate best lag
    best_idx = int(np.nanargmax(np.abs(corr)))
    best_lag = int(lags[best_idx])
    best_val = float(corr[best_idx])
    ax.plot([best_lag], [best_val], 'ro')
    ax.annotate(f"lag={best_lag} steps\nρ={best_val:.3f}",
                xy=(best_lag, best_val), xytext=(10, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'), color='red')

    ax.set_xlabel('Lag (steps)')
    ax.set_title('Time-lag correlation: paddle_y vs ball_y')
    ax.set_ylabel('Pearson correlation')
    ax.grid(True, alpha=0.3)

    # Lag is measured and displayed in steps only (no seconds axis)

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    plt.show()


def plot_time_lag_correlation_multi(lags: np.ndarray, corrs: list[np.ndarray], dt: float, output: str | None = None, labels: list[str] | None = None):
    """
    Plot multiple correlation curves (one per time partition) on a single figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    N = len(corrs)
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    if labels is None:
        labels = [f"part {i+1}" for i in range(N)]

    for i, c in enumerate(corrs):
        ax.plot(lags, c, lw=2, color=colors[i], label=labels[i])

    ax.axvline(0, color='k', lw=1, alpha=0.6)
    ax.set_xlabel('Lag (steps)')
    ax.set_title(f'Time-lag correlation: paddle_y vs ball_y ({N} partitions)')
    ax.set_ylabel('Pearson correlation')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Partitions', ncol=min(4, N))

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    plt.show()


def compute_time_lag_correlation_2d(h5_path: str, horizon: int = 200, use_npen: bool = False, xbins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a 2D time-lag Pearson correlation heatmap conditioned on ball X.

    Returns (lags, x_centers, corr2d, dt)
    - lags: integer lags in steps, from -horizon to +horizon
    - x_centers: centers of the ball X bins
    - corr2d: array shape (len(x_centers), len(lags)) with Pearson r per (x_bin, lag)
    - dt: timestep size in seconds (0.0 if unavailable)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ball_ds = f["game/ball_pos"]  # shape (T, 2)
        plat_ds = f["game/platform_pos"]  # shape (T,)
        T = int(min(ball_ds.shape[0], plat_ds.shape[0]))
        if T < 2:
            raise ValueError("Not enough timesteps to compute correlation (need >= 2).")
        bx = ball_ds[:T, 0].astype(float)
        by = ball_ds[:T, 1].astype(float)
        py = plat_ds[:T].astype(float)
        dt = float(f.attrs.get("dt", 0.0))

    H = int(max(0, horizon))
    H = min(H, T - 1)
    lags = np.arange(-H, H + 1, dtype=int)

    xbins = int(max(1, xbins))
    x_min = float(np.nanmin(bx))
    x_max = float(np.nanmax(bx))
    if x_max == x_min:
        x_max = x_min + 1.0
    edges = np.linspace(x_min, x_max, xbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    corr2d = np.full((xbins, len(lags)), np.nan, dtype=float)

    for li, k in enumerate(lags):
        if k >= 0:
            a = py[0:T - k]
            b = by[k:T]
            x = bx[k:T]
        else:
            a = py[-k:T]
            b = by[0:T + k]
            x = bx[0:T + k]

        # Bin indices for this lag slice
        bin_idx = np.digitize(x, edges) - 1  # 0..xbins-1
        # Ensure rightmost edge is included in last bin
        bin_idx = np.clip(bin_idx, 0, xbins - 1)

        for bi in range(xbins):
            sel = (bin_idx == bi)
            n = int(sel.sum())
            if n < 3:
                corr2d[bi, li] = np.nan
                continue
            a_sub = a[sel]
            b_sub = b[sel]
            a_std = a_sub.std()
            b_std = b_sub.std()
            if a_std == 0.0 or b_std == 0.0:
                corr2d[bi, li] = 0.0
            else:
                corr2d[bi, li] = float(np.dot(a_sub - a_sub.mean(), b_sub - b_sub.mean()) / (n * a_std * b_std))

    return lags, centers, corr2d, dt


def compute_time_lag_correlation_2d_time(h5_path: str, horizon: int = 200, use_npen: bool = False, tbins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a 2D time-lag Pearson correlation heatmap conditioned on time.

    Returns (lags, t_centers, corr2d, dt)
    - lags: integer lags in steps, from -horizon to +horizon
    - t_centers: centers of the time bins (in seconds if dt>0, else in steps)
    - corr2d: array shape (len(t_centers), len(lags)) with Pearson r per (time_bin, lag)
    - dt: timestep size in seconds (0.0 if unavailable)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ball_ds = f["game/ball_pos"]  # shape (T, 2)
        plat_ds = f["game/platform_pos"]  # shape (T,)
        T = int(min(ball_ds.shape[0], plat_ds.shape[0]))
        if T < 2:
            raise ValueError("Not enough timesteps to compute correlation (need >= 2).")
        by = ball_ds[:T, 1].astype(float)
        py = plat_ds[:T].astype(float)
        dt = float(f.attrs.get("dt", 0.0))

    H = int(max(0, horizon))
    H = min(H, T - 1)
    lags = np.arange(-H, H + 1, dtype=int)

    tbins = int(max(1, tbins))
    # Time-bin edges over absolute indices [0, T-1]
    edges_steps = np.linspace(0.0, float(T - 1), tbins + 1)
    centers_steps = 0.5 * (edges_steps[:-1] + edges_steps[1:])
    centers = centers_steps * dt if dt > 0.0 else centers_steps

    corr2d = np.full((tbins, len(lags)), np.nan, dtype=float)

    for li, k in enumerate(lags):
        if k >= 0:
            a = py[0:T - k]
            b = by[k:T]
            t_idx = np.arange(k, T, dtype=float)
        else:
            a = py[-k:T]
            b = by[0:T + k]
            t_idx = np.arange(0, T + k, dtype=float)

        # Bin by absolute time index (consistent across lags)
        bin_idx = np.digitize(t_idx, edges_steps) - 1  # 0..tbins-1
        bin_idx = np.clip(bin_idx, 0, tbins - 1)

        for bi in range(tbins):
            sel = (bin_idx == bi)
            n = int(sel.sum())
            if n < 3:
                corr2d[bi, li] = np.nan
                continue
            a_sub = a[sel]
            b_sub = b[sel]
            a_std = a_sub.std()
            b_std = b_sub.std()
            if a_std == 0.0 or b_std == 0.0:
                corr2d[bi, li] = 0.0
            else:
                corr2d[bi, li] = float(np.dot(a_sub - a_sub.mean(), b_sub - b_sub.mean()) / (n * a_std * b_std))

    return lags, centers, corr2d, dt


def plot_time_lag_correlation_2d(lags: np.ndarray, x_centers: np.ndarray, corr2d: np.ndarray, dt: float, output: str | None = None):
    """
    Plot 2D heatmap of correlation vs lag (x-axis) and ball X position (y-axis).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Compute extents from centers
    if len(x_centers) > 1:
        dy = x_centers[1] - x_centers[0]
    else:
        dy = 1.0
    y0 = x_centers[0] - 0.5 * dy
    y1 = x_centers[-1] + 0.5 * dy
    x0 = lags[0] - 0.5
    x1 = lags[-1] + 0.5

    # Mask NaNs so they appear as transparent/white
    data = np.ma.masked_invalid(corr2d)
    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[x0, x1, y0, y1], cmap='coolwarm', vmin=-1, vmax=1)
    ax.axvline(0, color='k', lw=1, alpha=0.6)
    ax.set_xlabel('Lag (steps)')
    ax.set_ylabel('Ball X position')
    ax.set_title('2D time-lag correlation: paddle_y vs ball_y conditioned on ball_x')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson correlation')

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    plt.show()


def plot_time_lag_correlation_2d_time(lags: np.ndarray, t_centers: np.ndarray, corr2d: np.ndarray, dt: float, output: str | None = None):
    """
    Plot 2D heatmap of correlation vs lag (x-axis) and time (y-axis).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(t_centers) > 1:
        dy = t_centers[1] - t_centers[0]
    else:
        dy = 1.0
    y0 = t_centers[0] - 0.5 * dy
    y1 = t_centers[-1] + 0.5 * dy
    x0 = lags[0] - 0.5
    x1 = lags[-1] + 0.5

    data = np.ma.masked_invalid(corr2d)
    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[x0, x1, y0, y1], cmap='coolwarm', vmin=-1, vmax=1)
    ax.axvline(0, color='k', lw=1, alpha=0.6)
    ax.set_xlabel('Lag (steps)')
    ax.set_ylabel('Time (s)' if dt > 0.0 else 'Time (steps)')
    ax.set_title('2D time-lag correlation: paddle_y vs ball_y conditioned on time')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson correlation')

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Time-lag correlation between paddle Y and ball Y from pong HDF5 log.')
    parser.add_argument('-i', '--h5', default=os.path.join('output', 'pong_simulation.h5'),
                        help='Path to HDF5 file (default: output/pong_simulation.h5)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='Maximum absolute lag in steps to consider (default: 200)')
    parser.add_argument('--output', default=os.path.join('output', 'pong_time_lag_correlation.png'),
                        help='Output PNG filename')
    parser.add_argument('--npen', action='store_true', help='Use NPEN HDF5 layout (ignored for ball/platform datasets)')
    parser.add_argument('--2d', dest='plot2d', action='store_true',
                        help='Compute a 2D cross-correlation heatmap vs ball X and time lag')
    parser.add_argument('--xbins', type=int, default=50,
                        help='Number of ball X bins for --2d (default: 50)')
    parser.add_argument('--time', nargs='?', const=0, type=int, default=None,
                        help='1D: split into N time partitions (use N>1). 2D: if present, use time on Y-axis; if N omitted, uses --xbins; if provided, uses N time bins.')
    args = parser.parse_args()

    try:
        if args.plot2d:
            # If --time is provided, switch to time-binned 2D (Y-axis is time)
            if args.time is not None:
                tbins = args.xbins if int(args.time) <= 0 else int(args.time)
                lags, t_centers, corr2d, dt = compute_time_lag_correlation_2d_time(
                    args.h5, horizon=args.horizon, use_npen=args.npen, tbins=int(max(1, tbins))
                )
                if np.isfinite(corr2d).any():
                    max_idx = int(np.nanargmax(np.abs(corr2d)))
                    bi, li = np.unravel_index(max_idx, corr2d.shape)
                    best_lag = int(lags[li])
                    best_t = float(t_centers[bi])
                    best_val = float(corr2d[bi, li])
                    t_unit = 's' if dt > 0.0 else 'steps'
                    print(f"Best cell: lag={best_lag} steps, time≈{best_t:.2f} {t_unit}, correlation: {best_val:.4f}")
                else:
                    print("No valid correlations to display for the chosen settings.")
                plot_time_lag_correlation_2d_time(lags, t_centers, corr2d, dt, output=args.output)
            else:
                lags, x_centers, corr2d, dt = compute_time_lag_correlation_2d(
                    args.h5, horizon=args.horizon, use_npen=args.npen, xbins=args.xbins
                )
                if np.isfinite(corr2d).any():
                    max_idx = int(np.nanargmax(np.abs(corr2d)))
                    bi, li = np.unravel_index(max_idx, corr2d.shape)
                    best_lag = int(lags[li])
                    best_x = float(x_centers[bi])
                    best_val = float(corr2d[bi, li])
                    print(f"Best cell: lag={best_lag} steps, ball_x≈{best_x:.2f}, correlation: {best_val:.4f}")
                else:
                    print("No valid correlations to display for the chosen settings.")
                plot_time_lag_correlation_2d(lags, x_centers, corr2d, dt, output=args.output)
        else:
            if args.time is not None and int(args.time) > 1:
                # Partitioned 1D correlations
                py, by, dt = load_series(args.h5)
                parts = int(max(1, args.time))
                py_parts = np.array_split(py, parts)
                by_parts = np.array_split(by, parts)
                min_len = min(len(p) for p in py_parts)
                if min_len < 2:
                    raise ValueError("Not enough timesteps per partition to compute correlation (need >= 2).")
                H_all = int(max(0, min(args.horizon, min_len - 1)))
                if H_all < args.horizon:
                    print(f"Adjusted horizon to {H_all} due to shortest partition length {min_len}.")
                lags, _ = compute_corr_series(py_parts[0], by_parts[0], H_all)
                corrs = []
                labels = []
                for i in range(parts):
                    _, c = compute_corr_series(py_parts[i], by_parts[i], H_all)
                    corrs.append(c)
                    labels.append(f"part {i+1}")
                    # Print best lag summary for each partition
                    best_idx = int(np.nanargmax(np.abs(c)))
                    best_lag = int(lags[best_idx])
                    best_val = float(c[best_idx])
                    print(f"Partition {i+1}/{parts}: best lag {best_lag} steps, correlation {best_val:.4f}, span {len(py_parts[i])} steps")
                plot_time_lag_correlation_multi(lags, corrs, dt, output=args.output, labels=labels)
            else:
                # Single full-series correlation (default behavior)
                lags, corr, dt = compute_time_lag_correlation(args.h5, horizon=args.horizon, use_npen=args.npen)
                best_idx = int(np.nanargmax(np.abs(corr)))
                best_lag = int(lags[best_idx])
                best_val = float(corr[best_idx])
                print(f"Best lag: {best_lag} steps, correlation: {best_val:.4f}")
                plot_time_lag_correlation(lags, corr, dt, output=args.output)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
