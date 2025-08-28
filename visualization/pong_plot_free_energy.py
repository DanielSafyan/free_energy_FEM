import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# Support both NPP (default) and NPEN readers
from pong_simulation.pong_simulation import PongH5Reader as PongH5ReaderNPP
try:
    from pong_simulation.pong_sim_npen import PongH5Reader as PongH5ReaderNPEN
except Exception:
    PongH5ReaderNPEN = None


def tetra_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute the volume of a tetrahedron defined by 4 points."""
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def tetra_gradients(nodes_xyz: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute per-element geometric data for linear finite elements:
    - volumes: (Ne,) array
    - grad_shape: (Ne, 4, 3) array of gradients of the 4 linear shape functions

    For a linear tetrahedral element, the gradient of the interpolated scalar field
    phi(x) = sum_i phi_i * N_i(x) is constant inside the element and equals
    grad_phi = sum_i phi_i * grad(N_i).
    """
    Ne = elements.shape[0]
    volumes = np.zeros(Ne, dtype=float)
    grad_shape = np.zeros((Ne, 4, 3), dtype=float)

    for e in range(Ne):
        n0, n1, n2, n3 = elements[e]
        p0, p1, p2, p3 = nodes_xyz[n0], nodes_xyz[n1], nodes_xyz[n2], nodes_xyz[n3]

        # Build matrix of node coordinates (with 1 appended) to compute gradients
        # Reference: grad(N_i) = inv(J).T * grad_hat(N_i) where grad_hat are reference grads.
        # For linear tetra, an efficient formula exists using opposite face normals.
        # We use the formula: grad(N_i) = n_i / (6*V), where n_i is the outward normal vector
        # of the face opposite node i, with magnitude equal to twice the face area, oriented outward.
        V = tetra_volume(p0, p1, p2, p3)
        if V <= 0.0:
            # Degenerate; skip to avoid division by zero
            volumes[e] = 0.0
            grad_shape[e, :, :] = 0.0
            continue
        volumes[e] = V

        # Opposite face normals (unnormalized): cross of two edges of the face
        # Face opposite node 0 is (1,2,3)
        n0_vec = np.cross(p2 - p1, p3 - p1)
        # Opposite node 1 -> face (0,2,3)
        n1_vec = np.cross(p3 - p0, p2 - p0)
        # Opposite node 2 -> face (0,1,3)
        n2_vec = np.cross(p1 - p0, p3 - p0)
        # Opposite node 3 -> face (0,1,2)
        n3_vec = np.cross(p2 - p0, p1 - p0)

        # Gradients of shape functions
        grad_shape[e, 0, :] = n0_vec / (6.0 * V)
        grad_shape[e, 1, :] = n1_vec / (6.0 * V)
        grad_shape[e, 2, :] = n2_vec / (6.0 * V)
        grad_shape[e, 3, :] = n3_vec / (6.0 * V)

    return volumes, grad_shape


def compute_total_free_energy_over_time(h5_path: str, output_png: str = None,
                                         start_frac: float = 0.0, until_frac: float = 1.0,
                                         use_npen: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load HDF5 simulation data and compute total free energy vs time.

    Returns (time_array, total_free_energy).
    Saves a PNG if output_png is provided.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    Reader = PongH5ReaderNPEN if use_npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
    with Reader(h5_path) as data:
        nodes = data.nodes[...]
        elements = data.elements[...]
        attrs = data.attrs
        consts = data.constants

        # Required series
        # If NPEN: use single c for both c1 and c2 channels
        if use_npen and hasattr(data, 'c'):
            c1_ds = data.c
            c2_ds = data.c
        else:
            c1_ds = data.c1
            c2_ds = data.c2
        phi_ds = data.phi

        Tsteps = min(c1_ds.shape[0], c2_ds.shape[0], phi_ds.shape[0])
        # Print dataset length (timesteps)
        print(f"Dataset length (timesteps): {Tsteps}")

        # Validate and compute slicing indices from fractions
        if not (0.0 <= start_frac <= 1.0) or not (0.0 <= until_frac <= 1.0):
            raise ValueError("--from and --until must be between 0 and 1 (inclusive)")
        if start_frac >= until_frac:
            raise ValueError("--from must be less than --until")

        start_idx = int(np.floor(start_frac * Tsteps))
        end_idx = int(np.ceil(until_frac * Tsteps))
        start_idx = max(0, min(start_idx, Tsteps))
        end_idx = max(0, min(end_idx, Tsteps))
        if end_idx - start_idx <= 0:
            raise ValueError("Slicing range is empty after applying --from/--until")

        dt = float(attrs.get("dt", 0.0))
        time_array = np.arange(end_idx - start_idx, dtype=float) * dt + start_idx * dt

        # Physical parameters (fallback to sane defaults if missing)
        R = float(consts.get("R", 1.0))
        T = float(consts.get("T", 1.0))
        RT = R * T
        chi = float(consts.get("chi", 0.0))
        epsilon = float(consts.get("epsilon", 1.0))
        F = float(consts.get("F", 1.0))
        z1 = float(consts.get("z1", 1.0))
        z2 = float(consts.get("z2", -1.0))
        c0 = float(consts.get("c0", 1.0))
        phi_c = float(consts.get("phi_c", 1.0))

        # If concentrations are stored dimensionless and need scaling, include here.
        # We assume they are already in physical units consistent with the constants.

        # Precompute element volumes and shape gradients
        if elements.shape[1] != 4:
            raise ValueError("Mesh elements in HDF5 are not tetrahedra (expected 4-node connectivity).")
        volumes, grad_shape = tetra_gradients(nodes, elements)
        Ne = elements.shape[0]

        total_free_energy = np.zeros(end_idx - start_idx, dtype=float)

        # Precompute element to node indices for fast gather
        elem_nodes = elements.astype(int)

        print(f"Using slice [{start_idx}:{end_idx}] (fraction {start_frac:.3f} to {until_frac:.3f})")

        for t_out, t in enumerate(range(start_idx, end_idx)):
            c1 = c1_ds[t]
            c2 = c2_ds[t]
            phi = phi_ds[t]

            # Gather nodal values per element: shape (Ne, 4)
            c1_e = c1[elem_nodes]
            c2_e = c2[elem_nodes]
            phi_e = phi[elem_nodes]

            # Element-averaged values (simple arithmetic mean of vertices)
            # Scale to physical units using c0 and phi_c
            c1_avg = np.mean(c1_e, axis=1) * c0  # (Ne,)
            c2_avg = np.mean(c2_e, axis=1) * c0
            phi_avg = np.mean(phi_e, axis=1) * phi_c

            # Entropy + interaction + coupling energy densities (per element, constant approximation)
            # Avoid log(0)
            eps = 1e-12
            c1_safe = np.maximum(c1_avg, eps)
            c2_safe = np.maximum(c2_avg, eps)

            entropy = RT * (c1_safe * np.log(c1_safe) + c2_safe * np.log(c2_safe))
            interaction = chi * c1_avg * c2_avg
            coupling = F * (z1 * c1_avg + z2 * c2_avg) * phi_avg

            # Gradient energy: for each element, grad_phi = sum_i phi_i * grad(N_i)
            # phi_e: (Ne,4); grad_shape: (Ne,4,3) -> grad_phi: (Ne,3)
            # Scale phi_e to physical units before gradient: grad(phi_phys) = phi_c * grad(phi_dimless)
            grad_phi = np.einsum('ei,eik->ek', phi_e * phi_c, grad_shape, optimize=True)
            grad_sq = np.einsum('ek,ek->e', grad_phi, grad_phi, optimize=True)
            grad_energy = 0.5 * epsilon * grad_sq

            # Total energy density per element
            fe_density = entropy + interaction + coupling + grad_energy  # (Ne,)

            # Integrate over volume
            E_total_t = np.sum(fe_density * volumes)
            total_free_energy[t_out] = E_total_t

    # Optional save plot
    if output_png:
        os.makedirs(os.path.dirname(output_png) or '.', exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(time_array, total_free_energy, 'k-', lw=2, label='Total Free Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Free Energy (J)')
        plt.title('Total Free Energy vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_png}")

    return time_array, total_free_energy


def compute_cumulative_score_over_time(h5_path: str, start_frac: float = 0.0, until_frac: float = 1.0,
                                       use_npen: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Read cumulative score per timestep from the HDF5 file.

    Uses the stored `game/score` dataset written during simulation.
    Returns (time_array_s, score_window).
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    Reader = PongH5ReaderNPEN if use_npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
    with Reader(h5_path) as data:
        if getattr(data, "score", None) is None:
            raise RuntimeError("This HDF5 file has no 'game/score' dataset. Regenerate the data with the updated simulator.")
        score_ds = data.score
        attrs = data.attrs
        dt = float(attrs.get("dt", 0.0))
        Tsteps = int(score_ds.shape[0])

        # Slice to requested window
        if not (0.0 <= start_frac <= 1.0) or not (0.0 <= until_frac <= 1.0):
            raise ValueError("--from and --until must be between 0 and 1 (inclusive)")
        if start_frac >= until_frac:
            raise ValueError("--from must be less than --until")
        start_idx = int(np.floor(start_frac * Tsteps))
        end_idx = int(np.ceil(until_frac * Tsteps))
        start_idx = max(0, min(start_idx, Tsteps))
        end_idx = max(0, min(end_idx, Tsteps))
        if end_idx - start_idx <= 0:
            raise ValueError("Slicing range is empty after applying --from/--until")

        time_array = np.arange(end_idx - start_idx, dtype=float) * dt + start_idx * dt
        cum_scores = score_ds[start_idx:end_idx].astype(float)

    return time_array, cum_scores

 
def compute_hitrate_from_cumulative_scores(cum_scores: np.ndarray, horizon: int) -> np.ndarray:
    """Compute rolling count of score increases over a fixed timestep horizon.
    Only in-window increases are counted; decreases are ignored.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    # Differences between consecutive cumulative scores (first diff = 0 within window)
    delta = np.diff(cum_scores, prepend=cum_scores[0])
    # Count only positive increments (hits)
    delta = np.clip(delta, 0, None)
    # Rolling sum over the last 'horizon' steps for each t
    N = int(delta.shape[0])
    cs = np.cumsum(np.concatenate(([0], delta)))
    idx = np.arange(1, N + 1)
    start_idx = np.maximum(0, idx - horizon)
    rolling = cs[idx] - cs[start_idx]
    return rolling


def compute_left_right_sequence(h5_path: str, start_frac: float = 0.0, until_frac: float = 1.0,
                                use_npen: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute left/right sequence over time along with x positions and midline.

    Returns (time_array_s, sides_window, x_window, mid), where sides_window is 0 for left, 1 for right.
    The midline is determined globally from max_x across the entire file: mid = max_x / 2.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    Reader = PongH5ReaderNPEN if use_npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
    with Reader(h5_path) as data:
        ball = data.ball_pos[...]
        attrs = data.attrs
        dt = float(attrs.get("dt", 0.0))
        Tsteps = int(ball.shape[0])

        bx = ball[:, 0].astype(float)
        max_x = np.nanmax(bx)
        mid = max_x / 2.0
        sides = (bx >= mid).astype(int)
        # Forward-fill for NaNs if present
        if np.any(np.isnan(bx)):
            for i in range(Tsteps):
                if np.isnan(bx[i]):
                    sides[i] = sides[i - 1] if i > 0 else 1

        # Windowing
        if not (0.0 <= start_frac <= 1.0) or not (0.0 <= until_frac <= 1.0):
            raise ValueError("--from and --until must be between 0 and 1 (inclusive)")
        if start_frac >= until_frac:
            raise ValueError("--from must be less than --until")
        start_idx = int(np.floor(start_frac * Tsteps))
        end_idx = int(np.ceil(until_frac * Tsteps))
        start_idx = max(0, min(start_idx, Tsteps))
        end_idx = max(0, min(end_idx, Tsteps))
        if end_idx - start_idx <= 0:
            raise ValueError("Slicing range is empty after applying --from/--until")

        time_array = np.arange(end_idx - start_idx, dtype=float) * dt + start_idx * dt
        sides_window = sides[start_idx:end_idx].astype(int)
        x_window = bx[start_idx:end_idx]

    return time_array, sides_window, x_window, float(mid)


def main():
    parser = argparse.ArgumentParser(description='Plot total free energy vs time from pong HDF5 log.')
    parser.add_argument('-i', '--h5', default=os.path.join('output', 'pong_simulation.h5'),
                        help='Path to HDF5 file (default: output/pong_simulation.h5)')
    parser.add_argument('--output', default=os.path.join('output', 'pong_total_free_energy.png'),
                        help='Output PNG filename')
    parser.add_argument('--static', action='store_true', help='No-op flag to mirror other tools; always static plot')
    parser.add_argument('--from', dest='start_frac', type=float, default=0.0,
                        help='Fraction [0,1] from start to begin processing (e.g., 0.3 drops first 30%)')
    parser.add_argument('--until', dest='until_frac', type=float, default=1.0,
                        help='Fraction [0,1] from start to end processing window (e.g., 0.8 stops at 80%)')
    parser.add_argument('--lr-output', default=None,
                        help='Optional CSV to save left-right time sequence: columns time_s, side(0/1), x')
    parser.add_argument('--npen', action='store_true', help='Use NPEN HDF5 layout (states/c instead of c1/c2)')
    parser.add_argument('--ts', choices=['s', 'ms', 'mms', 'ns'], default=None,
                        help='Time scale for plotting/CSV: s=seconds, ms=milliseconds, mms=microseconds, ns=nanoseconds')
    parser.add_argument('--hitrate', nargs='?', const=200, type=int, metavar='H', default=None,
                        help='Replace lower plot with rolling hit count: number of score increases in the last H timesteps (default H=200 if omitted).')
    args = parser.parse_args()

    try:
        # Determine time scaling and unit label (explicit flag overrides auto)
        Reader = PongH5ReaderNPEN if args.npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
        with Reader(args.h5) as _r:
            dt = float(_r.attrs.get('dt', 0.0))
        def _time_scale_from_flag(flag: str | None, dt_seconds: float):
            if flag:
                m = flag.lower()
                if m == 's':
                    return 1.0, 's'
                if m == 'ms':
                    return 1e3, 'ms'
                if m == 'mms':  # microseconds
                    return 1e6, 'µs'
                if m == 'ns':
                    return 1e9, 'ns'
            # Auto from dt if not specified
            if dt_seconds <= 0 or not np.isfinite(dt_seconds):
                return 1.0, 'steps'
            if dt_seconds >= 1.0:
                return 1.0, 's'
            if dt_seconds >= 1e-3:
                return 1e3, 'ms'
            if dt_seconds >= 1e-6:
                return 1e6, 'µs'
            return 1e9, 'ns'
        t_scale, t_unit = _time_scale_from_flag(args.ts, dt)

        # Compute free energy over time (no immediate plotting here)
        time_s, E = compute_total_free_energy_over_time(
            args.h5,
            output_png=None,
            start_frac=args.start_frac,
            until_frac=args.until_frac,
            use_npen=args.npen,
        )
        # Compute cumulative score over time using the same slicing window
        score_t, cum_scores = compute_cumulative_score_over_time(
            args.h5, start_frac=args.start_frac, until_frac=args.until_frac, use_npen=args.npen
        )

        print(score_t.shape)
        print(cum_scores.shape)
        print(np.sum(cum_scores))


        # Combined figure: Free energy (top) and score metric (bottom)
        fig, (axE, axS) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        # Rescale time vectors for plotting
        time_plot = (time_s * t_scale) if (dt > 0 or args.ts) else time_s
        score_time_plot = (score_t * t_scale) if (dt > 0 or args.ts) else score_t
        axE.plot(time_plot, E, 'k-', lw=2, label='Total Free Energy')
        axE.set_xlabel(f'Time ({t_unit})')
        axE.set_ylabel('Free Energy (J)')
        axE.set_title('Total Free Energy vs Time')
        axE.grid(True, alpha=0.3)
        axE.legend()

        if args.hitrate is not None:
            H = int(args.hitrate)
            hitrate = compute_hitrate_from_cumulative_scores(cum_scores, H)
            axS.plot(score_time_plot, hitrate, color='C1', lw=1.8, label=f'Hitrate (last {H} steps)')
            axS.set_xlabel(f'Time ({t_unit})')
            axS.set_ylabel(f'Hits in last {H} steps')
            axS.set_title('Rolling Hit Count')
            axS.grid(True, alpha=0.3)
            axS.legend()
            # Optional intuitive y-limit
            axS.set_ylim(0, max(np.max(hitrate) * 1.1, 1))
        else:
            axS.step(score_time_plot, cum_scores, where='post')
            axS.set_xlabel(f'Time ({t_unit})')
            axS.set_ylabel('Cumulative score')
            axS.set_title('Maximum score up to time t')
            axS.grid(True, alpha=0.3)

        # Optionally save left-right time sequence CSV
        if args.lr_output:
            lr_t, lr_sides, lr_x, lr_mid = compute_left_right_sequence(
                args.h5, start_frac=args.start_frac, until_frac=args.until_frac, use_npen=args.npen
            )
            os.makedirs(os.path.dirname(args.lr_output) or '.', exist_ok=True)
            lr_t_plot = (lr_t * t_scale) if (dt > 0 or args.ts) else lr_t
            out_arr = np.column_stack([lr_t_plot, lr_sides, lr_x])
            header = f"time_{t_unit},side_0left_1right,x,midline={lr_mid}"
            np.savetxt(args.lr_output, out_arr, delimiter=',', header=header, comments='')
            print(f"Saved left-right sequence: {args.lr_output} (midline={lr_mid:.3f})")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Saved: {args.output}")

        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
