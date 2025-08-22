import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np
import pygame
import matplotlib.pyplot as plt
import shutil

# Reuse the game's rendering/look by importing PongGame
from gameplay.pong_game import PongGame, SCREEN_WIDTH, SCREEN_HEIGHT
# Use the HDF5 reader utilities from the simulation module
# Default NPP reader; NPEN reader is optional
from pong_simulation.pong_simulation import PongH5Reader as PongH5ReaderNPP
try:
    from pong_simulation.pong_sim_npen import PongH5Reader as PongH5ReaderNPEN
except Exception:
    PongH5ReaderNPEN = None


def _structured_get_node_idx(i: int, j: int, k: int, ny: int, nz: int) -> int:
    """Map structured grid indices (i,j,k) to node index.

    Matches the indexing used in `pong_simulation.pong_simulation`:
    idx = i*(ny+1)*(nz+1) + j*(nz+1) + k
    """
    return i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k


def _build_midplane_triangulation(nodes: np.ndarray, attrs: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D triangulation on the mid-plane (k = nz//2) for tripcolor.

    Returns (xy_coords, triangles, slice_node_indices)
    - xy_coords: (Ns, 2) array of x,y for mid-plane nodes
    - triangles: (Nt, 3) int array of local indices into xy_coords
    - slice_node_indices: (Ns,) int array mapping local -> global node index
    """
    nx = int(attrs.get("nx"))
    ny = int(attrs.get("ny"))
    nz = int(attrs.get("nz"))
    k = nz // 2

    # Collect all node indices on this k-slice in structured order
    slice_indices = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            idx = _structured_get_node_idx(i, j, k, ny, nz)
            slice_indices.append(idx)
    slice_indices = np.array(slice_indices, dtype=int)

    # Build local remap: global -> local
    global_to_local = {g: l for l, g in enumerate(slice_indices.tolist())}

    # Extract XY coordinates
    xy = nodes[slice_indices][:, :2]

    # Build triangles by splitting each cell quad into two triangles
    tris = []
    # local indexing helper: within the (nx+1) x (ny+1) grid on this slice
    def local(i, j):
        return i * (ny + 1) + j

    for i in range(nx):
        for j in range(ny):
            v00 = local(i, j)
            v10 = local(i + 1, j)
            v01 = local(i, j + 1)
            v11 = local(i + 1, j + 1)
            # Split into two triangles (consistent orientation not critical for tripcolor)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    triangles = np.array(tris, dtype=int)
    return xy, triangles, slice_indices


def detect_round_starts(ball_xy: np.ndarray) -> np.ndarray:
    """Return indices t where a new round starts (ball reset to exact initial center).

    Heuristic: A reset in the simulation constructs a new PongGame and we log the
    first frame after reset with the ball at the exact initial center. We detect
    exact equality with the t=0 position and ignore t=0. Only the first index of
    any contiguous run at center is kept.
    """
    if ball_xy.ndim != 2 or ball_xy.shape[1] != 2 or ball_xy.shape[0] == 0:
        return np.array([], dtype=int)
    init = ball_xy[0]
    eq0 = np.where((ball_xy == init).all(axis=1))[0]
    eq0 = eq0[eq0 != 0]
    if eq0.size:
        keep = [eq0[0]]
        for i in range(1, len(eq0)):
            if eq0[i] != eq0[i - 1] + 1:
                keep.append(eq0[i])
        eq0 = np.array(keep, dtype=int)
    return eq0


def replay_pong(h5_path: str, fps: int = 60, show_fields: bool = True, sum_z: bool = False,
                start_frac: float = 0.0, until_frac: float = 1.0,
                video: bool = False, video_output: str = os.path.join("output", "pong_replay.mp4"),
                frame_dir: str = os.path.join("output", "pong_replay_frames"), dpi: int = 100,
                use_npen: bool = False,
                step: int = 1,
                change: bool = False,
                time_scale: str | None = None):
    """
    Replay the Pong game using positions recorded in an HDF5 file.

    Parameters:
    - h5_path: Path to the HDF5 file produced by pong_simulation.
    - fps: Frames per second for replay.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    # Choose appropriate reader (NPP default, NPEN if requested)
    Reader = PongH5ReaderNPEN if use_npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
    # Load time series lazily
    with Reader(h5_path) as data:
        ball_ds = data.ball_pos  # shape (T, 2)
        plat_ds = data.platform_pos  # shape (T,)
        # NPEN provides a single salt concentration `c`; map to both channels for compatibility
        is_npen = bool(use_npen and hasattr(data, 'c'))
        if is_npen:
            c1_ds = data.c
            c2_ds = data.c
        else:
            c1_ds = data.c1
            c2_ds = data.c2
        phi_ds = data.phi
        nodes = data.nodes[...]
        attrs = data.attrs
        num_frames = min(ball_ds.shape[0], plat_ds.shape[0], c1_ds.shape[0], c2_ds.shape[0], phi_ds.shape[0])

        # Determine slice window [start_idx:end_idx)
        if not (0.0 <= start_frac <= 1.0) or not (0.0 <= until_frac <= 1.0):
            raise ValueError("--from and --until must be between 0 and 1 (inclusive)")
        if start_frac >= until_frac:
            raise ValueError("--from must be less than --until")
        start_idx = int(np.floor(start_frac * num_frames))
        end_idx = int(np.ceil(until_frac * num_frames))
        start_idx = max(0, min(start_idx, num_frames))
        end_idx = max(0, min(end_idx, num_frames))
        if end_idx - start_idx <= 0:
            raise ValueError("Slicing range is empty after applying --from/--until")

        # Build list of global indices to play, honoring the step size
        if step is None:
            step = 1
        if not isinstance(step, int) or step < 1:
            raise ValueError("--step must be a positive integer (>= 1)")
        indices = np.arange(start_idx, end_idx, step, dtype=int)
        if indices.size == 0:
            raise ValueError("No frames selected after applying --from/--until and --step")

        # Initialize pygame and the game renderer
        pygame.init()
        # If video mode, create a hidden window to avoid showing realtime gameplay
        if 'video' in locals() and video:
            try:
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.HIDDEN)
            except Exception:
                # Fallback: normal window if HIDDEN not supported
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pong Replay")
        clock = pygame.time.Clock()

        # logging=False to avoid CSV side effects during replay
        game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT, logging=False)
        # Score overlay setup
        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
        score = 0
        # For robust detection, precompute center equals for all frames
        try:
            all_ball = ball_ds[...]
            center_frames = (all_ball == all_ball[0]).all(axis=1)
        except Exception:
            center_frames = None

        # If video is requested, we also enable field plotting for the video frames.
        if show_fields or video:
            # Precompute mid-plane triangulation and global color limits
            xy_mid, tri_mid, slice_idx = _build_midplane_triangulation(nodes, attrs)

            # Precompute column indices across k for z-sum mode
            nx = int(attrs.get("nx"))
            ny = int(attrs.get("ny"))
            nz = int(attrs.get("nz"))
            if sum_z:
                num_ij = (nx + 1) * (ny + 1)
                cols = np.zeros((num_ij, nz + 1), dtype=int)
                # local indexing helper within 2D grid
                def local(i, j):
                    return i * (ny + 1) + j
                for i in range(nx + 1):
                    for j in range(ny + 1):
                        L = local(i, j)
                        for k in range(nz + 1):
                            cols[L, k] = _structured_get_node_idx(i, j, k, ny, nz)
            else:
                cols = None

            # Time step from attrs (float seconds)
            dt = float(attrs.get("dt", 0.0))
            # Determine time unit scale/label
            def _time_scale_from_flag(flag: str | None, dt_seconds: float):
                # explicit flag overrides auto
                if flag:
                    m = flag.lower()
                    if m == 's':
                        return 1.0, 's'
                    if m == 'ms':
                        return 1e3, 'ms'
                    if m in ('mms', 'us', 'µs'):
                        return 1e6, 'µs'
                    if m == 'ns':
                        return 1e9, 'ns'
                # auto from dt
                if dt_seconds <= 0 or not np.isfinite(dt_seconds):
                    return 1.0, 'steps'
                if dt_seconds >= 1.0:
                    return 1.0, 's'
                if dt_seconds >= 1e-3:
                    return 1e3, 'ms'
                if dt_seconds >= 1e-6:
                    return 1e6, 'µs'
                return 1e9, 'ns'
            t_scale, t_unit = _time_scale_from_flag(time_scale, dt)

            # For video we create a 2x2 layout: gameplay + three fields
            if video:
                # Use non-interactive backend to avoid opening windows
                try:
                    plt.switch_backend('Agg')
                except Exception:
                    pass
                # Prepare frame directory
                if os.path.exists(frame_dir):
                    shutil.rmtree(frame_dir)
                os.makedirs(frame_dir, exist_ok=True)

                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(2, 2)
                ax_game = fig.add_subplot(gs[0, 0])
                ax_c2 = fig.add_subplot(gs[0, 1])
                ax_c1 = fig.add_subplot(gs[1, 0])
                ax_phi = fig.add_subplot(gs[1, 1])
            else:
                plt.ion()
                if is_npen:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

            # Helper to compute triangle face values for a field
            def tri_face_values(field_1d: np.ndarray) -> np.ndarray:
                if sum_z:
                    # sum across k for each (i,j)
                    vals2d = field_1d[cols].sum(axis=1)
                    return vals2d[tri_mid].mean(axis=1)
                else:
                    vals = field_1d[slice_idx]
                    return vals[tri_mid].mean(axis=1)

            # Initialize collections at the first selected timestep
            first_idx = int(indices[0])
            c2_faces = tri_face_values(c2_ds[first_idx])
            c1_faces = tri_face_values(c1_ds[first_idx])
            phi_faces = tri_face_values(phi_ds[first_idx])

            if not video:
                if is_npen:
                    ax_c2, ax_phi = axes  # use ax_c2 as the single 'c' panel
                else:
                    ax_c2, ax_c1, ax_phi = axes
            title_suffix = "∑z " if sum_z else ""
            conc_prefix = "Δ " if change else ""
            # Field names: when NPEN, show single 'c' panel (reuse ax_c2) and label accordingly
            name_c2 = 'c' if is_npen else 'c2'
            name_c1 = 'c' if is_npen else 'c1'
            # For change mode, start at zero field for the first frame
            init_c2_faces = (c2_faces - c2_faces) if change else c2_faces
            coll_c2 = ax_c2.tripcolor(xy_mid[:, 0], xy_mid[:, 1], tri_mid, facecolors=init_c2_faces, cmap='plasma')
            ax_c2.triplot(xy_mid[:, 0], xy_mid[:, 1], tri_mid, 'k-', lw=0.1, alpha=0.3)
            t0_disp = ((first_idx * dt * t_scale) if dt > 0 else first_idx)
            ax_c2.set_title(f"{title_suffix}{conc_prefix}{name_c2} @ t = {t0_disp:.2f} {t_unit}")
            ax_c2.set_aspect('equal')
            ax_c2.set_xlabel("x (m)")
            ax_c2.set_ylabel("y (m)")
            fig.colorbar(coll_c2, ax=ax_c2, label=f"Concentration {name_c2}")

            if not (is_npen and not video):
                # Only create the second concentration panel when not collapsing for NPEN in live view
                init_c1_faces = (c1_faces - c1_faces) if change else c1_faces
                coll_c1 = ax_c1.tripcolor(xy_mid[:, 0], xy_mid[:, 1], tri_mid, facecolors=init_c1_faces, cmap='viridis')
                ax_c1.triplot(xy_mid[:, 0], xy_mid[:, 1], tri_mid, 'k-', lw=0.1, alpha=0.3)
                ax_c1.set_title(f"{title_suffix}{conc_prefix}{name_c1} @ t = {t0_disp:.2f} {t_unit}")
                ax_c1.set_aspect('equal')
                ax_c1.set_xlabel("x (m)")
                ax_c1.set_ylabel("y (m)")
                fig.colorbar(coll_c1, ax=ax_c1, label=f"Concentration {name_c1}")

            coll_phi = ax_phi.tripcolor(xy_mid[:, 0], xy_mid[:, 1], tri_mid, facecolors=phi_faces, cmap='coolwarm')
            ax_phi.triplot(xy_mid[:, 0], xy_mid[:, 1], tri_mid, 'k-', lw=0.1, alpha=0.3)
            ax_phi.set_title(f"{title_suffix}φ @ t = {t0_disp:.2f} {t_unit}")
            ax_phi.set_aspect('equal')
            ax_phi.set_xlabel("x (m)")
            ax_phi.set_ylabel("y (m)")
            fig.colorbar(coll_phi, ax=ax_phi, label="Electric Potential φ (V)")
            if video:
                # Apply a one-time layout tightening for the static grid
                try:
                    fig.tight_layout()
                except Exception:
                    pass

            # For video: add an AxesImage for the gameplay frame in ax_game
            if video:
                import pygame.surfarray as surfarray
                frame_rgb = np.transpose(surfarray.array3d(screen), (1, 0, 2))
                img_game = ax_game.imshow(frame_rgb)
                ax_game.set_title("Pong Gameplay")
                ax_game.axis('off')

        running = True
        # local frame counter over selected indices
        t = 0
        prev_center = False
        total_in_window = int(indices.size)
        while running and t < total_in_window:
            # Handle minimal events to keep window responsive and allow quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            

            # Global index within full arrays (respecting step)
            g = int(indices[t])
            # Apply recorded positions for this frame
            bx, by = ball_ds[g]
            py = plat_ds[g]

            # Update game object's state directly (no physics step)
            game.ball.x = int(bx)
            game.ball.y = int(by)
            game.set_platform_position(int(py))

            # Draw current frame using the game's draw routine
            game.draw(screen)
            # Detect round restart and update score
            is_center = False
            if center_frames is not None:
                is_center = bool(center_frames[g])
            else:
                init_bx, init_by = ball_ds[0]
                is_center = (bx == init_bx) and (by == init_by) and (t != 0)
            if is_center and (t > 0) and (not prev_center):
                score += 1
            prev_center = is_center

            # Overlay score text (top-left)
            try:
                text = font.render(f"Score: {score}", True, (255, 255, 255))
                # Draw a simple shadow for readability
                shadow = font.render(f"Score: {score}", True, (0, 0, 0))
                screen.blit(shadow, (11, 11))
                screen.blit(text, (10, 10))
            except Exception:
                pass
            pygame.display.flip()

            clock.tick(fps)
            t += 1

            # Update plots (and frames if video)
            if show_fields or video:
                try:
                    if sum_z:
                        vals2d_c2 = c2_ds[g][cols].sum(axis=1)
                        vals2d_c1 = c1_ds[g][cols].sum(axis=1)
                        vals2d_phi = phi_ds[g][cols].sum(axis=1)
                        c2_faces_curr = vals2d_c2[tri_mid].mean(axis=1)
                        c1_faces_curr = vals2d_c1[tri_mid].mean(axis=1)
                        phi_faces = vals2d_phi[tri_mid].mean(axis=1)
                    else:
                        c2_faces_curr = (c2_ds[g][slice_idx])[tri_mid].mean(axis=1)
                        c1_faces_curr = (c1_ds[g][slice_idx])[tri_mid].mean(axis=1)
                        phi_faces = (phi_ds[g][slice_idx])[tri_mid].mean(axis=1)

                    if change:
                        # Compare against 'step' raw timesteps earlier (clamped to start_idx)
                        g_prev = int(max(start_idx, g - step))
                        if sum_z:
                            vals2d_c2_prev = c2_ds[g_prev][cols].sum(axis=1)
                            vals2d_c1_prev = c1_ds[g_prev][cols].sum(axis=1)
                            c2_faces_prev = vals2d_c2_prev[tri_mid].mean(axis=1)
                            c1_faces_prev = vals2d_c1_prev[tri_mid].mean(axis=1)
                        else:
                            c2_faces_prev = (c2_ds[g_prev][slice_idx])[tri_mid].mean(axis=1)
                            c1_faces_prev = (c1_ds[g_prev][slice_idx])[tri_mid].mean(axis=1)
                        c2_faces = c2_faces_curr - c2_faces_prev
                        c1_faces = c1_faces_curr - c1_faces_prev
                    else:
                        c2_faces = (c2_faces_curr - c2_faces_curr) if change else c2_faces_curr
                        c1_faces = (c1_faces_curr - c1_faces_curr) if change else c1_faces_curr

                    coll_c2.set_array(c2_faces)
                    if not (is_npen and not video):
                        coll_c1.set_array(c1_faces)
                    coll_phi.set_array(phi_faces)
                    # Per-frame autoscaling like free_energy_visualization (no fixed vmin/vmax)
                    coll_c2.set_clim(vmin=float(np.nanmin(c2_faces)), vmax=float(np.nanmax(c2_faces)))
                    #print(f"all 0 = {np.all(c2_faces == 0)}")
                    if not (is_npen and not video):
                        coll_c1.set_clim(vmin=float(np.nanmin(c1_faces)), vmax=float(np.nanmax(c1_faces)))
                    coll_phi.set_clim(vmin=float(np.nanmin(phi_faces)), vmax=float(np.nanmax(phi_faces)))
                    # Titles with selected time scale
                    t_disp = (g * dt * t_scale) if dt > 0 else g
                    ax_c2.set_title(f"{title_suffix}{conc_prefix}{name_c2} @ t = {t_disp:.2f} {t_unit}")
                    if not (is_npen and not video):
                        ax_c1.set_title(f"{title_suffix}{conc_prefix}{name_c1} @ t = {t_disp:.2f} {t_unit}")
                    ax_phi.set_title(f"{title_suffix}φ @ t = {t_disp:.2f} {t_unit}")
                    if video:
                        import pygame.surfarray as surfarray
                        frame_rgb = np.transpose(surfarray.array3d(screen), (1, 0, 2))
                        img_game.set_data(frame_rgb)
                        fig.canvas.draw()
                        frame_no = max(0, t - 1)
                        fig.savefig(os.path.join(frame_dir, f"frame_{frame_no:04d}.png"), dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                    else:
                        fig.canvas.draw_idle()
                        plt.pause(0.001)
                except Exception:
                    pass

            # small delay to make replay viewable (skip extra delay when rendering video)
            if not video:
                time.sleep(0.01)

        pygame.quit()

        if show_fields and not video:
            try:
                plt.ioff()
                plt.show(block=False)
            except Exception:
                pass

        # If video requested, stitch frames into a video using ffmpeg and clean up
        if video:
            try:
                # Ensure output directory exists
                out_dir = os.path.dirname(video_output)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                print(f"Creating video '{video_output}'...")
                ffmpeg_command = (
                    f"ffmpeg -y -r {fps} -i {frame_dir}/frame_%04d.png "
                    f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
                    f"-vcodec libx264 -crf 25 -pix_fmt yuv420p {video_output}"
                )
                os.system(ffmpeg_command)
            finally:
                if os.path.exists(frame_dir):
                    shutil.rmtree(frame_dir)


def main():
    parser = argparse.ArgumentParser(description="Replay Pong from HDF5 log.")
    parser.add_argument(
        "--h5", "-i", default=os.path.join("output", "pong_simulation.h5"),
        help="Path to pong_simulation HDF5 file (default: output/pong_simulation.h5)"
    )
    parser.add_argument("--fps", type=int, default=60, help="Replay frames per second (default: 60)")
    parser.add_argument("--no-fields", action="store_true", help="Disable c1/c2/phi plotting window")
    parser.add_argument("--sum", action="store_true", help="Display z-sum of fields instead of single mid-plane slice")
    parser.add_argument("--change", action="store_true", help="Plot change (Δ) in concentration vs previous selected timestep instead of absolute values")
    parser.add_argument("--from", dest="start_frac", type=float, default=0.0,
                        help="Fraction [0,1] from start to begin playback (e.g., 0.3 starts at 30%)")
    parser.add_argument("--until", dest="until_frac", type=float, default=1.0,
                        help="Fraction [0,1] from start to end playback window (e.g., 0.8 stops at 80%)")
    parser.add_argument("--video", action="store_true", help="Enable video recording of gameplay + fields")
    parser.add_argument("--video-output", default=os.path.join("output", "pong_replay.mp4"),
                        help="Output MP4 file path (default: output/pong_replay.mp4)")
    parser.add_argument("--npen", action="store_true", help="Use NPEN HDF5 layout (states/c instead of c1/c2)")
    parser.add_argument("--ts", choices=["s", "ms", "mms", "ns"], default=None,
                        help="Time scale for plotting (overrides auto): s seconds, ms milliseconds, mms microseconds, ns nanoseconds")
    parser.add_argument("--step", type=int, default=1,
                        help=(
                            "Sampling stride (>=1): only every Nth timestep is plotted. "
                            "When --change is set, the delta is computed vs the frame that is 'step' raw timesteps earlier."
                        ))
    args = parser.parse_args()

    try:
        # Quick info printout about the selected window
        ReaderInfo = PongH5ReaderNPEN if args.npen and (PongH5ReaderNPEN is not None) else PongH5ReaderNPP
        with ReaderInfo(args.h5) as _data:
            if args.npen and hasattr(_data, 'c'):
                _T = int(min(_data.ball_pos.shape[0], _data.platform_pos.shape[0],
                             _data.c.shape[0], _data.c.shape[0], _data.phi.shape[0]))
            else:
                _T = int(min(_data.ball_pos.shape[0], _data.platform_pos.shape[0],
                             _data.c1.shape[0], _data.c2.shape[0], _data.phi.shape[0]))
        s_idx = int(np.floor(max(0.0, min(1.0, args.start_frac)) * _T))
        e_idx = int(np.ceil(max(0.0, min(1.0, args.until_frac)) * _T))
        print(f"Dataset length (timesteps): {_T}")
        print(f"Using slice [{s_idx}:{e_idx}] (fraction {args.start_frac:.3f} to {args.until_frac:.3f}), step={args.step}")

        replay_pong(
            args.h5,
            fps=args.fps,
            show_fields=(not args.no_fields) or args.video,
            sum_z=args.sum,
            start_frac=args.start_frac,
            until_frac=args.until_frac,
            video=args.video,
            video_output=args.video_output,
            use_npen=args.npen,
            step=args.step,
            change=args.change,
            time_scale=args.ts,
        )
    except Exception as e:
        print(f"Error during replay: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
