#!/usr/bin/env python3
"""
Run stimulation-only NPEN memory experiments by cycling through checkpoint(s)
and patterned voltage sequences of the form:
    idle, voltage, idle, voltage, ..., idle_end
where idle and voltage segment lengths stay constant within a run, and the final
idle_end fills remaining steps up to --total.

This uses simulations.electrodes_memory_npen.MemoryElectrodes in stimulation-only
mode (no game/current), preserving PongH5Reader-compatible HDF5 output.

Examples:
    python -m metasimulation.memory_experiment \
        --idle 100 --voltage 50 --total 1000 \
        --targets all --amplitude 1.0 \
        --checkpoints output/seed1.h5 output/seed2.h5 \
        --outdir metasimulation/output --dt 0.01 --volts 20.0 --size 16,16,4

    python -m metasimulation.memory_experiment \
        --idle 200 --voltage 100 --total 2000 \
        --targets 0,3,7 --amplitude 0.5 \
        --checkpoints output/seed1.h5 \
        --measuring 2.0
"""
import argparse
import os
import time
from datetime import datetime
from typing import List, Sequence

import numpy as np
import random

from simulations.electrodes_memory_npen import MemoryElectrodes
from utils.temporal_voltages import TemporalVoltage


def parse_size(arg: str):
    parts = arg.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--size must be 'nx,ny,nz'")
    return tuple(int(x) for x in parts)


def parse_targets(arg: str, allow_all: bool = True) -> Sequence[int] | str:
    s = arg.strip().lower()
    if allow_all and s == 'all':
        return 'all'
    if not s:
        return []
    return [int(x) for x in s.split(',')]


def build_repeating_sequence(total: int, idle: int, voltage: int, amplitude: float) -> np.ndarray:
    """Build a 1D time sequence of repeated idle/voltage cycles, ending with idle_end.

    IMPORTANT: Idle is encoded as NaN (no stimulation), and active voltage phase
    is encoded as `amplitude`. The final remainder (idle_end) is also NaN.
    This matches the NPEN usage, where NaN means "do not impose Dirichlet here".
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if idle < 0 or voltage < 0:
        raise ValueError("idle and voltage must be non-negative")
    if idle == 0 and voltage == 0:
        return np.zeros(total, dtype=float)

    cycle = idle + voltage
    if cycle == 0:
        return np.zeros(total, dtype=float)
    cycles = total // cycle
    rem = total - cycles * cycle

    seq = np.empty(0, dtype=float)
    if cycles > 0:
        one = np.concatenate([np.full(idle, np.nan, dtype=float), np.full(voltage, float(amplitude), dtype=float)])
        seq = np.tile(one, cycles)
    if rem > 0:
        # idle_end remainder encoded as NaN (no stimulation)
        seq = np.concatenate([seq, np.full(rem, np.nan, dtype=float)])
    return seq


def ramp_sequence_linearly(sequence: np.ndarray, ramp_steps: int):

    sequences = np.split(sequence, np.where(np.diff(sequence) != 0)[0] + 1)
    #print(sequences[0][:10])
    for seq_id,seq in enumerate(sequences):

        seq_ramped=[]
        if ramp_steps == 0 or seq[0] == np.nan or len(seq) < ramp_steps:
            seq_ramped = seq
        else: 
            ramp_sequence = np.linspace(0, seq[ramp_steps-1], ramp_steps)
            deramp_sequence = np.linspace(seq[-ramp_steps-1],0,ramp_steps)
            seq_ramped = np.concatenate([ramp_sequence, seq[ramp_steps:-ramp_steps], deramp_sequence])
        sequences[seq_id] = seq_ramped
    

    return np.concatenate(sequences)
    
    


def build_temporal_voltages(sim: MemoryElectrodes,
                            total: int,
                            idle: int,
                            voltage: int,
                            amplitude: float,
                            target_sequence,
                            seed: int,
                            ramp_steps: int = 0,
                            scramble_steps: int = 0,
                            scramble_voltage: float | None = None) -> List[TemporalVoltage]:
    """Construct TemporalVoltage objects from a block-wise target sequence.

    Semantics:
    - target_sequence is a list of blocks. Each block lasts `voltage` steps.
    - Between blocks, insert `idle` steps where all stimulating electrodes are idle (NaN).
    - A block can be:
        * int p in [0..5]: activate pair p for this block
        * list/tuple of ints: activate all listed pairs simultaneously for this block
        * np.nan/None/"idle": idle block (no activation)
    - After the last block, any remaining steps up to `total` remain idle (NaN), effectively padding the timeline.
    """
    stim_nodes = list(sim.voltage_indices[6:])
    if len(stim_nodes) != 12:
        raise ValueError(f"Expected 12 stimulating nodes, got {len(stim_nodes)}")

    def _is_nan_like(v):
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"nan", "idle", "none"}:
            return True
        try:
            return bool(np.isnan(v))
        except Exception:
            return False

    # Normalize target_sequence to list[ list[int] | None ]
    blocks: List[List[int] | None] = []
    if isinstance(target_sequence, str) and target_sequence == 'all':
        blocks = [list(range(6))]

    elif isinstance(target_sequence, (list, tuple, np.ndarray)):
        for entry in target_sequence:
            if _is_nan_like(entry):
                blocks.append(None)
            elif isinstance(entry, str) and 'scramble' in entry:
                blocks.append(entry)
            elif isinstance(entry, (list, tuple, np.ndarray)):
                pairs: List[int] = []
                for p in entry:
                    try:
                        pi = int(p)
                    except Exception:
                        continue
                    if 0 <= pi < 6:
                        pairs.append(pi)
                blocks.append(sorted(set(pairs)) if pairs else None)
            else:
                try:
                    pi = int(entry)
                except Exception:
                    pi = None
                blocks.append([pi] if (pi is not None and 0 <= pi < 6) else None)
    else:
        raise ValueError("target_sequence must be 'all' or a list of blocks (ints/lists or NaN)")

    # Prepare per-pair time sequences (NaN = idle/no Dirichlet)
    pair_left = [np.full(total, np.nan, dtype=float) for _ in range(6)]
    pair_right = [np.full(total, np.nan, dtype=float) for _ in range(6)]

    t = 0
    rng = np.random.default_rng(seed)
    for bi, block in enumerate(blocks):
        # Handle scramble blocks
        if isinstance(block, str) and 'scramble' in block:
            scramble_amp = scramble_voltage if scramble_voltage is not None else amplitude
            is_negative = block.startswith('-')
            applied_amplitude = -float(scramble_amp) if is_negative else float(scramble_amp)
            start = t
            end = min(start + int(scramble_steps), total)
            if start < end:
                for i in range(start, end):
                    p = rng.integers(0, 6)
                    pair_left[p][i] = applied_amplitude
                    pair_right[p][i] = 0.0
            t = end
        # Apply active window for this block (voltage phase)
        else:
            start = t
            end = min(start + max(int(voltage), 0), total)
            if block is not None and start < end:
                for p_val in block:
                    # Check for negative target encoding
                    p_str = str(p_val)
                    is_negative = p_str.startswith('-')
                    p = int(p_str)

                    applied_amplitude = -float(amplitude) if is_negative else float(amplitude)
                    pair_left[abs(p)][start:end] = applied_amplitude
                    pair_right[abs(p)][start:end] = 0.0
            t = end
        # Insert idle gap between blocks (except after last)
        if bi < len(blocks) - 1 and idle > 0 and t < total:

            t = min(t + int(idle), total)

    # Build TemporalVoltage list for pairs that were ever active


    tv_list: List[TemporalVoltage] = []
    for p in range(6):
        if not np.all(np.isnan(pair_left[p])):


            left_node = int(stim_nodes[2 * p])
            right_node = int(stim_nodes[2 * p + 1])
            if ramp_steps > 0:
                pair_left[p] = ramp_sequence_linearly(pair_left[p],ramp_steps=ramp_steps)
                pair_right[p] = ramp_sequence_linearly(pair_right[p],ramp_steps=ramp_steps)

            tv_list.append(TemporalVoltage(left_node, pair_left[p]))
            tv_list.append(TemporalVoltage(right_node, pair_right[p]))
    return tv_list


def memory_experiment_loop(
    checkpoints: Sequence[str],
    idle_steps_list: Sequence[int],
    voltage_steps_list: Sequence[int],
    total_steps_list: Sequence[int],
    amplitude_list: Sequence[float],
    targets_list: Sequence[Sequence[int] | str],
    measuring_voltage: float | None,
    dt_list: Sequence[float],
    applied_voltage: float,
    size_list: Sequence[tuple[int, int, int]],
    k_reaction: float,
    L_c_list: Sequence[float],
    outdir: str,
    sleep_between_runs: float = 0.0,
    experiment_list: Sequence[str] | None = None,
    ramp_steps_list: Sequence[int] | None = None,
    scramble_steps_list: Sequence[int] | None = None,
    scramble_voltage_list: Sequence[float | None] | None = None,
    ramp_voltage = True,
    make_plot = False
):
    os.makedirs(outdir, exist_ok=True)

    # Build combinations
    import itertools
    exps = list(experiment_list) if experiment_list else ["random"]
    scramble_volts = list(scramble_voltage_list) if scramble_voltage_list is not None else [None]
    combos = list(itertools.product(
        dt_list,
        checkpoints, idle_steps_list, voltage_steps_list, total_steps_list,
        amplitude_list, targets_list,
        size_list, L_c_list, ramp_steps_list, scramble_steps_list, scramble_volts,
        exps,
    ))
    if not combos:
        raise ValueError("No experiment combinations provided")

    for (dt,checkpoint, idle_steps, voltage_steps, total_steps, amplitude, target_seq, size, L_c,ramp_steps, scramble_steps, scramble_voltage, experiment) in combos:
        seed = random.randint(1, 100_000_000)
        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[memory] {ts_human} -> Run: idle={idle_steps}, voltage={voltage_steps}, total={total_steps}, amp={amplitude}, target_sequence={target_seq}, L_c={L_c}, exp={experiment}, checkpoint={checkpoint}")

        # Create runner
        sim = MemoryElectrodes(dt=dt,L_c=L_c, applied_voltage=applied_voltage, nx=size[0], ny=size[1], nz=size[2], experiment=experiment)

        # Build stimulation sequences from block-wise target sequence
        if not ramp_voltage:
            stim = build_temporal_voltages(sim, total_steps, idle_steps, voltage_steps, amplitude, target_seq, seed, scramble_steps=scramble_steps, scramble_voltage=scramble_voltage)
        else:
            stim = build_temporal_voltages(sim, total_steps, idle_steps, voltage_steps, amplitude, target_seq, seed, scramble_steps=scramble_steps, ramp_steps=ramp_steps, scramble_voltage=scramble_voltage)

        # Target tag for filename
        def _blk_to_str(b):
            if isinstance(b, str) and 'scramble' in b:
                return b
            if b is None:
                return "id"
            if isinstance(b, (list, tuple, np.ndarray)):
                parts = []
                for x in b:
                    try:
                        parts.append(str(int(x)))
                    except Exception:
                        continue
                return "+".join(parts) if parts else "id"
            try:
                return "id" if (isinstance(b, float) and np.isnan(b)) else str(int(b))
            except Exception:
                return "id"
        if isinstance(target_seq, (list, tuple, np.ndarray)):
            blocks_desc = [_blk_to_str(b) for b in target_seq]
            target_tag_full = "blk(" + ";".join(blocks_desc) + ")"
        elif isinstance(target_seq, str) and target_seq.strip().lower() == "all":
            target_tag_full = "blk(all)"
        else:
            target_tag_full = "blk(?)"

        # Build Windows-safe, compact base filename using inline slug helpers
        import re as _re
        def _slug(text: str, max_len: int) -> str:
            s = _re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.")
            if len(s) <= max_len:
                return s
            import hashlib as _hl
            h = _hl.sha1(str(text).encode("utf-8")).hexdigest()[:8]
            head = max(0, max_len - 9)
            return (s[:head] + "-" + h) if head > 0 else h

        amp_slug = _slug(str(amplitude).replace('.', 'p'), 8)
        Lc_slug = _slug(str(L_c).replace('.', 'p'), 8)
        exp_slug = _slug(str(experiment), 16)
        tgt_slug = _slug(target_tag_full, 48)
        dt_slug  = _slug(str(dt), 12)
        if checkpoint:
            cp_base = os.path.splitext(os.path.basename(checkpoint))[0]
            cp_slug = _slug(cp_base, 20)
        else:
            cp_slug = 'nocp'

        base_name = (
            f"mem_i{idle_steps}_v{voltage_steps}_t{total_steps}_amp{amp_slug}_Lc{Lc_slug}_exp{exp_slug}_tgt{tgt_slug}_dt{dt_slug}_cp{cp_slug}_seed{seed}"
        )
        # Ensure overall base length bound
        if len(base_name) > 96:
            import hashlib as _hl2
            h = _hl2.sha1(base_name.encode('utf-8')).hexdigest()[:10]
            base_name = base_name[:85] + "-" + h

        ts_ns = time.time_ns()
        out_path = os.path.join(outdir, f"{base_name}_{ts_ns}.h5")
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(outdir, f"{base_name}_{ts_ns}_{suffix}.h5")
            suffix += 1

        # Run stimulation-only
        try:
            sim.run(
                stim_voltages=stim,
                measuring_voltage=measuring_voltage,
                num_steps=total_steps,
                k_reaction=k_reaction,
                output_path=out_path,
                checkpoint=checkpoint,
            )
            # Enrich HDF5 with full, human-readable metadata (non-fatal if it fails)
            try:
                import h5py as _h5
                with _h5.File(out_path, 'a') as h5f:
                    h5f.attrs['target_tag_full'] = str(target_tag_full)
                    h5f.attrs['targets_list'] = str(target_seq)
                    h5f.attrs['experiment_name'] = str(experiment)
                    h5f.attrs['amplitude'] = float(amplitude)
                    h5f.attrs['idle_steps'] = int(idle_steps)
                    h5f.attrs['voltage_steps'] = int(voltage_steps)
                    h5f.attrs['total_steps'] = int(total_steps)
                    h5f.attrs['L_c'] = float(L_c)
                    if checkpoint:
                        h5f.attrs['checkpoint_basename'] = os.path.basename(checkpoint)
            except Exception:
                pass

            print(f"[memory] Saved: {out_path}")

            if make_plot:

                plot_results_in_folder(idle_steps, voltage_steps, total_steps,out_path,scramble_steps)
                print(f"[memory] Plotted")


        except Exception as e:
            import traceback
            traceback.print_exc()

            print(f"[memory] Run failed: {e}")
        finally:
            if sleep_between_runs > 0:
                time.sleep(sleep_between_runs)


def plot_results_in_folder(idle_steps, voltage_steps, total_steps,out_path,scramble_steps):
    """Extract, plot and save results for a given NPEN memory simulation HDF5.

    Creates a timestamp-named folder beside the .h5 with subfolders:
      - current/
      - concentration/
      - other/

    Then:
      1) Plot measured currents over time and save under current/
      2) Save concentration 'c' screenshots (mid-plane) at two frames per phase
         (idle/voltage), specifically: one at (phase_start + 1) and one at
         (phase_end - 1), clamped to [1, total_steps]. Phases repeat as
         idle, voltage, idle, voltage, ..., with a final idle remainder.
      3) Plot change of total concentration sum over time (Δ sum(c)) and save under other/
    """

    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt

    # Lazy import readers and helpers to avoid heavy module load at import time
    from pong_simulation.pong_sim_npen import PongH5Reader
    from visualization.pong_replay import _build_midplane_triangulation

    # Ensure non-interactive backend for headless environments
    try:
        plt.switch_backend('Agg')
    except Exception:
        pass

    # -----------------------------
    # 1) Prepare output folders
    # -----------------------------
    base_dir = os.path.dirname(out_path)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    # Name the folder the same as the .h5 stem (without extension)
    out_root = os.path.join(base_dir, stem)
    out_current = os.path.join(out_root, "current")
    out_conc = os.path.join(out_root, "concentration")
    out_other = os.path.join(out_root, "other")
    os.makedirs(out_current, exist_ok=True)
    os.makedirs(out_conc, exist_ok=True)
    os.makedirs(out_other, exist_ok=True)

    # -----------------------------
    # 2) Read data lazily
    # -----------------------------
    with PongH5Reader(out_path) as data:
        attrs = dict(data.attrs)
        dt = float(attrs.get('dt', 0.0)) if ('dt' in attrs) else 0.0

        # Time indexing: writer appends an initial state at t=0, then total_steps updates
        # So c.shape[0] == total_steps + 1 typically
        T_states = int(data.c.shape[0])
        total_in_file = max(0, T_states - 1)
        usable_total = int(min(total_steps, total_in_file)) if total_steps is not None else total_in_file

        # -----------------------------
        # 2a) Plot measured currents over time
        # -----------------------------
        try:
            I_full = np.asarray(data.measured_current[...], dtype=float)  # (T,3)
        except Exception:
            I_full = None

        if I_full is not None and I_full.ndim == 2 and I_full.shape[1] == 3:
            T_I = I_full.shape[0]
            # Build time vector aligned with I entries
            if dt > 0 and np.isfinite(dt):
                t_I = np.arange(T_I, dtype=float) * dt
                x_label = 'time (s)'
            else:
                t_I = np.arange(T_I, dtype=float)
                x_label = 'time (steps)'

            good = np.isfinite(I_full).all(axis=1)
            I = I_full[good]
            t_I = t_I[good]

            fig_cur, ax_cur = plt.subplots(1, 1, figsize=(9, 4.8))
            # Style inspired by data_analyze.ipynb plot_currents
            if I.size > 0:
                l1, = ax_cur.plot(t_I, I[:, 0], label='I2', color='#1f77b4', alpha=1, linestyle='-')
                l2, = ax_cur.plot(t_I, I[:, 1], label='I1', color='#ff7f0e', alpha=1, linestyle='-')
                l3, = ax_cur.plot(t_I, I[:, 2], label='I0', color='#2ca02c', alpha=1, linestyle='-')
                ax_cur.set_xlabel(x_label)
                ax_cur.set_ylabel('Current (A)')
                ax_cur.grid(True, alpha=0.3)
                ax_cur.legend(loc='best')
                ax_cur.set_title('Measurement currents')
            fig_cur.tight_layout()
            fig_cur.savefig(os.path.join(out_current, 'currents.png'), dpi=150)
            plt.close(fig_cur)

        # -----------------------------
        # 2b) Prepare triangulation for concentration screenshots (mid-plane, NPEN)
        # -----------------------------
        nodes = np.array(data.nodes[...], dtype=float)
        xy_mid, tri_mid, slice_idx = _build_midplane_triangulation(nodes, attrs)

        def tri_face_values(field_1d: np.ndarray) -> np.ndarray:
            """Compute per-triangle face colors by averaging node values."""
            vals = field_1d[slice_idx]
            return vals[tri_mid].mean(axis=1)

        # Axis labels for mid-plane (XY)
        axis_x_label, axis_y_label = "x (m)", "y (m)"

        # Helper: render a single concentration frame at global index g
        def save_c_frame(g_idx: int, out_png: str):
            c_faces = tri_face_values(np.asarray(data.c[g_idx], dtype=float))
            # Compute display time
            if dt > 0 and np.isfinite(dt):
                t_val = g_idx * dt
                t_label = 's'
            else:
                t_val = g_idx
                t_label = 'steps'
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            coll = ax.tripcolor(xy_mid[:, 0], xy_mid[:, 1], tri_mid, facecolors=c_faces, cmap='plasma')
            ax.triplot(xy_mid[:, 0], xy_mid[:, 1], tri_mid, 'k-', lw=0.1, alpha=0.3)
            ax.set_title(f"c @ t = {t_val:.5f} {t_label}")
            ax.set_aspect('equal')
            ax.set_xlabel(axis_x_label)
            ax.set_ylabel(axis_y_label)
            fig.colorbar(coll, ax=ax, label="Concentration c")
            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)

        # -----------------------------
        # 2c) Determine phases and save two screenshots per phase
        # -----------------------------
        # Phases begin at t=1 (after the initial state at t=0)
        phases = []  # list of (label, start_g, end_g) inclusive, in global indices
        tptr = 1
        # Helper to append a phase of given length with label if length>0
        def _add_phase(lbl: str, length: int):
            nonlocal tptr
            if length is None or length <= 0:
                return
            if tptr > usable_total:
                return
            start = tptr
            end = min(tptr + int(length) - 1, usable_total)
            if end >= start:
                phases.append((lbl, start, end))
            tptr = end + 1

        # Repeat cycles until we fill usable_total steps. Determine first phase
        # type from the voltage_pattern at the first update (g=1): if any of the
        # stimulating electrode entries (last 12 of length-18 vector) are finite
        # (including 0.0), treat as 'voltage'; if all NaN, treat as 'idle'.
        first_is_voltage = False
        try:
            vp1 = np.asarray(data.voltage_pattern[1], dtype=float)
            stim_slice = vp1[-12:] if vp1.size >= 12 else vp1
            if stim_slice.size > 0:
                # active if any value is finite (including zeros)
                first_is_voltage = np.any(np.isfinite(stim_slice))
        except Exception:
            # Fallback to the original assumption (idle first)
            first_is_voltage = False

        idle_len = max(0, int(idle_steps))
        volt_len = max(0, int(voltage_steps))
        scramble_len = max(0, int(scramble_steps))
        cycle = idle_len + volt_len + scramble_len
        while tptr <= usable_total:
            if first_is_voltage:
                _add_phase('voltage', volt_len)
                if tptr > usable_total:
                    break
                _add_phase('idle', idle_len)
            else:
                _add_phase('idle', idle_len)
                if tptr > usable_total:
                    break
                _add_phase('voltage', volt_len)
            if cycle == 0:
                # Avoid infinite loop if both lengths are zero
                break

        # Final remainder after last complete cycle is implicitly treated as idle by _add_phase logic

        # For each phase, save two frames: start and end-1 (clamped)
        used_frames = set()
        for k, (lbl, g_start, g_end) in enumerate(phases):
            if g_end < g_start:
                continue
            # Pick indices, ensuring within [1, usable_total]
            a = int(np.clip(g_start, 1, usable_total))
            b = int(np.clip(g_end - 1, 1, usable_total))
            # If very short phase, frames might collapse; still attempt unique saves
            candidates = []
            candidates.append(('a', a))
            if b != a:
                candidates.append(('b', b))
            for tag, g in candidates:
                key = (k, g)
                if key in used_frames:
                    continue
                used_frames.add(key)
                fname = f"phase_{k:03d}_{lbl}_t{g:05d}_{tag}.png"
                save_c_frame(g, os.path.join(out_conc, fname))

        # -----------------------------
        # 2d) Plot absolute sum(c) over time with scaled y-axis
        # -----------------------------
        T = T_states
        times = (np.arange(T, dtype=float) * dt) if (dt > 0 and np.isfinite(dt)) else np.arange(T, dtype=float)
        # Compute sum(c) per time step lazily
        c_sums = np.zeros(T, dtype=np.float64)
        for t in range(T):
            c_sums[t] = float(np.nansum(np.asarray(data.c[t], dtype=float)))
        y_min = float(np.min(c_sums))
        y_max = float(np.max(c_sums))
        rng = max(1e-12, y_max - y_min)
        margin = 0.05 * rng

        fig_d, ax_d = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_d.plot(times, c_sums, label='sum(c)', color='k')
        ax_d.set_xlabel('time (s)' if (dt > 0 and np.isfinite(dt)) else 'timestep')
        ax_d.set_ylabel('sum over nodes')
        ax_d.set_title('Sum of concentration over time')
        ax_d.grid(True, alpha=0.3)
        ax_d.set_ylim(y_min - margin, y_max + margin)
        ax_d.legend(loc='best')
        fig_d.tight_layout()
        fig_d.savefig(os.path.join(out_other, 'sum_c.png'), dpi=150)
        plt.close(fig_d)

    return


def main():  
    

    memory_experiment_loop(
        checkpoints=[None],
        idle_steps_list=[0],
        voltage_steps_list=[125],
        total_steps_list=[1500],
        amplitude_list=[1.0],
        targets_list=[[[0], '-scramble', [0], [0], [0], '-scramble',[0], [0], [0]],
                    [[0], 'scramble', [0], [0], [0], 'scramble',[0], [0], [0]],
                    [['-0'], '-scramble', ['-0'], ['-0'], ['-0'], '-scramble',['-0'], ['-0'], ['-0']],
                    [['-0'], 'scramble', ['-0'], ['-0'], ['-0'], 'scramble',['-0'], ['-0'], ['-0']]],
        measuring_voltage=2.0,
        dt_list=[0.001],
        L_c_list=[1e-2],
        applied_voltage=20.0,
        size_list=[(16, 16, 8)],
        k_reaction=0.0,
        outdir="metasimulation/output/memory/10-10-25/scramble0vs-scramble-0",
        sleep_between_runs=0.0,
        experiment_list=["gradientx-"],
        ramp_steps_list=[20],
        scramble_steps_list=[125],
        scramble_voltage_list=[0.5],
        make_plot = True,
    )


if __name__ == '__main__':
    main()
