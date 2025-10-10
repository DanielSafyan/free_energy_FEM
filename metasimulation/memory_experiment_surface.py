#!/usr/bin/env python3
"""
Stimulation-only NPEN memory experiments using SURFACE electrodes + SG/EAFE.
Mirrors metasimulation/memory_experiment2.py but calls MemoryElectrodesSurface.

Non-CLI: edit parameters in main() and run.
"""
import os, time
from datetime import datetime
from typing import List, Sequence
import numpy as np, random

from simulations.electrodes_memory_npen_surface import MemoryElectrodesSurface


def build_repeating_sequence(total: int, idle: int, voltage: int, amplitude: float) -> np.ndarray:
    """Utility retained for parity, currently unused here."""
    if total <= 0:
        raise ValueError("total must be positive")
    cycle = idle + voltage
    if cycle <= 0:
        return np.zeros(total, dtype=float)
    cycles = total // cycle
    rem = total - cycles * cycle
    one = np.concatenate([np.full(max(idle,0), np.nan, dtype=float), np.full(max(voltage,0), float(amplitude), dtype=float)])
    seq = np.tile(one, max(cycles,0))
    if rem > 0:
        seq = np.concatenate([seq, np.full(rem, np.nan, dtype=float)])
    return seq


def build_surface_sequences(total: int, idle: int, voltage: int, amplitude: float, target_blocks, seed: int) -> List[np.ndarray]:
    """Build per-surface time sequences (12 arrays of shape (total,)).
    Blocks semantics mirror memory_experiment2: entries may be ints 0..5 (pair id), lists of ints, or None/NaN for idle.
    Left surface gets amplitude, right surface 0 for now (to match pair-left stimulation behavior in node-based flow).
    """
    rng = np.random.default_rng(seed)
    seqs_left = [np.full(total, np.nan, dtype=float) for _ in range(6)]
    seqs_right = [np.full(total, np.nan, dtype=float) for _ in range(6)]
    t = 0
    blocks = target_blocks if isinstance(target_blocks, (list, tuple, np.ndarray)) else []
    for bi, b in enumerate(blocks):
        start = t; end = min(start + max(int(voltage), 0), total)
        if b is not None and start < end:
            ids = b if isinstance(b,(list,tuple,np.ndarray)) else [b]
            for p in ids:
                try:
                    pid = int(p)
                except Exception:
                    continue
                if pid < 0 or pid >= 6: continue
                seqs_left[pid][start:end] = float(amplitude)
                seqs_right[pid][start:end] = 0.0
        t = end
        if bi < len(blocks) - 1 and idle > 0 and t < total:
            t = min(t + int(idle), total)
    # order: [L0,R0,L1,R1,...,L5,R5]
    out: List[np.ndarray] = []
    for p in range(6):
        out.append(seqs_left[p]); out.append(seqs_right[p])
    return out


def surface_memory_experiment_loop(
    idle_steps_list: list[int],
    voltage_steps_list: list[int],
    total_steps_list: list[int],
    amplitude_list: list[float],
    targets_list: list[Sequence[int] | Sequence[Sequence[int]] | None],
    dt_list: list[float],
    L_c_list: list[float],
    size_list: list[tuple[int,int,int]],
    applied_voltage: float,
    k_reaction: float,
    outdir: str,
    experiment_list: list[str] | None = None,
):
    os.makedirs(outdir, exist_ok=True)
    import itertools
    exps = list(experiment_list) if experiment_list else ["random"]
    combos = list(itertools.product(
        dt_list, idle_steps_list, voltage_steps_list, total_steps_list,
        amplitude_list, targets_list, size_list, L_c_list, exps,
    ))
    if not combos:
        raise ValueError("No experiment combinations provided")

    for (dt, idle_steps, voltage_steps, total_steps, amplitude, targets, size, L_c, experiment) in combos:
        seed = random.randint(1, 100_000_000)
        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[surface-mem] {ts_human} -> Run: idle={idle_steps}, voltage={voltage_steps}, total={total_steps}, amp={amplitude}, targets={targets}, L_c={L_c}, exp={experiment}")

        sim = MemoryElectrodesSurface(dt=dt, L_c=L_c, nx=size[0], ny=size[1], nz=size[2], experiment=experiment)
        sequences = build_surface_sequences(total_steps, idle_steps, voltage_steps, amplitude, targets, seed)

        base = f"surf_i{idle_steps}_v{voltage_steps}_t{total_steps}_amp{amplitude}_dt{dt}_Lc{L_c}_seed{seed}"
        out_path = os.path.join(outdir, base + f"_{time.time_ns()}.h5")
        saved = sim.run(surface_voltages=sequences, applied_voltage=applied_voltage, num_steps=total_steps,
                        k_reaction=k_reaction, output_path=out_path)
        print(f"[surface-mem] Saved: {saved}")

def main():
    # Edit these parameters and run the script/module directly.
    surface_memory_experiment_loop(
        idle_steps_list=[0],
        voltage_steps_list=[50],
        total_steps_list=[100],
        amplitude_list=[1.0],
        targets_list=[[[0], [1], [2]]],  # list of blocks; each element can be an int (pair id) or list of ints
        dt_list=[0.01],
        L_c_list=[1e-2],
        size_list=[(16,16,4)],
        applied_voltage=20.0,
        k_reaction=0.5,
        outdir="metasimulation/output_surface",
        experiment_list=["random"],
    )

if __name__ == '__main__':
    main()
