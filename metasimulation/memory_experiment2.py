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


def build_temporal_voltages(sim: MemoryElectrodes,
                            total: int,
                            idle: int,
                            voltage: int,
                            amplitude: float,
                            target_sequence) -> List[TemporalVoltage]:
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
    if isinstance(target_sequence, str) and target_sequence.strip().lower() == 'all':
        blocks = [list(range(6))]
    elif isinstance(target_sequence, (list, tuple, np.ndarray)):
        for entry in target_sequence:
            if _is_nan_like(entry):
                blocks.append(None)
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
    for bi, block in enumerate(blocks):
        # Apply active window for this block
        start = t
        end = min(start + max(int(voltage), 0), total)
        if block is not None and start < end:
            for p in block:
                pair_left[p][start:end] = float(amplitude)
                pair_right[p][start:end] = 0.0
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
):
    os.makedirs(outdir, exist_ok=True)

    # Build combinations
    import itertools
    exps = list(experiment_list) if experiment_list else ["random"]
    combos = list(itertools.product(
        dt_list,
        checkpoints, idle_steps_list, voltage_steps_list, total_steps_list,
        amplitude_list, targets_list,
        size_list, L_c_list,
        exps,
    ))
    if not combos:
        raise ValueError("No experiment combinations provided")

    for (dt,checkpoint, idle_steps, voltage_steps, total_steps, amplitude, target_seq, size, L_c, experiment) in combos:
        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[memory] {ts_human} -> Run: idle={idle_steps}, voltage={voltage_steps}, total={total_steps}, amp={amplitude}, target_sequence={target_seq}, L_c={L_c}, exp={experiment}, checkpoint={checkpoint}")

        # Create runner
        sim = MemoryElectrodes(dt=dt,L_c=L_c, applied_voltage=applied_voltage, nx=size[0], ny=size[1], nz=size[2], experiment=experiment)

        # Build stimulation sequences from block-wise target sequence
        stim = build_temporal_voltages(sim, total_steps, idle_steps, voltage_steps, amplitude, target_seq)

        # Target tag for filename
        def _blk_to_str(b):
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
            target_tag = "blk(" + ";".join(blocks_desc) + ")"
        elif isinstance(target_seq, str) and target_seq.strip().lower() == "all":
            target_tag = "blk(all)"
        else:
            target_tag = "blk(?)"

        cp_tag = os.path.basename(checkpoint) if checkpoint else 'nocp'
        base_name = (
            f"mem_idle{idle_steps}_volt{voltage_steps}_tot{total_steps}_amp{str(amplitude).replace('.', 'p')}_Lc{str(L_c).replace('.', 'p')}_exp{experiment}_{target_tag}_{dt}_{cp_tag}"
        )
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
            print(f"[memory] Saved: {out_path}")
        except Exception as e:
            print(f"[memory] Run failed: {e}")
        finally:
            if sleep_between_runs > 0:
                time.sleep(sleep_between_runs)


def main():
    
    

    memory_experiment_loop(
        checkpoints=[None],
        idle_steps_list=[50],
        voltage_steps_list=[50],
        total_steps_list=[300],
        amplitude_list=[1.0],
        targets_list=[[[0], [1], [0]]],
        measuring_voltage=2.0,
        dt_list=[0.001],
        L_c_list=[1e-3],
        applied_voltage=20.0,
        size_list=[(16, 16, 8)],
        k_reaction=0.0,
        outdir="metasimulation/output/memory",
        sleep_between_runs=0.0,
        experiment_list=["random"],
    )


if __name__ == '__main__':
    main()
