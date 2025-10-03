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
                            targets: Sequence[int] | str) -> List[TemporalVoltage]:
    """Construct TemporalVoltage objects for selected stimulating PAIRS.

    - sim.voltage_indices[6:] gives the 12 stimulating node indices, grouped as 6 pairs.
    - `targets` supports two modes:
        1) Static set: 'all' or a list of PAIR indices in [0..5] => all selected pairs are active in every cycle (V,0 during voltage, idle otherwise).
        2) Sequence per cycle: a list whose elements can be ints in [0..5] or 'nan'/None/np.nan. Each entry selects the active pair for that cycle,
           or idle if nan-like. Example: [1, 2, np.nan] => first cycle pair 1 active, second cycle pair 2 active, third cycle idle.
      Non-targeted pairs are omitted (interpreted as NaN by run()).
    """
    stim_nodes = list(sim.voltage_indices[6:])
    if len(stim_nodes) != 12:
        raise ValueError(f"Expected 12 stimulating nodes, got {len(stim_nodes)}")

    # Helper to detect nan-like entries
    def _is_nan_like(v):
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"nan", "idle"}:
            return True
        try:
            return bool(np.isnan(v))
        except Exception:
            return False

    cycle = idle + voltage
    cycles = total // cycle if cycle > 0 else 0

    # Sequence mode if targets contains any nan-like or non-int entries
    sequence_mode = False
    if isinstance(targets, (list, tuple)):
        for _v in targets:
            if _is_nan_like(_v) or not isinstance(_v, (int, np.integer)):
                sequence_mode = True
                break

    tv_list: List[TemporalVoltage] = []
    if targets == 'all' and not sequence_mode:
        # Static: all pairs active during every voltage window
        base_seq = build_repeating_sequence(total, idle, voltage, amplitude)
        active_mask = ~np.isnan(base_seq)
        pair_zero_seq = np.where(active_mask, 0.0, np.nan).astype(float)
        for p in range(6):
            left_node = int(stim_nodes[2 * p])
            right_node = int(stim_nodes[2 * p + 1])
            tv_list.append(TemporalVoltage(left_node, base_seq.copy()))
            tv_list.append(TemporalVoltage(right_node, pair_zero_seq.copy()))
        return tv_list

    if not sequence_mode:
        # Static: explicit subset of pairs active in every voltage window
        pair_indices = list(targets) if isinstance(targets, (list, tuple)) else [int(targets)]
        for p in pair_indices:
            if p < 0 or p >= 6:
                raise ValueError("target PAIR indices must be in [0..5]")
        base_seq = build_repeating_sequence(total, idle, voltage, amplitude)
        active_mask = ~np.isnan(base_seq)
        pair_zero_seq = np.where(active_mask, 0.0, np.nan).astype(float)
        for p in pair_indices:
            left_node = int(stim_nodes[2 * p])
            right_node = int(stim_nodes[2 * p + 1])
            tv_list.append(TemporalVoltage(left_node, base_seq.copy()))
            tv_list.append(TemporalVoltage(right_node, pair_zero_seq.copy()))
        return tv_list

    # Sequence mode: per-cycle pair selection or idle
    # Build blank sequences for all 6 pairs (omit later if always NaN)
    pair_left = [np.full(total, np.nan, dtype=float) for _ in range(6)]
    pair_right = [np.full(total, np.nan, dtype=float) for _ in range(6)]

    for ci in range(cycles):
        sel = None
        if isinstance(targets, (list, tuple)) and ci < len(targets):
            sel = targets[ci]
        # Normalize selection
        if _is_nan_like(sel):
            continue  # idle this cycle
        try:
            p = int(sel)
        except Exception:
            continue
        if p < 0 or p >= 6:
            continue
        # Voltage window for this cycle
        start = ci * cycle + idle
        end = min(start + voltage, total)
        if start >= total or start >= end:
            continue
        pair_left[p][start:end] = float(amplitude)
        pair_right[p][start:end] = 0.0

    # Emit only pairs that are ever active (non-all-NaN)
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
        size_list,
        exps,
    ))
    if not combos:
        raise ValueError("No experiment combinations provided")

    for (dt,checkpoint, idle_steps, voltage_steps, total_steps, amplitude, targets, size, experiment) in combos:
        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[memory] {ts_human} -> Run: idle={idle_steps}, voltage={voltage_steps}, total={total_steps}, amp={amplitude}, targets={targets}, exp={experiment}, checkpoint={checkpoint}")

        # Create runner
        sim = MemoryElectrodes(dt=dt, applied_voltage=applied_voltage, nx=size[0], ny=size[1], nz=size[2], experiment=experiment)

        # Build per-electrode sequences
        print(targets)
        stim = build_temporal_voltages(sim, total_steps, idle_steps, voltage_steps, amplitude, targets)

        # Build output path
        # Robust target tag: handle per-cycle sequences with NaN/idle entries
        if targets == 'all':
            target_tag = 'all'
        else:
            seq = targets if isinstance(targets, (list, tuple, np.ndarray)) else [targets]
            parts = []
            for x in seq:
                if isinstance(x, str) and x.strip().lower() in {"nan", "idle"}:
                    parts.append("id")
                    continue
                try:
                    if np.isnan(x):
                        parts.append("id")
                        continue
                except Exception:
                    pass
                try:
                    parts.append(str(int(x)))
                except Exception:
                    parts.append("id")
            target_tag = "t" + ",".join(parts)
        cp_tag = os.path.basename(checkpoint) if checkpoint else 'nocp'
        base_name = (
            f"mem_idle{idle_steps}_volt{voltage_steps}_tot{total_steps}_amp{str(amplitude).replace('.', 'p')}_exp{experiment}_{target_tag}_{cp_tag}"
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
    # Run memory experiment
    memory_experiment_loop(
        checkpoints=[None],
        idle_steps_list=[0,200,100,300],
        voltage_steps_list=[1000,20,40,160,320,640],
        total_steps_list=[900],
        amplitude_list=[1.0],
        targets_list=[[1, 2, np.nan]],
        measuring_voltage=2.0,
        dt_list=[0.01,0.001],
        applied_voltage=20.0,
        size_list=[(16,16,4),(16,16,8)],
        k_reaction=0.0,
        outdir="metasimulation/output/memory",
        sleep_between_runs=1.0,
        experiment_list=["random"],
    )


if __name__ == '__main__':
    main()
