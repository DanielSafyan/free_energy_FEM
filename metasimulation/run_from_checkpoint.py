import os
import sys
import time
import traceback
from datetime import datetime

from pong_simulation.pong_sim_npen import PongSimulationNPEN


def run_from_checkpoint(
    checkpoint,
    sim_ticks=5,
    game_ticks=1,
    num_steps=10000,
    k_reaction=0.01,
    rl=False,
    rl_steps=4,
    output_dir="metasimulation/output",
):
    """
    Run a single PongSimulationNPEN continuation starting from an HDF5 checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to an existing HDF5 file containing prior simulation/game history.
    sim_ticks : int
        Number of NPEN steps per decision/update.
    game_ticks : int
        Number of game frames per decision/update.
    num_steps : int
        Number of decision/update cycles to run.
    k_reaction : float
        First-order reaction rate passed to solver step2().
    output_dir : str
        Directory to store the resulting HDF5 history file.

    Returns
    -------
    str
        The output HDF5 path written by this run.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    os.makedirs(output_dir, exist_ok=True)

    # Build a unique, collision-resistant output file path (mirrors gridsearch style)
    base_name = (
        f"pong_fromchk_s{sim_ticks}_g{game_ticks}_n{num_steps}_k{str(k_reaction).replace('.', 'p')}"
    )
    ts_ns = time.time_ns()
    fname = f"{base_name}_{ts_ns}.h5"
    out_path = os.path.join(output_dir, fname)
    suffix = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{base_name}_{ts_ns}_{suffix}.h5")
        suffix += 1

    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[run_from_checkpoint] {ts_human} -> Run: sim_ticks={sim_ticks}, game_ticks={game_ticks}, num_steps={num_steps}, "
        f"k_reaction={k_reaction}, checkpoint={checkpoint} -> out={out_path}"
    )

    runner = PongSimulationNPEN()
    runner.run(
        sim_ticks=sim_ticks,
        game_ticks=game_ticks,
        num_steps=num_steps,
        k_reaction=k_reaction,
        output_path=out_path,
        rl=rl,
        rl_steps=rl_steps,
        checkpoint=checkpoint,
    )

    return out_path


def main():
    # Minimal CLI: python -m metasimulation.run_from_checkpoint [checkpoint] [sim_ticks] [game_ticks] [num_steps] [k_reaction]
    # Falls back to output/pong_simulation.h5 if no checkpoint is provided.
    checkpoint = None
    if len(sys.argv) >= 2:
        checkpoint = sys.argv[1]
    else:
        checkpoint = os.path.join("output", "pong_simulation.h5")

    try:
        sim_ticks = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        game_ticks = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
        num_steps = int(sys.argv[4]) if len(sys.argv) >= 5 else 10000
        k_reaction = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.0
        rl = bool(sys.argv[6]) if len(sys.argv) >= 7 else False
        rl_steps = int(sys.argv[7]) if len(sys.argv) >= 8 else 40
    except Exception:
        print("[run_from_checkpoint] Warning: Failed to parse optional numeric args; using defaults.")
        sim_ticks, game_ticks, num_steps, k_reaction = 5, 1, 10000, 0.0
        rl = False
        rl_steps = 40

    try:
        out = run_from_checkpoint(
            checkpoint=checkpoint,
            sim_ticks=sim_ticks,
            game_ticks=game_ticks,
            num_steps=num_steps,
            k_reaction=k_reaction,
            rl=rl,
            rl_steps=rl_steps,
            )
        print(f"[run_from_checkpoint] Saved to: {out}")
    except Exception as e:
        print(f"[run_from_checkpoint] Run failed with error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
