import itertools
import os
import time
import traceback
from datetime import datetime

from pong_simulation.pong_simulation import PongSimulation


def gridsearch_loop(
    sim_ticks_list=(1, 2, 3),
    game_ticks_list=(6, 12),
    num_steps_list=(50, 100),
    k_reaction_list=(0.1, 0.5, 1.0),
    output_dir="metasimulation/output",
    sleep_between_runs=1.0,
):
    os.makedirs(output_dir, exist_ok=True)

    combos = list(itertools.product(sim_ticks_list, game_ticks_list, num_steps_list, k_reaction_list))
    if not combos:
        raise ValueError("No parameter combinations provided")

    print(f"[gridsearch] Starting infinite loop over {len(combos)} combinations…")

    while True:
        for sim_ticks, game_ticks, num_steps, k_reaction in combos:
            ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[gridsearch] {ts_human} -> Run: sim_ticks={sim_ticks}, game_ticks={game_ticks}, num_steps={num_steps}, k_reaction={k_reaction}"
            )

            # Build a unique, collision-resistant output file path
            base_name = (
                f"pong_sim_s{sim_ticks}_g{game_ticks}_n{num_steps}_k{str(k_reaction).replace('.', 'p')}"
            )
            # Use high-resolution timestamp to minimize collision chance
            ts_ns = time.time_ns()
            fname = f"{base_name}_{ts_ns}.h5"
            out_path = os.path.join(output_dir, fname)
            # Collision avoidance loop (very unlikely to trigger)
            suffix = 1
            while os.path.exists(out_path):
                out_path = os.path.join(output_dir, f"{base_name}_{ts_ns}_{suffix}.h5")
                suffix += 1

            try:
                runner = PongSimulation()
                # Save each run to its unique file path
                runner.run(
                    sim_ticks=sim_ticks,
                    game_ticks=game_ticks,
                    num_steps=num_steps,
                    k_reaction=k_reaction,
                    output_path=out_path,
                )
            except Exception as e:
                print(f"[gridsearch] Run failed with error: {e}")
                print(traceback.format_exc())
            finally:
                if sleep_between_runs > 0:
                    time.sleep(sleep_between_runs)


def main():
    # Simple entrypoint with default grids. You can edit lists above or adapt to argparse if needed.
    gridsearch_loop()


if __name__ == "__main__":
    main()
