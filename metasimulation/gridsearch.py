import itertools
import os
import time
import traceback
from datetime import datetime

from pong_simulation.pong_sim_npen import PongSimulationNPEN, VisionImpairmentType


def gridsearch_loop(
    sim_ticks_list=[1],
    game_ticks_list=[1],
    num_steps_list=[10000],
    k_reaction_list=[0.01],
    electrode_type_list=["anode"],
    activation_list=["poly_normed", "poly_raw"],
    rl_list=[True],
    rl_steps_list=[100,50,400],
    rl_type_list=["idle", "-scramble+idle","backrow+-backrow", "-backrow+idle"],
    ramp_steps_list=[20],
    dt_list=[0.001],
    voltage_list=[20.0],
    dt_st_pair_list=[(0.001, 1)],
    size_list=[(16,16,4)],
    vision_impairment_type_list=[VisionImpairmentType.NONE],
    # Optional split transport coefficients to sweep (None -> use defaults D1/D2)
    d_diff1_list=[None],
    d_mig1_list=[None],
    d_diff2_list=[None],
    d_mig2_list=[None],
    checkpoint_list=[None],
    output_dir="metasimulation/output",
    sleep_between_runs=1.0,
):
    os.makedirs(output_dir, exist_ok=True)

    combos = list(itertools.product(
        sim_ticks_list, game_ticks_list, num_steps_list, k_reaction_list,
        electrode_type_list, activation_list, rl_list, rl_steps_list, rl_type_list, ramp_steps_list,
        dt_list, voltage_list, size_list, dt_st_pair_list,
        vision_impairment_type_list,
        d_diff1_list, d_mig1_list, d_diff2_list, d_mig2_list,
        checkpoint_list,
    ))
    if not combos:
        raise ValueError("No parameter combinations provided")

    print(f"[gridsearch] Starting infinite loop over {len(combos)} combinations…")

    while True:
        for (
            sim_ticks, game_ticks, num_steps, k_reaction,
            electrode_type, activation, rl, rl_steps, rl_type, ramp_steps,
            dt, voltage, size, dt_st_pair,
            vision_impairment_type,
            d_diff1, d_mig1, d_diff2, d_mig2,
            checkpoint,
        ) in combos:


            if dt_st_pair is not None:
                dt, sim_ticks = dt_st_pair

            ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Build a brief tag for split coefficients if provided
            split_tag_parts = []
            if d_diff1 is not None:
                split_tag_parts.append(f"dd1={d_diff1}")
            if d_mig1 is not None:
                split_tag_parts.append(f"dm1={d_mig1}")
            if d_diff2 is not None:
                split_tag_parts.append(f"dd2={d_diff2}")
            if d_mig2 is not None:
                split_tag_parts.append(f"dm2={d_mig2}")
            split_tag = (", " + ", ".join(split_tag_parts)) if split_tag_parts else ""

            print(
                f"[gridsearch] {ts_human} -> Run: sim_ticks={sim_ticks}, game_ticks={game_ticks}, num_steps={num_steps}, k_reaction={k_reaction}, electrode_type={electrode_type}, activation={activation}, rl={rl}, rl_steps={rl_steps}, rl_type={rl_type}, ramp_steps={ramp_steps}, dt={dt}, voltage={voltage}, size={size}, vision={getattr(vision_impairment_type, 'value', str(vision_impairment_type))}, checkpoint={checkpoint}{split_tag}"
            )

            # Build a unique, collision-resistant output file path
            base_name = (
                f"pong_sim_s{sim_ticks}_g{game_ticks}_n{num_steps}_k{str(k_reaction).replace('.', 'p')}"
            )
            if split_tag_parts:
                # Create a compact filename-safe tag
                fname_tag = "_" + "_".join(p.replace("=", "=") for p in split_tag_parts)
                base_name += fname_tag.replace(".", "p")
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
                runner = PongSimulationNPEN(dt=dt,
                                            applied_voltage=voltage,
                                            nx=size[0],
                                            ny=size[1],
                                            nz=size[2])
                # If split transport coefficients are provided, set them on the runner instance
                # These propagate into HDF5 constants and current decomposition utilities.
                if d_diff1 is not None:
                    runner.D_diff1 = float(d_diff1)
                if d_mig1 is not None:
                    runner.D_mig1 = float(d_mig1)
                if d_diff2 is not None:
                    runner.D_diff2 = float(d_diff2)
                if d_mig2 is not None:
                    runner.D_mig2 = float(d_mig2)
                # Save each run to its unique file path
                runner.run(
                    sim_ticks=sim_ticks,
                    game_ticks=game_ticks,
                    num_steps=num_steps,
                    k_reaction=k_reaction,
                    electrode_type=electrode_type,
                    activation=activation,
                    rl=rl,
                    rl_steps=rl_steps,
                    rl_type=rl_type,
                    ramp_steps=ramp_steps,
                    output_path=out_path,
                    vision_impairment_type=vision_impairment_type,
                    checkpoint=checkpoint,
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
