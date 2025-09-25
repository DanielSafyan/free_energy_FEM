import itertools
import os
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence, Tuple

from pong_simulation.pong_sim_npen import VisionImpairmentType
from pong_simulation.colab_npen_simulation import PongSimulationNPENColab


def _make_unique_out(output_dir: str, base_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts_ns = time.time_ns()
    out_path = os.path.join(output_dir, f"{base_name}_{ts_ns}.h5")
    suffix = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{base_name}_{ts_ns}_{suffix}.h5")
        suffix += 1
    return out_path


def _run_single(
    combo: Tuple,
    output_dir: str,
    use_gpu: bool,
    gpu_assembly: bool,
    # Keep signature pickle-friendly for multiprocessing
) -> Optional[str]:
    """
    Worker function: runs a single simulation combo and returns output_path on success.
    combo order mirrors gridsearch_loop_colab packing.
    """
    # Headless for Colab
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    # Unpack
    (
        sim_ticks, game_ticks, num_steps, k_reaction,
        electrode_type, activation, rl, rl_steps,
        dt, voltage, size, dt_st_pair,
        vision_impairment_value, checkpoint,
    ) = combo

    try:
        if dt_st_pair is not None:
            dt, sim_ticks = dt_st_pair

        # Build base name and unique path
        base_name = (
            f"pong_sim_s{sim_ticks}_g{game_ticks}_n{num_steps}_k{str(k_reaction).replace('.', 'p')}"
        )
        out_path = _make_unique_out(output_dir, base_name)

        # Convert vision impairment back to Enum
        try:
            vision_impairment = VisionImpairmentType(vision_impairment_value)
        except Exception:
            vision_impairment = VisionImpairmentType.NONE

        runner = PongSimulationNPENColab(
            dt=dt,
            applied_voltage=voltage,
            nx=size[0], ny=size[1], nz=size[2],
            use_gpu=use_gpu,
            gpu_assembly=gpu_assembly,
        )
        runner.run(
            sim_ticks=sim_ticks,
            game_ticks=game_ticks,
            num_steps=num_steps,
            k_reaction=k_reaction,
            electrode_type=electrode_type,
            activation=activation,
            rl=rl,
            rl_steps=rl_steps,
            output_path=out_path,
            vision_impairment_type=vision_impairment,
            checkpoint=checkpoint,
        )
        return out_path
    except Exception as e:
        print(f"[gridsearch_colab worker] Error: {e}")
        print(traceback.format_exc())
        return None


def gridsearch_loop_colab(
    sim_ticks_list: Sequence[int] = (5,),
    game_ticks_list: Sequence[int] = (1,),
    num_steps_list: Sequence[int] = (10000, 20000),
    k_reaction_list: Sequence[float] = (0.01,),
    electrode_type_list: Sequence[str] = ("anode",),
    activation_list: Sequence[str] = ("poly_normed",),
    rl_list: Sequence[bool] = (False,),
    rl_steps_list: Sequence[int] = (8,),
    dt_list: Sequence[float] = (0.001,),
    voltage_list: Sequence[float] = (20.0,),
    dt_st_pair_list: Sequence[Optional[Tuple[float,int]]] = (None,),
    size_list: Sequence[Tuple[int,int,int]] = ((16,16,4), (32,32,4)),
    vision_impairment_type_list: Sequence[VisionImpairmentType] = (VisionImpairmentType.NONE,),
    checkpoint_list: Sequence[Optional[str]] = (None,),
    output_dir: str = "metasimulation/output",
    num_workers: int = 2,
    use_gpu: bool = True,
    gpu_assembly: bool = True,
    infinite: bool = False,
    sleep_between_runs: float = 0.0,
):
    """
    Parallel grid search tailored for Google Colab.

    - Uses ProcessPoolExecutor to run multiple simulations concurrently.
    - If use_gpu=True, all workers will attempt to use the single GPU. In practice,
      you typically want num_workers=1 with use_gpu=True to avoid VRAM contention.
      If you want to sweep many combos quickly, set num_workers>1 with use_gpu=False
      to run on CPU in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    combos = list(itertools.product(
        sim_ticks_list, game_ticks_list, num_steps_list, k_reaction_list,
        electrode_type_list, activation_list, rl_list, rl_steps_list,
        dt_list, voltage_list, size_list, dt_st_pair_list,
        # pass enum value string to make it pickle-stable across processes
        [v.value if isinstance(v, VisionImpairmentType) else str(v) for v in vision_impairment_type_list],
        checkpoint_list,
    ))
    if not combos:
        raise ValueError("No parameter combinations provided")

    print(f"[gridsearch_colab] Prepared {len(combos)} combinations; num_workers={num_workers}, use_gpu={use_gpu}, gpu_assembly={gpu_assembly}")

    def submit_all(executor: ProcessPoolExecutor):
        futs = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for combo in combos:
            print(f"[gridsearch_colab] {timestamp} -> queue {combo}")
            fut = executor.submit(_run_single, combo, output_dir, use_gpu, gpu_assembly)
            futs.append(fut)
        return futs

    if infinite:
        while True:
            with ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
                futures = submit_all(ex)
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        print(f"[gridsearch_colab] Finished: {res}")
                if sleep_between_runs > 0:
                    time.sleep(sleep_between_runs)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
            futures = submit_all(ex)
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    print(f"[gridsearch_colab] Finished: {res}")


def main():
    gridsearch_loop_colab()


if __name__ == "__main__":
    main()
