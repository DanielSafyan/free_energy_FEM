from metasimulation.memory_experiment2 import *
import numpy as np

memory_experiment_loop(
        checkpoints=[None],
        idle_steps_list=[100,10],
        voltage_steps_list=[200,400],
        total_steps_list=[10000],
        amplitude_list=[1.0],
        targets_list=[[[2],[2],[2],[0],[0],[0],[1],[1],[1], [0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]]],
        measuring_voltage=2.0,
        dt_list=[0.001],
        L_c_list=[1e-2],
        applied_voltage=20.0,
        size_list=[(16, 16, 8)],
        k_reaction=0.0,
        outdir="metasimulation/experiments/short_run_memory",
        sleep_between_runs=0.0,
        experiment_list=["gradientz","gradientx-","gradienty-","gradientz-","gradientx","gradienty"],
        ramp_steps_list=[20],#
        make_plot = True,
    )