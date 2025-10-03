import numpy as np
import os
from tqdm import tqdm
try:
    import pygame
except Exception:
    pygame = None
import h5py
import sys
from enum import Enum



from utils.fem_mesh import TetrahedralMesh, create_structured_mesh_3d
from simulations.NPENwithFOReaction import NPENwithFOReaction

# Try to import the Hybrid NPEN simulation class
try:
    from pong_simulation.hybrid_npen_simulation import HybridNPENwithFOReaction
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("Hybrid NPEN simulation not available. Using standard Python implementation.")
from utils.temporal_voltages import TemporalVoltage, NPhasesVoltage

try:
    from gameplay.pong_game import PongGame
except Exception:
    PongGame = None
from simulations.electrode_3d_npp import get_node_idx as _unused_external_get_node_idx
from pong_simulation.pong_sim_npen import PongSimulationNPEN,PongH5Reader,VisionImpairmentType,calculate_current
import pong_simulation.pong_sim_npen as pong_npen_mod


# -----------------------------
# HDF5 logging utilities
# -----------------------------
def _create_ext_dataset(h5f, name, shape_tail, dtype=np.float64):
    maxshape = (None,) + tuple(shape_tail)
    chunks = (1,) + tuple(shape_tail)
    return h5f.create_dataset(
        name,
        shape=(0,) + tuple(shape_tail),
        maxshape=maxshape,
        chunks=chunks,
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
    )

def init_h5_output(mesh, meta, constants):
    """Initialize HDF5 file and extendable datasets.

    Returns (h5f, dsets) where dsets is a dict of datasets.
    """
    os.makedirs("output", exist_ok=True)
    h5_path = os.path.join("output", "pong_simulation.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)
    h5f = h5py.File(h5_path, mode="w")

    # Global metadata
    h5f.attrs.update({
        "Lx": meta["Lx"],
        "Ly": meta["Ly"],
        "Lz": meta["Lz"],
        "nx": meta["nx"],
        "ny": meta["ny"],
        "nz": meta["nz"],
        "num_nodes": mesh.num_nodes(),
        "num_cells": mesh.num_cells(),
        "dt": meta["dt"],
        "num_steps": meta["num_steps"],
        "experiment": meta["experiment"],
    })

    # Constants as attributes on a group
    const_grp = h5f.create_group("constants")
    const_grp.attrs.update(constants)

    # Mesh (static)
    h5f.create_dataset("mesh/nodes", data=mesh.nodes, compression="gzip", compression_opts=4)
    h5f.create_dataset("mesh/elements", data=mesh.elements, compression="gzip", compression_opts=4)

    N = mesh.num_nodes()
    dsets = {
        "c": _create_ext_dataset(h5f, "states/c", (N,), np.float64),
        "phi": _create_ext_dataset(h5f, "states/phi", (N,), np.float64),
        "ball": _create_ext_dataset(h5f, "game/ball_pos", (2,), np.float64),
        "platform": _create_ext_dataset(h5f, "game/platform_pos", tuple(), np.float64),
        "score": _create_ext_dataset(h5f, "game/score", tuple(), np.int32),
        "current": _create_ext_dataset(h5f, "measurements/measured_current", (3,), np.float64),
        "voltage": _create_ext_dataset(h5f, "electrodes/voltage_pattern", (18,), np.float64),
    }

    return h5f, dsets

def load_pong_h5(path: str = os.path.join("output", "pong_simulation.h5"), eager: bool = False):
    """Load data from HDF5 output.

    - If eager=False (default): returns a PongH5Reader for lazy, memory-efficient access.
    - If eager=True: loads arrays into RAM and returns a nested dict; file is closed.
    """
    if not eager:
        return PongH5Reader(path)

    # Eager load: read all arrays then close file
    with h5py.File(path, "r") as f:
        data = {
            "attrs": dict(f.attrs),
            "constants": dict(f["constants"].attrs),
            "mesh": {
                "nodes": f["mesh/nodes"][...],
                "elements": f["mesh/elements"][...],
            },
            "states": {
                "c": f["states/c"][...],
                "phi": f["states/phi"][...],
            },
            "game": {
                "ball_pos": f["game/ball_pos"][...],
                "platform_pos": f["game/platform_pos"][...],
                # Backwards compatibility: include score only if present
                "score": (f["game/score"][...] if "game/score" in f else None),
            },
            "measurements": {
                "measured_current": f["measurements/measured_current"][...],
            },
            "electrodes": {
                "voltage_pattern": f["electrodes/voltage_pattern"][...],
            },
        }
    return data

def get_last_state_from_h5(path: str):
    """Return the last state of the fields and game from an NPEN HDF5 file.

    Returns a dict with keys:
      - 'c': np.ndarray (N,)
      - 'phi': np.ndarray (N,)
      - 'ball_pos_xy': tuple(float, float)
      - 'platform_pos': float
      - 'score': Optional[int]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with PongH5Reader(path) as data:
        num_states = int(data.c.shape[0])
        if num_states == 0:
            raise ValueError("Checkpoint contains no states to resume from.")
        t_last = num_states - 1

        # Extract last field states
        c_last = np.array(data.c[t_last])
        phi_last = np.array(data.phi[t_last])

        # Extract last game state
        try:
            ball_last = tuple(np.array(data.ball_pos[t_last]).tolist())
        except Exception as e:
            raise ValueError(f"Failed to read last ball position: {e}")
        try:
            platform_last = float(np.array(data.platform_pos[t_last]))
        except Exception as e:
            raise ValueError(f"Failed to read last platform position: {e}")

        score_last = None
        if getattr(data, "score", None) is not None:
            if data.score.shape[0] > 0:
                try:
                    score_last = int(np.array(data.score[t_last]))
                except Exception:
                    score_last = None

    return {
        "c": c_last,
        "phi": phi_last,
        "ball_pos_xy": ball_last,
        "platform_pos": platform_last,
        "score": score_last,
    }



class MemoryElectrodes(PongSimulationNPEN):
    """Encapsulates the Pong + NPEN simulation for reusable runs.
        default dimensions:   
        - time step: 0.01 s
        - domain size: 1x1x0.25 mm
        - grid resolution: 16x16x4  
    """

    def __init__(self,
                 Lx=4.0, Ly=4.0, Lz=1.0,
                 screen_width=600, screen_height=600,
                 R=8.314, T=298.0, F=96485.33,
                 epsilon=80 * 8.854e-12,
                 D1=1.33e-9, D2=2.03e-9, D3=1e-9,
                 z1=1, z2=-1,
                 chi=0.0,
                 applied_voltage=20,
                 c0=10.0,
                 L_c=1e-2,
                 dt=1e-2,
                 nx=16, ny=16, nz=4,
                 experiment="random"):
        super().__init__(Lx, Ly, Lz, screen_width, screen_height, R, T, F, epsilon, D1, D2, D3, z1, z2, chi, applied_voltage, c0, L_c, dt, nx, ny, nz, experiment)


    # -----------------------------
    # HDF5 logging utilities (instance)
    # -----------------------------
    def _create_ext_dataset(self, h5f, name, shape_tail, dtype=np.float64):
        return _create_ext_dataset(h5f, name, shape_tail, dtype)

    def _init_h5_output(self, meta, constants, output_path=None):
        # Determine target HDF5 path: use provided output_path if given, otherwise default
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            h5_path = output_path
        else:
            os.makedirs("output", exist_ok=True)
            h5_path = os.path.join("output", "pong_simulation.h5")
        h5f = h5py.File(h5_path, mode="w")
        # Remember where we wrote the raw simulation history
        self._h5_path = h5_path

        # Global metadata
        h5f.attrs.update({
            "Lx": meta["Lx"],
            "Ly": meta["Ly"],
            "Lz": meta["Lz"],
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "num_nodes": self.mesh.num_nodes(),
            "num_cells": self.mesh.num_cells(),
            "dt": meta["dt"],
            "num_steps": meta["num_steps"],
            "experiment": meta["experiment"],
        })

        # Constants as attributes on a group
        const_grp = h5f.create_group("constants")
        const_grp.attrs.update(constants)

        # Mesh (static)
        h5f.create_dataset("mesh/nodes", data=self.mesh.nodes, compression="gzip", compression_opts=4)
        h5f.create_dataset("mesh/elements", data=self.mesh.elements, compression="gzip", compression_opts=4)

        N = self.mesh.num_nodes()
        dsets = {
            "c": self._create_ext_dataset(h5f, "states/c", (N,), np.float64),
            "phi": self._create_ext_dataset(h5f, "states/phi", (N,), np.float64),
            "ball": self._create_ext_dataset(h5f, "game/ball_pos", (2,), np.float64),
            "platform": self._create_ext_dataset(h5f, "game/platform_pos", tuple(), np.float64),
            "score": self._create_ext_dataset(h5f, "game/score", tuple(), np.int32),
            "current": self._create_ext_dataset(h5f, "measurements/measured_current", (3,), np.float64),
            "voltage": self._create_ext_dataset(h5f, "electrodes/voltage_pattern", (18,), np.float64),
        }

        return h5f, dsets

    def _append_row(self, ds, row):
        new_len = ds.shape[0] + 1
        ds.resize((new_len,) + ds.shape[1:])
        ds[-1] = row

    def _append_initial_state(self, dsets, pong_game, c, phi):
        self._append_row(dsets["c"], c)
        self._append_row(dsets["phi"], phi)
        self._append_row(dsets["ball"], np.array(pong_game.get_ball_position(), dtype=np.float64))
        self._append_row(dsets["platform"], float(pong_game.get_platform_position()))
        self._append_row(dsets["score"], int(pong_game.score))
        # placeholders to align lengths for step 0
        self._append_row(dsets["current"], np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        self._append_row(dsets["voltage"], np.full(18, np.nan, dtype=np.float64))

    def _append_initial_state_no_game(self, dsets, c, phi):
        """Append initial fields and placeholder game/measurement rows."""
        self._append_row(dsets["c"], c)
        self._append_row(dsets["phi"], phi)
        # Placeholder game data for compatibility with PongH5Reader
        self._append_row(dsets["ball"], np.array([np.nan, np.nan], dtype=np.float64))
        self._append_row(dsets["platform"], np.nan)
        self._append_row(dsets["score"], 0)
        # placeholders to align lengths for step 0
        self._append_row(dsets["current"], np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        self._append_row(dsets["voltage"], np.full(18, np.nan, dtype=np.float64))

    def _append_step(self, dsets, c, phi, ball_pos_xy, platform_pos, score, measured_current, voltage_amount):
        self._append_row(dsets["c"], c)
        self._append_row(dsets["phi"], phi)
        self._append_row(dsets["ball"], np.array(ball_pos_xy, dtype=np.float64))
        self._append_row(dsets["platform"], float(platform_pos))
        self._append_row(dsets["score"], int(score))
        self._append_row(dsets["current"], np.array(measured_current, dtype=np.float64))
        self._append_row(dsets["voltage"], np.array(voltage_amount, dtype=np.float64))

    def _init_conditions(self, checkpoint: str | None = None):
        """Initialize c, phi either from a checkpoint file or from presets.

        If a checkpoint path is provided, restores the last state from that HDF5
        file using get_last_state_from_h5() and validates that the node count
        matches the current mesh. Otherwise, uses the configured experiment preset.
        """
        if checkpoint is not None:
            last = get_last_state_from_h5(checkpoint)
            c, phi = last["c"], last["phi"]
            if c.shape[0] != self.mesh.num_nodes():
                raise ValueError(
                    f"Checkpoint node count ({c.shape[0]}) does not match current mesh ({self.mesh.num_nodes()})."
                )
            return c, phi

        if self.experiment == "gaussian":
            center_x, center_y, center_z = self.Lx / 2, self.Ly / 2, self.Lz / 2
            sigma = self.Lx / 10
            c = 0.05 + 0.04 * np.exp(-((self.nodes[:, 0] - center_x) ** 2 +
                                       (self.nodes[:, 1] - center_y) ** 2 +
                                       (self.nodes[:, 2] - center_z) ** 2) / (2 * sigma ** 2))
        elif isinstance(self.experiment, str) and self.experiment.startswith("gradient"):
            # Gradient presets over x/y/z with optional combinations and modifiers:
            #   gradientx, gradienty, gradientz, gradientxy, gradientxz, gradientxyz
            # Modifiers:
            #   '+' => steeper slope; '-' => reverse direction.
            # Example: 'gradientx+' (steeper along +x), 'gradientyz-' (reverse along y&z)
            exp = self.experiment
            sign = 1.0
            steep_mul = 1.0
            if exp.endswith('+'):
                steep_mul = 2.0
                exp = exp[:-1]
            elif exp.endswith('-'):
                sign = -1.0
                exp = exp[:-1]

            axes_part = exp[len("gradient"):]
            if not axes_part:
                axes_part = 'x'  # default to x if not specified

            # Normalized coordinates in [0,1]
            Lx = max(float(self.Lx), 1e-12)
            Ly = max(float(self.Ly), 1e-12)
            Lz = max(float(self.Lz), 1e-12)
            x_norm = self.nodes[:, 0] / Lx
            y_norm = self.nodes[:, 1] / Ly
            z_norm = self.nodes[:, 2] / Lz

            comps = []
            if 'x' in axes_part:
                comps.append(x_norm)
            if 'y' in axes_part:
                comps.append(y_norm)
            if 'z' in axes_part:
                comps.append(z_norm)
            if not comps:
                comps = [x_norm]

            avg_comp = sum(comps) / float(len(comps))
            slope_base = 0.10  # baseline slope magnitude
            slope = slope_base * steep_mul * sign
            # Baseline concentration similar to other presets
            c = 0.25 + slope * avg_comp
            # Optional clipping to reasonable bounds
            c = np.clip(c, 1e-6, 1.0)
        elif self.experiment == "random":
            c = 0.25 + np.random.uniform(-0.1, 0.1, self.mesh.num_nodes())
        else:
            c = np.full(self.mesh.num_nodes(), 0.5)
        phi = np.zeros(self.mesh.num_nodes())
        return c, phi

    def run(self, electrode_type="anode",activation = "poly_normed",rl=False,
            rl_steps=8,sim_ticks=1, game_ticks=6, num_steps=50, k_reaction=0.5,
            output_path=None, checkpoint=None,
            vision_impairment_type: "VisionImpairmentType" = VisionImpairmentType.NONE, 
            rl_diffusion=False,
            stim_voltages=None, measuring_voltage=2.0, fill_missing="nan"):
        """
        Run the simulation.

        If stim_voltages is provided, runs in stimulation-only mode without the game.
        Otherwise, defers to the base Pong+NPEN behavior.

        Args
        - stim_voltages: list of TemporalVoltage objects (or anything with
          .node_index and .time_sequence) specifying per-node time sequences for
          stimulating electrodes. node_index must be one of self.voltage_indices.
          Values are scaled by self.applied_voltage to get physical volts.
        - measuring_voltage: if None (default), measuring electrodes are not
          imposed (written as NaN in voltage pattern). If a float, each measuring
          pair is set to [measuring_voltage, 0] per time step.
        - fill_missing: how to fill stimulating electrodes without a provided
          sequence. "nan" (default) means no Dirichlet is imposed for that node;
          "zero" means Dirichlet 0 V is imposed.
        """
        # Stimulation-only branch
        if stim_voltages is not None:
            # Validate and normalize stim_voltages to a mapping {node_index: seq}
            seq_map = {}
            for tv in stim_voltages:
                if not hasattr(tv, "node_index") or not hasattr(tv, "time_sequence"):
                    raise TypeError("Each stim_voltages item must have node_index and time_sequence attributes.")
                if not isinstance(tv.time_sequence, np.ndarray) or tv.time_sequence.ndim != 1:
                    raise ValueError("time_sequence must be a 1D numpy array.")
                seq_map[int(tv.node_index)] = tv.time_sequence.astype(float)

            # Determine timeline length
            if num_steps is None:
                if len(seq_map) == 0:
                    raise ValueError("stim_voltages is empty and num_steps is None.")
                total_steps = max(len(seq) for seq in seq_map.values())
            else:
                total_steps = int(num_steps)

            # Initial conditions (optionally from checkpoint)
            c, phi = self._init_conditions(checkpoint=checkpoint)

            # HDF5
            meta = {
                "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
                "nx": self.nx, "ny": self.ny, "nz": self.nz,
                "dt": self.dt, "num_steps": total_steps,
                "experiment": self.experiment,
            }
            phi_c_val = getattr(self.sim, 'phi_c', (self.R * self.T / self.F) if self.F != 0 else 1.0)
            d_diff1_meta = getattr(self, 'D_diff1', self.D1)
            d_mig1_meta  = getattr(self, 'D_mig1',  self.D1)
            d_diff2_meta = getattr(self, 'D_diff2', self.D2)
            d_mig2_meta  = getattr(self, 'D_mig2',  self.D2)
            constants = {
                "R": self.R, "T": self.T, "F": self.F, "epsilon": self.epsilon,
                "D1": self.D1, "D2": self.D2, "D3": self.D3,
                "D_diff1": float(d_diff1_meta), "D_mig1": float(d_mig1_meta),
                "D_diff2": float(d_diff2_meta), "D_mig2": float(d_mig2_meta),
                "z1": self.z1, "z2": self.z2, "chi": self.chi, "c0": self.c0,
                "phi_c": float(phi_c_val),
                "k_reaction": k_reaction,
                "applied_voltage": self.applied_voltage,
            }
            h5f, dsets = self._init_h5_output(meta, constants, output_path)
            # Append initial placeholders (no game)
            self._append_initial_state_no_game(dsets, c, phi)

            # Pre-compute measuring pattern per step
            if measuring_voltage is None:
                measuring_vec = [np.nan] * 6
            else:
                mv = float(measuring_voltage)
                measuring_vec = [mv, 0, mv, 0, mv, 0]

            # Voltage indices
            meas_idx = self.voltage_indices[:6]
            stim_idx = self.voltage_indices[6:]
            if len(stim_idx) != 12:
                raise ValueError(f"Expected 12 stimulating nodes, got {len(stim_idx)}")

            # Choose fill value for missing stimulating sequences
            missing_val = np.nan if str(fill_missing).lower() == "nan" else 0.0

            # Provide globals for calculate_current without refactoring it (local module)
            global mesh, nodes, elements, R, T, F, D_diff1, D_mig1, D_diff2, D_mig2, z1, z2, c0, sim
            mesh = self.mesh
            nodes = self.nodes
            elements = self.elements
            R, T, F = self.R, self.T, self.F
            D_diff1 = getattr(self, 'D_diff1', self.D1)
            D_mig1  = getattr(self, 'D_mig1',  self.D1)
            D_diff2 = getattr(self, 'D_diff2', self.D2)
            D_mig2  = getattr(self, 'D_mig2',  self.D2)
            z1, z2 = self.z1, self.z2
            c0 = self.c0
            sim = self.sim
            # Also set globals on the defining module so calculate_current sees them
            try:
                pong_npen_mod.mesh = self.mesh
                pong_npen_mod.nodes = self.nodes
                pong_npen_mod.elements = self.elements
                pong_npen_mod.R, pong_npen_mod.T, pong_npen_mod.F = self.R, self.T, self.F
                pong_npen_mod.D_diff1 = getattr(self, 'D_diff1', self.D1)
                pong_npen_mod.D_mig1  = getattr(self, 'D_mig1',  self.D1)
                pong_npen_mod.D_diff2 = getattr(self, 'D_diff2', self.D2)
                pong_npen_mod.D_mig2  = getattr(self, 'D_mig2',  self.D2)
                pong_npen_mod.z1, pong_npen_mod.z2 = self.z1, self.z2
                pong_npen_mod.c0 = self.c0
                pong_npen_mod.sim = self.sim
            except Exception:
                pass
            # Reset calculate_current caches to avoid stale geometry across runs (both aliases)
            try:
                if hasattr(calculate_current, "_node_to_elements"):
                    calculate_current._node_to_elements = None
                if hasattr(calculate_current, "_node_faces"):
                    calculate_current._node_faces = {}
                if hasattr(calculate_current, "_consts"):
                    calculate_current._consts = None
                if hasattr(pong_npen_mod, 'calculate_current'):
                    fn = pong_npen_mod.calculate_current
                    if hasattr(fn, "_node_to_elements"):
                        fn._node_to_elements = None
                    if hasattr(fn, "_node_faces"):
                        fn._node_faces = {}
                    if hasattr(fn, "_consts"):
                        fn._consts = None
            except Exception:
                pass

            try:
                for t in tqdm(range(total_steps), desc="Stimulus Simulation Progress"):
                    # Build voltage pattern for this step
                    stim_vec = []
                    for node in stim_idx:
                        seq = seq_map.get(int(node), None)
                        if seq is None:
                            stim_val = missing_val
                        else:
                            val = seq[t] if t < len(seq) else 0.0
                            stim_val = float(val) * float(self.applied_voltage)
                        stim_vec.append(stim_val)
                    voltage_amount = measuring_vec + stim_vec

                    # One physics step
                    c_prev = c.copy()
                    c, phi = self.sim.step2(c_prev, phi, self.voltage_indices, voltage_amount, k_reaction=k_reaction)

                    # Log step (placeholders for game/current)
                    self._append_step(
                        dsets,
                        c,
                        phi,
                        (np.nan, np.nan),
                        np.nan,
                        0,
                        pong_npen_mod.calculate_current(c, phi, [self.voltage_indices[0], self.voltage_indices[2], self.voltage_indices[4]]),
                        voltage_amount,
                    )
            finally:
                h5f.flush()
                h5f.close()
                try:
                    print(f"Saved history to: {self._h5_path}")
                except Exception:
                    pass
            return


if __name__ == "__main__":
    sim_runner = PongSimulationNPEN(dt=0.1)
    sim_runner.run(sim_ticks=5, game_ticks=10, num_steps=20, k_reaction=0.01, rl=False, rl_steps=4)
