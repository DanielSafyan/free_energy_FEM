import numpy as np
import os
from tqdm import tqdm
import pygame
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
from utils.temporal_voltages import NPhasesVoltage

from gameplay.pong_game import PongGame
from simulations.electrode_3d_npp import get_node_idx as _unused_external_get_node_idx

class VisionImpairmentType(Enum):
    NONE = "NONE"
    DELAYED = "DELAYED"
    RANDOM_FULL = "RANDOM_FULL"
    RANDOM_FIRSTROW = "RANDOM_FIRSTROW"
    CONTINUOUS = "CONTINUOUS"
 
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

def _append_row(ds, row):
    new_len = ds.shape[0] + 1
    ds.resize((new_len,) + ds.shape[1:])
    ds[-1] = row

def append_initial_state(dsets, pong_game, c, phi):
    _append_row(dsets["c"], c)
    _append_row(dsets["phi"], phi)
    _append_row(dsets["ball"], np.array(pong_game.get_ball_position(), dtype=np.float64))
    _append_row(dsets["platform"], float(pong_game.get_platform_position()))
    _append_row(dsets["score"], int(pong_game.score))
    # placeholders to align lengths for step 0
    _append_row(dsets["current"], np.array([np.nan, np.nan, np.nan], dtype=np.float64))
    _append_row(dsets["voltage"], np.full(18, np.nan, dtype=np.float64))

def append_step(dsets, c, phi, ball_pos_xy, platform_pos, score, measured_current, voltage_amount):
    _append_row(dsets["c"], c)
    _append_row(dsets["phi"], phi)
    _append_row(dsets["ball"], np.array(ball_pos_xy, dtype=np.float64))
    _append_row(dsets["platform"], float(platform_pos))
    _append_row(dsets["score"], int(score))
    _append_row(dsets["current"], np.array(measured_current, dtype=np.float64))
    _append_row(dsets["voltage"], np.array(voltage_amount, dtype=np.float64))

# -----------------------------
# HDF5 reading utilities
# -----------------------------
class PongH5Reader:
    """Lightweight lazy reader for pong_simulation.h5 (NPEN).

    Usage:
        with PongH5Reader("output/pong_simulation.h5") as data:
            c_t0 = data.c[0]   # lazy slicing
            times = data.c.shape[0]
    """
    def __init__(self, path: str = os.path.join("output", "pong_simulation.h5")):
        self._path = path
        self._f = h5py.File(path, "r")
        # Attributes / constants
        self.attrs = dict(self._f.attrs)
        self.constants = dict(self._f["constants"].attrs)
        # Mesh
        self.nodes = self._f["mesh/nodes"]
        self.elements = self._f["mesh/elements"]
        # Time-series datasets (lazy h5py.Dataset objects)
        self.c = self._f["states/c"]
        self.phi = self._f["states/phi"]
        self.ball_pos = self._f["game/ball_pos"]
        self.platform_pos = self._f["game/platform_pos"]
        # Backwards compatibility: older files may not have score
        try:
            self.score = self._f["game/score"]
        except KeyError:
            self.score = None
        self.measured_current = self._f["measurements/measured_current"]
        self.voltage_pattern = self._f["electrodes/voltage_pattern"]

    def close(self):
        if self._f and self._f.id:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


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

def init_voltage():
    # Deprecated in favor of PongSimulation._init_voltage()
    raise NotImplementedError("Use PongSimulation._init_voltage() which uses instance nx, ny, nz.")


def calculate_current(c, phi, measuring_indices):
    """
    Compute electric current (A) at each measuring electrode for NPEN (single salt c).

    Returns a 3-tuple of currents corresponding to the three indices in measuring_indices.

    Runtime optimizations:
    - Precompute, once, for each electrode node: connected elements and incident faces' geometry.
    - Precompute constant factors from physical parameters.
    """
    # Access globals prepared in __main__
    global mesh, nodes, elements, R, T, F, D_diff1, D_mig1, D_diff2, D_mig2, z1, z2, c0, sim

    # Lazy init of caches
    if not hasattr(calculate_current, "_node_to_elements"):
        calculate_current._node_to_elements = None
    if not hasattr(calculate_current, "_node_faces"):
        # maps node_idx -> list of (elem_idx, [ (unit_normal, area), (..), (..) ])
        calculate_current._node_faces = {}
    if not hasattr(calculate_current, "_consts"):
        calculate_current._consts = None

    # Build node -> connected elements map once
    if calculate_current._node_to_elements is None:
        node_to_elements = [[] for _ in range(mesh.num_nodes())]
        for elem_idx in range(mesh.num_cells()):
            elem_nodes = mesh.elements[elem_idx]
            for n in elem_nodes:
                node_to_elements[n].append(elem_idx)
        calculate_current._node_to_elements = node_to_elements

    # Precompute constants once (uses sim.phi_c for scaling phi from dimensionless to V)
    if calculate_current._consts is None:
        phi_c = sim.phi_c
        # Factors for flux terms (split diffusion vs migration)
        K_GRAD_C = (-D_diff1 * c0, -D_diff2 * c0)
        K_MIG = (-(z1 * F * D_mig1 / (R * T)) * phi_c, -(z2 * F * D_mig2 / (R * T)) * phi_c)
        calculate_current._consts = (K_GRAD_C, K_MIG)

    (K_GRAD_C, K_MIG) = calculate_current._consts

    def build_node_faces(node_idx):
        """Precompute per-element incident faces for a node: unit normals and areas."""
        faces_per_elem = []
        p_i = mesh.nodes[node_idx]
        connected_elements = calculate_current._node_to_elements[node_idx]
        for elem_idx in connected_elements:
            if elem_idx not in mesh._element_data:
                continue
            elem_nodes = mesh.elements[elem_idx]
            # local index of node in element
            try:
                local_i = int(np.where(elem_nodes == node_idx)[0][0])
            except Exception:
                # Should not happen, but guard anyway
                continue
            # other local indices
            others = [j for j in range(4) if j != local_i]
            # define three faces and corresponding opposite indices
            face_pairs = [
                (others[0], others[1], others[2]),  # face (i, a, b), opp c
                (others[0], others[2], others[1]),  # face (i, a, c), opp b
                (others[1], others[2], others[0]),  # face (i, b, c), opp a
            ]
            face_list = []
            for a, b, opp in face_pairs:
                p_a = mesh.nodes[elem_nodes[a]]
                p_b = mesh.nodes[elem_nodes[b]]
                p_opp = mesh.nodes[elem_nodes[opp]]
                v1 = p_a - p_i
                v2 = p_b - p_i
                normal = np.cross(v1, v2)  # magnitude = 2*area, direction by right-hand rule
                # Ensure outward orientation from tetrahedron
                if np.dot(normal, (p_opp - p_i)) > 0.0:
                    normal = -normal
                normn = np.linalg.norm(normal)
                if normn < 1e-18:
                    continue
                area = 0.5 * normn
                unit_normal = normal / normn  # equals normal / (2*area)
                face_list.append((unit_normal, area))
            if face_list:
                faces_per_elem.append((elem_idx, face_list))
        return faces_per_elem

    # Build face caches for requested nodes if missing
    for node_idx in measuring_indices:
        if node_idx not in calculate_current._node_faces:
            calculate_current._node_faces[node_idx] = build_node_faces(node_idx)

    currents = []
    for electrode_node_idx in measuring_indices:
        total_current_at_step = 0.0
        for elem_idx, face_list in calculate_current._node_faces[electrode_node_idx]:
            if elem_idx not in mesh._element_data:
                continue
            elem_nodes_indices = mesh.elements[elem_idx]

            # Local fields (NPEN)
            c_local = c[elem_nodes_indices]
            phi_local = phi[elem_nodes_indices]  # dimensionless inside solver

            grads = mesh._element_data[elem_idx]['grads']  # physical-space gradients (1/m)

            # Element-wise gradients (physical units)
            grad_c = np.dot(c_local, grads)             # 1/m
            grad_phi = np.dot(phi_local, grads)         # 1/m, later scaled by phi_c in constants

            # Average physical concentration [mol/m^3]
            c_avg_phys = float(np.mean(c_local) * c0)

            # Current density vector J_elem [A/m^2]
            # NPEN uses a single salt concentration c for both ions
            J_elem = F * (
                z1 * (K_GRAD_C[0] * grad_c + K_MIG[0] * c_avg_phys * grad_phi)
                +
                z2 * (K_GRAD_C[1] * grad_c + K_MIG[1] * c_avg_phys * grad_phi)
            )

            # Integrate J·n over the three faces incident to the electrode node
            for unit_normal, face_area in face_list:
                current_through_face = float(np.dot(J_elem, unit_normal)) * face_area
                total_current_at_step += current_through_face / 3.0

        currents.append(total_current_at_step)

    return tuple(currents)


class PongSimulationNPEN:
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
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height

        # Physical constants
        self.R, self.T, self.F = R, T, F
        self.epsilon = epsilon
        self.D1, self.D2, self.D3 = D1, D2, D3
        self.z1, self.z2 = z1, z2
        self.chi = chi
        self.applied_voltage = applied_voltage
        self.c0 = c0
        self.L_c = L_c
        self.dt = dt
        self.experiment = experiment
        # Grid resolution
        self.nx, self.ny, self.nz = nx, ny, nz

        # Mesh
        self.nodes, self.elements, self.boundary_nodes = create_structured_mesh_3d(
            Lx=self.Lx, Ly=self.Ly, Lz=self.Lz, nx=self.nx, ny=self.ny, nz=self.nz
        )
        self.mesh = TetrahedralMesh(self.nodes, self.elements)

        # Simulation core
        use_cpp = True
        if HYBRID_AVAILABLE and use_cpp:
            self.sim = HybridNPENwithFOReaction(
                self.mesh, self.dt, self.D1, self.D2, self.D3, self.z1, self.z2,
                self.epsilon, self.R, self.T, self.L_c, self.c0,
                voltage=self.applied_voltage,
                alpha=0.5, alpha_phi=0.5,
                chemical_potential_terms=[],
                boundary_nodes=self.boundary_nodes,
            )
        else:
            self.sim = NPENwithFOReaction(
                self.mesh, self.dt, self.D1, self.D2, self.D3, self.z1, self.z2,
                self.epsilon, self.R, self.T, self.L_c, self.c0,
                voltage=self.applied_voltage,
                alpha=0.5, alpha_phi=0.5,
                chemical_potential_terms=[],
                boundary_nodes=self.boundary_nodes,
            )

        # Electrodes
        self.voltage_indices = self._init_voltage()

    # -----------------------------
    # Mesh indexing helpers
    # -----------------------------
    def get_node_idx(self, i, j, k):
        """Map i,j,k grid indices to node index for (nx,ny,nz)."""
        return i * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k

    def _init_voltage(self):
        """Compute electrode node indices using instance grid size."""
        nx, ny, nz = self.nx, self.ny, self.nz
        gi = self.get_node_idx
        sensing_electrode11_idx = gi(nx//4, ny//4, 1)
        sensing_electrode12_idx = gi(nx//4, ny//4, nz-1)
        sensing_electrode21_idx = gi(nx//4, 2*ny//4, 1)
        sensing_electrode22_idx = gi(nx//4, 2*ny//4, nz-1)
        sensing_electrode31_idx = gi(nx//4, 3*ny//4, 1)
        sensing_electrode32_idx = gi(nx//4, 3*ny//4, nz-1)

        # 3 stimulating electrode pairs in the middle row at y = 2*ny//4
        stimulating_electrode11_idx = gi(2*nx//4, ny//4, 1)
        stimulating_electrode12_idx = gi(2*nx//4, ny//4, nz-1)
        stimulating_electrode21_idx = gi(2*nx//4, 2*ny//4, 1)
        stimulating_electrode22_idx = gi(2*nx//4, 2*ny//4, nz-1)
        stimulating_electrode31_idx = gi(2*nx//4, 3*ny//4, 1)
        stimulating_electrode32_idx = gi(2*nx//4, 3*ny//4, nz-1)

        # 3 stimulating electrode pairs in the upper row at y = 3*ny//4
        stimulating_electrode41_idx = gi(3*nx//4, ny//4, 1)
        stimulating_electrode42_idx = gi(3*nx//4, ny//4, nz-1)
        stimulating_electrode51_idx = gi(3*nx//4, 2*ny//4, 1)
        stimulating_electrode52_idx = gi(3*nx//4, 2*ny//4, nz-1)
        stimulating_electrode61_idx = gi(3*nx//4, 3*ny//4, 1)
        stimulating_electrode62_idx = gi(3*nx//4, 3*ny//4, nz-1)

        return [
            sensing_electrode11_idx,
            sensing_electrode12_idx,
            sensing_electrode21_idx,
            sensing_electrode22_idx,
            sensing_electrode31_idx,
            sensing_electrode32_idx,
            stimulating_electrode11_idx,
            stimulating_electrode12_idx,
            stimulating_electrode21_idx,
            stimulating_electrode22_idx,
            stimulating_electrode31_idx,
            stimulating_electrode32_idx,
            stimulating_electrode41_idx,
            stimulating_electrode42_idx,
            stimulating_electrode51_idx,
            stimulating_electrode52_idx,
            stimulating_electrode61_idx,
            stimulating_electrode62_idx,
        ]

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

    def _append_step(self, dsets, c, phi, ball_pos_xy, platform_pos, score, measured_current, voltage_amount):
        self._append_row(dsets["c"], c)
        self._append_row(dsets["phi"], phi)
        self._append_row(dsets["ball"], np.array(ball_pos_xy, dtype=np.float64))
        self._append_row(dsets["platform"], float(platform_pos))
        self._append_row(dsets["score"], int(score))
        self._append_row(dsets["current"], np.array(measured_current, dtype=np.float64))
        self._append_row(dsets["voltage"], np.array(voltage_amount, dtype=np.float64))

    def _init_conditions(self):
        if self.experiment == "gaussian":
            center_x, center_y, center_z = self.Lx / 2, self.Ly / 2, self.Lz / 2
            sigma = self.Lx / 10
            c = 0.05 + 0.04 * np.exp(-((self.nodes[:, 0] - center_x) ** 2 +
                                       (self.nodes[:, 1] - center_y) ** 2 +
                                       (self.nodes[:, 2] - center_z) ** 2) / (2 * sigma ** 2))
        elif self.experiment == "random":
            c = 0.25 + np.random.uniform(-0.1, 0.1, self.mesh.num_nodes())
        else:
            c = np.full(self.mesh.num_nodes(), 0.5)
        phi = np.zeros(self.mesh.num_nodes())
        return c, phi

    def run(self, electrode_type="anode",activation = "poly_normed",rl=False,
            rl_steps=8, rl_type="idle", ramp_steps=20, sim_ticks=1, game_ticks=6, num_steps=50, k_reaction=0.5,
            output_path=None, checkpoint=None,
            vision_impairment_type: "VisionImpairmentType" = VisionImpairmentType.NONE, 
            rl_diffusion=False):
        # Initialize pygame/game
        pygame.init()
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Single Player Pong")
        clock = pygame.time.Clock()
        pong_game = PongGame(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, False)

        # Initial states (NPEN: c, phi). If a checkpoint is provided, restore from it.
        if checkpoint is not None:
            last = get_last_state_from_h5(checkpoint)
            c, phi = last["c"], last["phi"]
            # Validate compatibility with current mesh
            if c.shape[0] != self.mesh.num_nodes():
                raise ValueError(
                    f"Checkpoint node count ({c.shape[0]}) does not match current mesh ({self.mesh.num_nodes()})."
                )
            # Restore game state
            try:
                pong_game.set_platform_position(int(last["platform_pos"]))
            except Exception:
                pass
            try:
                bx, by = last["ball_pos_xy"]
                pong_game.ball.x = int(bx)
                pong_game.ball.y = int(by)
            except Exception:
                pass
            if last.get("score") is not None:
                try:
                    pong_game.score = int(last["score"])
                except Exception:
                    pass
        else:
            c, phi = self._init_conditions()

        # HDF5
        meta = {
            "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dt": self.dt, "num_steps": num_steps,
            "experiment": self.experiment,
        }
        # Include phi_c (thermal voltage) for visualization tools
        phi_c_val = getattr(self.sim, 'phi_c', (self.R * self.T / self.F) if self.F != 0 else 1.0)
        # Include split transport coefficients in metadata; fall back to legacy D1/D2
        d_diff1_meta = getattr(self, 'D_diff1', self.D1)
        d_mig1_meta  = getattr(self, 'D_mig1',  self.D1)
        d_diff2_meta = getattr(self, 'D_diff2', self.D2)
        d_mig2_meta  = getattr(self, 'D_mig2',  self.D2)
        constants = {
            "R": self.R, "T": self.T, "F": self.F, "epsilon": self.epsilon,
            # Legacy keys retained for backward compatibility
            "D1": self.D1, "D2": self.D2, "D3": self.D3,
            # New split coefficients
            "D_diff1": float(d_diff1_meta), "D_mig1": float(d_mig1_meta),
            "D_diff2": float(d_diff2_meta), "D_mig2": float(d_mig2_meta),
            "z1": self.z1, "z2": self.z2, "chi": self.chi, "c0": self.c0,
            "phi_c": float(phi_c_val),
            "k_reaction": k_reaction,
            "applied_voltage": self.applied_voltage,
            "measuring_voltage": self.applied_voltage / 10.0,
            "vision_impairment_type": str(vision_impairment_type.value),
        }
        h5f, dsets = self._init_h5_output(meta, constants, output_path)
        self._append_initial_state(dsets, pong_game, c, phi)

        measuring_voltage = self.applied_voltage / 10.0

        # Provide globals for calculate_current without refactoring it
        global mesh, nodes, elements, R, T, F, D_diff1, D_mig1, D_diff2, D_mig2, z1, z2, c0, sim
        mesh = self.mesh
        nodes = self.nodes
        elements = self.elements
        R, T, F = self.R, self.T, self.F
        # Backward compatibility: if split coefficients aren't part of the simulation, use D1/D2 for both
        D_diff1 = getattr(self, 'D_diff1', self.D1)
        D_mig1  = getattr(self, 'D_mig1',  self.D1)
        D_diff2 = getattr(self, 'D_diff2', self.D2)
        D_mig2  = getattr(self, 'D_mig2',  self.D2)
        z1, z2 = self.z1, self.z2
        c0 = self.c0
        sim = self.sim

        ball_pos_prev = 10
        ball_pos_rnd = 10
        try:
            
            for _ in tqdm(range(num_steps), desc="3D Simulation Progress"):
                # Game update
                for _ in range(game_ticks):
                    pong_game.step()
                    pong_game.draw(screen)
                    pygame.display.flip()
                    clock.tick(60)

                if pong_game.game_over:
                    pong_game = PongGame(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, False)
                    if rl:
                        # RL perturbations during game over
                        spec = "idle" if rl_diffusion else str(rl_type)
                        # Two-phase support with signed tokens, e.g., '-scramble+idle', '+backrow-backrow'
                        import re as _re
                        parts = _re.findall(r"[+-]?(?:scramble|backrow|idle)", spec.lower())
                        if not parts:
                            parts = ["idle"]
                        if len(parts) == 1:
                            parts = [parts[0], parts[0]]
                        n1 = int(rl_steps // 2)
                        n2 = int(rl_steps - n1)

                        rng = np.random.default_rng()

                        def ramp_fac(t: int, total: int, ramp: int) -> float:
                            if ramp <= 0 or total <= 0:
                                return 1.0
                            up = (t + 1) / float(ramp)
                            down = (total - t) / float(ramp)
                            v = min(1.0, max(0.0, min(up, down)))
                            return float(v)

                        def build_pattern(token: str, base_amp: float) -> list:
                            token = token.strip()
                            sign = 1.0
                            if token.startswith("-"):
                                sign = -1.0
                                token = token[1:]
                            if token.startswith("+"):
                                token = token[1:]
                            pat = [np.nan] * 12
                            if token.lower() == "idle" or token == "":
                                return pat
                            if token.lower() == "scramble":
                                p = int(rng.integers(0, 6))
                                pat[2 * p] = sign * base_amp
                                pat[2 * p + 1] = 0.0
                                return pat
                            if token.lower() == "backrow":
                                for p in (3, 4, 5):
                                    pat[2 * p] = sign * base_amp
                                    pat[2 * p + 1] = 0.0
                                return pat
                            # Unknown token -> idle
                            return pat

                        # Execute RL window step-by-step to support varying patterns
                        total_steps_rl = int(rl_steps)
                        for t_rl in range(total_steps_rl):
                            # Choose phase
                            token = parts[0] if t_rl < n1 else parts[1]
                            amp = self.applied_voltage * ramp_fac(t_rl, total_steps_rl, int(ramp_steps))
                            stim_pat = build_pattern(token, amp)
                            measuring_pattern = [0, measuring_voltage, 0, measuring_voltage, 0, measuring_voltage]
                            voltage_amount = measuring_pattern + stim_pat

                            c_prev = c.copy()
                            c, phi = self.sim.step2(
                                c_prev, phi, self.voltage_indices, voltage_amount, k_reaction=k_reaction
                            )
                            # During RL perturbation, measured current is undefined; log NaNs
                            self._append_step(
                                dsets,
                                c,
                                phi,
                                pong_game.get_ball_position(),
                                pong_game.get_platform_position(),
                                pong_game.score,
                                (np.nan, np.nan, np.nan),
                                voltage_amount,
                            )
                    

                # Sense ball position -> voltage pattern
                ball_pos = pong_game.get_ball_block_index()
                
                # Vision-Impairment
                if vision_impairment_type == VisionImpairmentType.NONE:
                    pass
                elif vision_impairment_type == VisionImpairmentType.DELAYED:
                    if ball_pos == 10: 
                        ball_pos_prev = ball_pos
                    else:
                        ball_tmp = ball_pos 
                        ball_pos = ball_pos_prev
                        ball_pos_prev = ball_tmp    
                elif vision_impairment_type == VisionImpairmentType.RANDOM_FULL:
                    ball_pos = np.random.randint(0, 5)
                elif vision_impairment_type == VisionImpairmentType.RANDOM_FIRSTROW:
                    if ball_pos == 10: 
                        ball_pos_prev = ball_pos
                    elif ball_pos in [0,1,2]: 
                        if ball_pos != ball_pos_prev:
                            ball_pos_prev = ball_pos 
                            ball_pos = np.random.randint(0, 3)
                            ball_pos_rnd = ball_pos
                        elif ball_pos == ball_pos_prev:
                            ball_pos = ball_pos_rnd
                        
                elif vision_impairment_type == VisionImpairmentType.CONTINUOUS:
                    # No discrete block manipulation; continuous pattern is applied below
                    pass
                 
                else: 
                    raise ValueError(f"Unknown vision impairment type: {vision_impairment_type}")
                


                
                voltage_pattern = [np.nan] * 12
                if vision_impairment_type == VisionImpairmentType.CONTINUOUS:
                    voltage_pattern = self.continuous_voltage_pattern(pong_game.get_ball_position())
                else:
                    voltage_pattern[2 * ball_pos] = self.applied_voltage
                    voltage_pattern[2 * ball_pos + 1] = 0
                
                measuring_pattern = [measuring_voltage, 0, measuring_voltage, 0, measuring_voltage, 0]
                voltage_amount = measuring_pattern + voltage_pattern

                # Simulation steps
                if hasattr(self.sim, 'step2_many'):
                    c_hist, phi_hist = self.sim.step2_many(
                        c, phi, self.voltage_indices, voltage_amount,
                        int(sim_ticks), rtol=1e-3, atol=1e-14, max_iter=50, k_reaction=k_reaction
                    )
                    # Use the final state; intermediate sim ticks are not logged in this phase
                    if c_hist.shape[1] > 0:
                        c = c_hist[:, -1]
                        phi = phi_hist[:, -1]
                else:
                    for _ in range(sim_ticks):
                        c_prev = c.copy()
                        c, phi = self.sim.step2(
                            c_prev, phi, self.voltage_indices, voltage_amount, k_reaction=k_reaction
                        )

                # Measure current and update platform
                if electrode_type == "cathode":
                    measured_current = calculate_current(c, phi, [self.voltage_indices[1], self.voltage_indices[3], self.voltage_indices[5]])
                elif electrode_type == "anode":
                    measured_current = calculate_current(c, phi, [self.voltage_indices[0], self.voltage_indices[2], self.voltage_indices[4]])
                plat_pos = calculate_platform_position2(measured_current, self.SCREEN_HEIGHT, activation=activation)
                pong_game.set_platform_position(int(plat_pos))

                # Log step
                self._append_step(dsets, c, phi, pong_game.get_ball_position(), plat_pos, pong_game.score, measured_current, voltage_amount)
        finally:
            h5f.flush()
            h5f.close()
            # Use the RLE logger from PongGame if enabled
            try:
                pong_game.finalize_logging()
            except Exception:
                pass
            pygame.quit()
            # Report saved history path (file was written directly to target)
            try:
                print(f"Saved history to: {self._h5_path}")
            except Exception:
                pass
    

    def continuous_voltage_pattern(self, ball_pos):
        ball_x, ball_y = ball_pos
        ball_x_frac = (ball_x - 100) / (self.SCREEN_WIDTH - 100)
        ball_x_frac = min(max(ball_x_frac, 0), 1)
        ball_y_upper = ball_y / (self.SCREEN_HEIGHT/2)
        ball_y_upper = ball_y_upper if ball_y_upper < 1 else 1 
        ball_y_lower = (ball_y-self.SCREEN_HEIGHT/2)/2
        ball_y_lower = ball_y_lower if ball_y_lower > 0 else 1
        volt_00 = (1-ball_x_frac) * (1-ball_y_upper) * self.applied_voltage
        volt_01 = (1-ball_x_frac) * (ball_y_upper*(1-ball_y_lower)) * self.applied_voltage
        volt_02 = (1-ball_x_frac) * ball_y_lower * self.applied_voltage
        volt_10 = ball_x_frac * (1-ball_y_upper) * self.applied_voltage
        volt_11 = ball_x_frac * (ball_y_upper*(1-ball_y_lower)) * self.applied_voltage
        volt_12 = ball_x_frac * ball_y_lower * self.applied_voltage
        pre_voltage_pattern = [volt_02,volt_01,volt_00,volt_12,volt_11,volt_10]
        voltage_pattern = [np.nan]*12
        for i, volt in enumerate(pre_voltage_pattern):
            voltage_pattern[i*2] = volt if volt != 0 else np.nan
            voltage_pattern[i*2+1] = 0 if volt != 0 else np.nan
        return voltage_pattern

        


def calculate_platform_position(measured_current, screen_height):

    measured_current = np.abs(np.array(measured_current))
    denominator = np.sum(measured_current)
    if denominator == 0:
        raise ValueError("Denominator is zero")
    I1 = measured_current[0]/denominator
    I2 = measured_current[1]/denominator
    I3 = measured_current[2]/denominator
    # Map to 3 equally spaced vertical anchors: 0, H/3, 2H/3
    h_third = float(screen_height) / 3.0
    return np.floor(I1*0 + I2*h_third + I3*(2.0*h_third))

def calculate_platform_position2(measured_current, screen_height, activation="poly_normed"):
    if activation == "poly_normed" or activation == "poly_absolute":
        measured_current = np.abs(np.array(measured_current))
        denominator = np.sum(measured_current)
        if denominator == 0:
            raise ValueError("Denominator is zero")
        if activation == "poly_absolute":
            denominator = 1.0
        I1 = measured_current[0]/denominator
        I2 = measured_current[1]/denominator
        I3 = measured_current[2]/denominator
        # Map to 3 equally spaced vertical anchors: 0, H/3, 2H/3
        h_third = screen_height // 3    
        # Max of 2nd degree polynomial through points
        a,b,c = np.polyfit([0, h_third, 2*h_third], [I1, I2, I3], 2)
        x_space = np.linspace(0, 2*h_third, 2*h_third)
        return np.argmax(a*x_space**2 + b*x_space)
    elif activation == "poly_raw":
        denominator = np.sum(measured_current)
        if denominator == 0:
            raise ValueError("Denominator is zero")
        I1 = measured_current[0]/denominator
        I2 = measured_current[1]/denominator
        I3 = measured_current[2]/denominator
        # Map to 3 equally spaced vertical anchors: 0, H/3, 2H/3
        h_third = screen_height // 3    
        # Max of 2nd degree polynomial through points
        a,b,c = np.polyfit([0, h_third, 2*h_third], [I1, I2, I3], 2)
        x_space = np.linspace(0, 2*h_third, 2*h_third)
        return np.argmax(a*x_space**2 + b*x_space)
    elif activation == "average":
        return calculate_platform_position(measured_current, screen_height)
    else:
        raise ValueError("Invalid activation: " + activation)


if __name__ == "__main__":
    sim_runner = PongSimulationNPEN(dt=0.1)
    sim_runner.run(sim_ticks=5, game_ticks=10, num_steps=20, k_reaction=0.01, rl=False, rl_steps=4)
