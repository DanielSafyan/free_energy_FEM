import numpy as np
import os
from tqdm import tqdm
import pygame
import h5py
import sys



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
        "c3": _create_ext_dataset(h5f, "states/c3", (N,), np.float64),
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

def append_initial_state(dsets, pong_game, c, c3, phi):
    _append_row(dsets["c"], c)
    _append_row(dsets["c3"], c3)
    _append_row(dsets["phi"], phi)
    _append_row(dsets["ball"], np.array(pong_game.get_ball_position(), dtype=np.float64))
    _append_row(dsets["platform"], float(pong_game.get_platform_position()))
    _append_row(dsets["score"], int(pong_game.score))
    # placeholders to align lengths for step 0
    _append_row(dsets["current"], np.array([np.nan, np.nan, np.nan], dtype=np.float64))
    _append_row(dsets["voltage"], np.full(18, np.nan, dtype=np.float64))

def append_step(dsets, c, c3, phi, ball_pos_xy, platform_pos, score, measured_current, voltage_amount):
    _append_row(dsets["c"], c)
    _append_row(dsets["c3"], c3)
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
        self.c3 = self._f["states/c3"]
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
                "c3": f["states/c3"][...],
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

def init_voltage():
    # Deprecated in favor of PongSimulation._init_voltage()
    raise NotImplementedError("Use PongSimulation._init_voltage() which uses instance nx, ny, nz.")


def calculate_current(c, c3, phi, measuring_indices):
    """
    Compute electric current (A) at each measuring electrode for NPEN (single salt c).

    Returns a 3-tuple of currents corresponding to the three indices in measuring_indices.

    Runtime optimizations:
    - Precompute, once, for each electrode node: connected elements and incident faces' geometry.
    - Precompute constant factors from physical parameters.
    """
    # Access globals prepared in __main__
    global mesh, nodes, elements, R, T, F, D1, D2, z1, z2, c0, sim

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
        # Factors for flux terms
        K_GRAD_C = (-D1 * c0, -D2 * c0)  # for positive/negative ions, both using c
        K_MIG = (-(z1 * F * D1 / (R * T)) * phi_c, -(z2 * F * D2 / (R * T)) * phi_c)
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
                 Lx=1.0, Ly=1.0, Lz=0.25,
                 screen_width=400, screen_height=600,
                 R=8.314, T=298.0, F=96485.33,
                 epsilon=80 * 8.854e-12,
                 D1=1e-9, D2=1e-9, D3=1e-9,
                 z1=1, z2=-1,
                 chi=0.0,
                 applied_voltage=1e-1,
                 c0=10.0,
                 L_c=1e-3,
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
        sensing_electrode11_idx = gi(nx//4, ny//4, 0)
        sensing_electrode12_idx = gi(nx//4, ny//4, nz)
        sensing_electrode21_idx = gi(nx//4, 2*ny//4, 0)
        sensing_electrode22_idx = gi(nx//4, 2*ny//4, nz)
        sensing_electrode31_idx = gi(nx//4, 3*ny//4, 0)
        sensing_electrode32_idx = gi(nx//4, 3*ny//4, nz)

        # 3 stimulating electrode pairs in the middle row at y = 2*ny//4
        stimulating_electrode11_idx = gi(2*nx//4, ny//4, 0)
        stimulating_electrode12_idx = gi(2*nx//4, ny//4, nz)
        stimulating_electrode21_idx = gi(2*nx//4, 2*ny//4, 0)
        stimulating_electrode22_idx = gi(2*nx//4, 2*ny//4, nz)
        stimulating_electrode31_idx = gi(2*nx//4, 3*ny//4, 0)
        stimulating_electrode32_idx = gi(2*nx//4, 3*ny//4, nz)

        # 3 stimulating electrode pairs in the upper row at y = 3*ny//4
        stimulating_electrode41_idx = gi(3*nx//4, ny//4, 0)
        stimulating_electrode42_idx = gi(3*nx//4, ny//4, nz)
        stimulating_electrode51_idx = gi(3*nx//4, 2*ny//4, 0)
        stimulating_electrode52_idx = gi(3*nx//4, 2*ny//4, nz)
        stimulating_electrode61_idx = gi(3*nx//4, 3*ny//4, 0)
        stimulating_electrode62_idx = gi(3*nx//4, 3*ny//4, nz)

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
            "c3": self._create_ext_dataset(h5f, "states/c3", (N,), np.float64),
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

    def _append_initial_state(self, dsets, pong_game, c, c3, phi):
        self._append_row(dsets["c"], c)
        self._append_row(dsets["c3"], c3)
        self._append_row(dsets["phi"], phi)
        self._append_row(dsets["ball"], np.array(pong_game.get_ball_position(), dtype=np.float64))
        self._append_row(dsets["platform"], float(pong_game.get_platform_position()))
        self._append_row(dsets["score"], int(pong_game.score))
        # placeholders to align lengths for step 0
        self._append_row(dsets["current"], np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        self._append_row(dsets["voltage"], np.full(18, np.nan, dtype=np.float64))

    def _append_step(self, dsets, c, c3, phi, ball_pos_xy, platform_pos, score, measured_current, voltage_amount):
        self._append_row(dsets["c"], c)
        self._append_row(dsets["c3"], c3)
        self._append_row(dsets["phi"], phi)
        self._append_row(dsets["ball"], np.array(ball_pos_xy, dtype=np.float64))
        self._append_row(dsets["platform"], float(platform_pos))
        self._append_row(dsets["score"], int(score))
        self._append_row(dsets["current"], np.array(measured_current, dtype=np.float64))
        self._append_row(dsets["voltage"], np.array(voltage_amount, dtype=np.float64))

    def _init_conditions(self):
        if self.experiment == "gaussian":
            c3 = np.full(self.mesh.num_nodes(), 0.9)
            center_x, center_y, center_z = self.Lx / 2, self.Ly / 2, self.Lz / 2
            sigma = self.Lx / 10
            c = 0.05 + 0.04 * np.exp(-((self.nodes[:, 0] - center_x) ** 2 +
                                       (self.nodes[:, 1] - center_y) ** 2 +
                                       (self.nodes[:, 2] - center_z) ** 2) / (2 * sigma ** 2))
        elif self.experiment == "random":
            c3 = np.full(self.mesh.num_nodes(), 0.5)
            c = 0.25 + np.random.uniform(-0.1, 0.1, self.mesh.num_nodes())
        else:
            c3 = np.full(self.mesh.num_nodes(), 0.0)
            c = np.full(self.mesh.num_nodes(), 0.5)
            c[self.nodes[:, 0] < self.Lx / 2] = 0.4
        phi = np.zeros(self.mesh.num_nodes())
        return c, c3, phi

    def run(self,rl=False, sim_ticks=1, game_ticks=6, num_steps=50, k_reaction=0.5, output_path=None, rl_steps=8):
        # Initialize pygame/game
        pygame.init()
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Single Player Pong")
        clock = pygame.time.Clock()
        pong_game = PongGame(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, False)

        # Initial states (NPEN: c, c3, phi)
        c, c3, phi = self._init_conditions()

        # HDF5
        meta = {
            "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dt": self.dt, "num_steps": num_steps,
            "experiment": self.experiment,
        }
        constants = {
            "R": self.R, "T": self.T, "F": self.F, "epsilon": self.epsilon,
            "D1": self.D1, "D2": self.D2, "D3": self.D3,
            "z1": self.z1, "z2": self.z2, "chi": self.chi, "c0": self.c0,
            "k_reaction": k_reaction,
            "applied_voltage": self.applied_voltage,
            "measuring_voltage": self.applied_voltage / 10.0,
        }
        h5f, dsets = self._init_h5_output(meta, constants, output_path)
        self._append_initial_state(dsets, pong_game, c, c3, phi)

        measuring_voltage = self.applied_voltage / 10.0

        # Provide globals for calculate_current without refactoring it
        global mesh, nodes, elements, R, T, F, D1, D2, z1, z2, c0, sim
        mesh = self.mesh
        nodes = self.nodes
        elements = self.elements
        R, T, F = self.R, self.T, self.F
        D1, D2 = self.D1, self.D2
        z1, z2 = self.z1, self.z2
        c0 = self.c0
        sim = self.sim

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
                        # give chaotic signals to increase plasticity for a short time
                        for _ in range(rl_steps):
                            # invert voltage pattern
                            voltage_pattern = [0] * 12
                            for i in range(6): voltage_pattern[2*i+1] = self.applied_voltage
                            measuring_pattern = [0, measuring_voltage, 0, measuring_voltage, 0, measuring_voltage]
                            voltage_amount = measuring_pattern + voltage_pattern

                            c_prev, c3_prev = c.copy(), c3.copy()
                            c, c3, phi = self.sim.step2(
                                c_prev, c3_prev, phi, self.voltage_indices, voltage_amount, k_reaction=k_reaction
                            )
                            self._append_step(dsets, c, c3, phi, pong_game.get_ball_position(), plat_pos, pong_game.score, measured_current, voltage_amount)
                    

                # Sense ball position -> voltage pattern
                ball_pos = pong_game.get_ball_block_index()


                voltage_pattern = [np.nan] * 12
                voltage_pattern[2 * ball_pos] = self.applied_voltage
                voltage_pattern[2 * ball_pos + 1] = 0
                measuring_pattern = [measuring_voltage, 0, measuring_voltage, 0, measuring_voltage, 0]
                voltage_amount = measuring_pattern + voltage_pattern

                # Simulation steps
                for _ in range(sim_ticks):
                    c_prev, c3_prev = c.copy(), c3.copy()
                    c, c3, phi = self.sim.step2(
                        c_prev, c3_prev, phi, self.voltage_indices, voltage_amount, k_reaction=k_reaction
                    )

                # Measure current and update platform
                measured_current = calculate_current(c, c3, phi, [self.voltage_indices[0], self.voltage_indices[2], self.voltage_indices[4]])
                plat_pos = calculate_platform_position(measured_current)
                pong_game.set_platform_position(int(plat_pos))

                # Log step
                self._append_step(dsets, c, c3, phi, pong_game.get_ball_position(), plat_pos, pong_game.score, measured_current, voltage_amount)
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


def calculate_platform_position(measured_current):

    measured_current = np.abs(np.array(measured_current))
    denominator = np.sum(measured_current)
    if denominator == 0:
        raise ValueError("Denominator is zero")
    I1 = measured_current[0]/denominator
    I2 = measured_current[1]/denominator
    I3 = measured_current[2]/denominator
    return np.floor(I1*0 + I2*200 + I3*400)

if __name__ == "__main__":
    sim_runner = PongSimulationNPEN(dt=0.1)
    sim_runner.run(sim_ticks=5, game_ticks=10, num_steps=20, k_reaction=0.01, rl=False, rl_steps=4)
