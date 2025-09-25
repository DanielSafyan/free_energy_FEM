"""
Colab NPEN accelerator: GPU-enabled linear solve (CuPy GMRES with CPU fallback)
plus a convenience runner class that prefers the GPU core.
"""
from __future__ import annotations

import os
import time
from typing import Sequence, Optional, Tuple, Dict, Any
import numpy as np

# Optional GPU stack (CuPy)
_GPU_AVAILABLE = False
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.sparse import csr_matrix as cpx_csr_matrix  # type: ignore
    from cupyx.scipy.sparse import coo_matrix as cpx_coo_matrix  # type: ignore
    from cupyx.scipy.sparse import vstack as cpx_vstack  # type: ignore
    from cupyx.scipy.sparse import hstack as cpx_hstack  # type: ignore
    from cupyx.scipy.sparse.linalg import gmres as cpx_gmres  # type: ignore
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from pong_simulation.pong_sim_npen import PongSimulationNPEN, VisionImpairmentType
from simulations.NPENwithFOReaction import NPENwithFOReaction


def _gpu_linear_solve(jacobian, rhs, rtol: float, max_iter: int, profile: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve Ax=b via CuPy GMRES if available; otherwise SciPy spsolve.

    Returns (solution, info) where info contains timing and backend details:
      - used_gpu: bool
      - fallback: bool
      - t_total, and if GPU used: t_csr, t_to_gpu, t_gmres, t_to_cpu
    """
    info: Dict[str, Any] = {
        "used_gpu": False,
        "fallback": False,
        "t_total": 0.0,
    }
    t0 = time.perf_counter()

    # CPU path (no CuPy)
    if not _GPU_AVAILABLE:
        x = spsolve(jacobian, rhs)
        info["t_total"] = time.perf_counter() - t0
        return x, info

    # Attempt GPU path
    try:
        t_csr0 = time.perf_counter()
        A = jacobian.tocsr()
        t_csr1 = time.perf_counter()
        # Ensure 1-D device arrays with correct dtypes for CuPy CSR
        data_gpu = cp.asarray(A.data, dtype=cp.float64).ravel()
        indices_gpu = cp.asarray(A.indices, dtype=cp.int32).ravel()
        indptr_gpu = cp.asarray(A.indptr, dtype=cp.int32).ravel()
        A_gpu = cpx_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=A.shape)
        b_gpu = cp.asarray(np.asarray(rhs, dtype=np.float64).ravel(), dtype=cp.float64)
        t_xfer1 = time.perf_counter()
        x_gpu, gmres_info = cpx_gmres(A_gpu, b_gpu, tol=max(rtol, 1e-8), maxiter=max_iter)
        t_solve1 = time.perf_counter()
        x = cp.asnumpy(x_gpu)
        t_back1 = time.perf_counter()
        info.update({
            "used_gpu": True,
            "fallback": False if gmres_info == 0 else True,
            "t_csr": t_csr1 - t_csr0,
            "t_to_gpu": t_xfer1 - t_csr1,
            "t_gmres": t_solve1 - t_xfer1,
            "t_to_cpu": t_back1 - t_solve1,
        })
        if gmres_info != 0:
            # Fallback to CPU direct solve if GMRES did not converge
            x = spsolve(jacobian, rhs)
        info["t_total"] = time.perf_counter() - t0
        if profile:
            backend = "GPU" if info["used_gpu"] and not info["fallback"] else "GPU->CPU"
            print(f"[linear_solve] backend={backend} total={info['t_total']:.4f}s csr={info.get('t_csr',0):.4f}s to_gpu={info.get('t_to_gpu',0):.4f}s gmres={info.get('t_gmres',0):.4f}s to_cpu={info.get('t_to_cpu',0):.4f}s")
        return x, info
    except Exception as e:
        # Robust CPU fallback
        x = spsolve(jacobian, rhs)
        info["t_total"] = time.perf_counter() - t0
        info["used_gpu"] = False
        info["fallback"] = True
        if profile:
            print(f"[linear_solve] exception -> CPU fallback, total={info['t_total']:.4f}s, err={e}")
        return x, info

def _to_gpu_csr(A):
    """Convert a SciPy CSR/CSC matrix A to a CuPy CSR matrix with correct dtypes."""
    A = A.tocsr()
    data_gpu = cp.asarray(A.data, dtype=cp.float64).ravel()
    indices_gpu = cp.asarray(A.indices, dtype=cp.int32).ravel()
    indptr_gpu = cp.asarray(A.indptr, dtype=cp.int32).ravel()
    return cpx_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=A.shape)

def _gpu_linear_solve_gpu(A_gpu, b_gpu, rtol: float, max_iter: int, profile: bool = False):
    """Solve on GPU given GPU CSR and GPU RHS. Returns (x_gpu, info)."""
    info: Dict[str, Any] = {"used_gpu": True, "fallback": False, "t_total": 0.0}
    t0 = time.perf_counter()
    x_gpu, gmres_info = cpx_gmres(A_gpu, b_gpu, tol=max(rtol, 1e-8), maxiter=max_iter)
    info["t_total"] = time.perf_counter() - t0
    if gmres_info != 0:
        info["fallback"] = True
    if profile:
        print(f"[linear_solve] backend=GPU total={info['t_total']:.4f}s")
    return x_gpu, info


class GPUNPENwithFOReaction(NPENwithFOReaction):
    """NPEN + FO reaction with GPU-accelerated linear solve when possible.

    Set `benchmark=True` to print per-iteration timing breakdowns.
    """

    def __init__(self, *args, benchmark: bool = False, gpu_assembly: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = bool(benchmark)
        self.gpu_assembly = bool(gpu_assembly) and _GPU_AVAILABLE
        # Precompute element-level data and global index maps for fast GPU assembly
        self._gpu_ready = False
        if self.gpu_assembly:
            try:
                self._prepare_gpu_assembly()
                self._gpu_ready = True
            except Exception as e:
                print(f"[GPUNPEN] GPU assembly prep failed, falling back to CPU assembly: {e}")
                self.gpu_assembly = False

    def _prepare_gpu_assembly(self):
        """Precompute arrays used for vectorized GPU assembly."""
        # Element nodes (E, 4)
        elem_nodes_np = np.asarray(self.mesh.elements, dtype=np.int32)
        self._E = elem_nodes_np.shape[0]
        self._N = self.mesh.num_nodes()
        # Precompute row/col index patterns for 4x4 local matrices across all elements
        # rows[e,i,j] = global node index of local i
        # cols[e,i,j] = global node index of local j
        rows = np.repeat(elem_nodes_np[:, :, None], 4, axis=2)  # (E,4,4)
        cols = np.repeat(elem_nodes_np[:, None, :], 4, axis=1)  # (E,4,4)
        self._rows_flat = rows.reshape(-1).astype(np.int32)
        self._cols_flat = cols.reshape(-1).astype(np.int32)
        # Gradients and volumes per element
        grads_list = []
        vols_list = []
        for e in range(self._E):
            if e in self.mesh._element_data:
                gd = self.mesh._element_data[e]['grads']  # (4,3)
                vol = self.mesh._element_data[e]['volume']
            else:
                gd = np.zeros((4,3), dtype=np.float64)
                vol = 0.0
            grads_list.append(gd)
            vols_list.append(vol)
        grads_np = np.asarray(grads_list, dtype=np.float64)   # (E,4,3)
        vols_np = np.asarray(vols_list, dtype=np.float64)     # (E,)
        # Move to GPU
        self._elem_nodes_gpu = cp.asarray(elem_nodes_np)
        self._grads_gpu = cp.asarray(grads_np)
        self._vols_gpu = cp.asarray(vols_np)
        # Cache row/col indices on GPU for GPU COO builds
        self._rows_gpu = cp.asarray(self._rows_flat)
        self._cols_gpu = cp.asarray(self._cols_flat)

        # Precompute fixed CSR sparsity structure (indices/indptr) once on CPU
        # and a mapping from each element-level COO contribution to its CSR slot.
        L = self._rows_flat.shape[0]
        pattern_cpu = coo_matrix(
            (np.ones(L, dtype=np.float64), (self._rows_flat, self._cols_flat)),
            shape=(self._N, self._N),
        ).tocsr()
        # Map (row, col) -> CSR position
        pos_dict: Dict[Tuple[int, int], int] = {}
        ind = pattern_cpu.indices
        indptr = pattern_cpu.indptr
        for r in range(self._N):
            start = indptr[r]
            end = indptr[r + 1]
            for p in range(start, end):
                c = int(ind[p])
                pos_dict[(r, c)] = p
        # Build mapping array for all element 4x4 entries (length L)
        coo2csr_pos = np.empty(L, dtype=np.int32)
        rf = self._rows_flat
        cf = self._cols_flat
        for k in range(L):
            coo2csr_pos[k] = pos_dict[(int(rf[k]), int(cf[k]))]
        # Store structure and mapping on GPU
        self._struct_indices_gpu = cp.asarray(ind.astype(np.int32))
        self._struct_indptr_gpu = cp.asarray(indptr.astype(np.int32))
        self._struct_nnz = int(pattern_cpu.nnz)
        self._coo2csr_pos_gpu = cp.asarray(coo2csr_pos)

        # Precompute M and K data in the same base-structure order and cache CSR for SpMV
        def _to_struct_data(csr_mat):
            csr = csr_mat.tocsr()
            out = np.zeros(self._struct_nnz, dtype=np.float64)
            for r in range(self._N):
                s = csr.indptr[r]
                e = csr.indptr[r + 1]
                cols = csr.indices[s:e]
                vals = csr.data[s:e]
                for j, col in enumerate(cols):
                    pos = pos_dict[(r, int(col))]
                    out[pos] = vals[j]
            return out
        self._M_struct_data_gpu = cp.asarray(_to_struct_data(self.M_mat))
        self._K_struct_data_gpu = cp.asarray(_to_struct_data(self.K_mat))
        self._M_struct_csr_gpu = cpx_csr_matrix((self._M_struct_data_gpu, self._struct_indices_gpu, self._struct_indptr_gpu), shape=(self._N, self._N))
        self._K_struct_csr_gpu = cpx_csr_matrix((self._K_struct_data_gpu, self._struct_indices_gpu, self._struct_indptr_gpu), shape=(self._N, self._N))

        # Build full 2N x 2N Jacobian sparsity once and block position maps
        N = self._N
        rf = self._rows_flat
        cf = self._cols_flat
        rows_full = np.concatenate([rf, rf, rf + N, rf + N]).astype(np.int32)
        cols_full = np.concatenate([cf, cf + N, cf, cf + N]).astype(np.int32)
        ones_full = np.ones(rows_full.shape[0], dtype=np.float64)
        Jpat_cpu = coo_matrix((ones_full, (rows_full, cols_full)), shape=(2 * N, 2 * N)).tocsr()
        J_ind = Jpat_cpu.indices
        J_indptr = Jpat_cpu.indptr
        self._J_nnz = int(Jpat_cpu.nnz)
        # Map (row,col) in full to CSR pos
        pos_full: Dict[Tuple[int, int], int] = {}
        for r in range(2 * N):
            s = J_indptr[r]
            e = J_indptr[r + 1]
            for p in range(s, e):
                c = int(J_ind[p])
                pos_full[(r, c)] = p
        # Build base-structure (N x N) row/col arrays of length struct_nnz
        base_rows = np.empty(self._struct_nnz, dtype=np.int32)
        base_cols = ind.astype(np.int32).copy()
        for r in range(N):
            s = indptr[r]
            e = indptr[r + 1]
            base_rows[s:e] = r
        # Block position arrays for struct_nnz entries
        pos11_struct = np.empty(self._struct_nnz, dtype=np.int32)
        pos13_struct = np.empty(self._struct_nnz, dtype=np.int32)
        pos31_struct = np.empty(self._struct_nnz, dtype=np.int32)
        pos33_struct = np.empty(self._struct_nnz, dtype=np.int32)
        for p in range(self._struct_nnz):
            r0 = int(base_rows[p])
            c0 = int(base_cols[p])
            pos11_struct[p] = pos_full[(r0, c0)]
            pos13_struct[p] = pos_full[(r0, c0 + N)]
            pos31_struct[p] = pos_full[(r0 + N, c0)]
            pos33_struct[p] = pos_full[(r0 + N, c0 + N)]
        # Cache full structure and mappings on GPU
        self._J_indices_gpu = cp.asarray(J_ind.astype(np.int32))
        self._J_indptr_gpu = cp.asarray(J_indptr.astype(np.int32))
        self._pos11_struct_gpu = cp.asarray(pos11_struct)
        self._pos13_struct_gpu = cp.asarray(pos13_struct)
        self._pos31_struct_gpu = cp.asarray(pos31_struct)
        self._pos33_struct_gpu = cp.asarray(pos33_struct)

    def _apply_bcs_bulk(self, jacobian, residual, phi, c, electrode_indices, applied_voltages, k_reaction):
        """
        Apply Dirichlet BCs for phi and first-order reaction on c in a single sparse pass.
        This avoids repeated conversions to LIL per electrode.
        """
        J = jacobian.tolil()
        phi_off = 1 * self.num_nodes
        kneg = -float(k_reaction)
        for n, elec_idx in enumerate(electrode_indices):
            v = applied_voltages[n]
            if np.isnan(v):
                continue
            # Dirichlet on phi row
            dof_phi = phi_off + int(elec_idx)
            # zero the row efficiently
            J.rows[dof_phi] = [dof_phi]
            J.data[dof_phi] = [1.0]
            residual[dof_phi] = float(phi[int(elec_idx)] - v / self.phi_c)
            # First-order reaction on c (diagonal tweak and residual add)
            dof_c = int(elec_idx)
            residual[dof_c] += kneg * float(c[dof_c])
            J[dof_c, dof_c] += kneg
        return J.tocsc(), residual

    def _assemble_coupling_matrix_gpu(self, coeff_node_np: np.ndarray, scale: float):
        """
        Assemble K = ∫ (scale*coeff * grad(phi_i)·grad(phi_j)) dΩ using GPU vectorization.
        coeff_node_np: node-wise coefficient array (N,)
        Returns CuPy CSR matrix of shape (N,N)
        """
        # Average coefficient per element: (E,)
        coeff_elem = cp.mean(cp.asarray(coeff_node_np, dtype=cp.float64)[self._elem_nodes_gpu], axis=1)
        # Local grad dot products per element: (E,4,4)
        # w_ij = volume * (grad_i · grad_j)
        w = cp.einsum('eik,ejk->eij', self._grads_gpu, self._grads_gpu)
        w = (w * self._vols_gpu[:, None, None]) * (coeff_elem[:, None, None] * float(scale))
        data = w.reshape(-1)
        # Accumulate into precomputed CSR structure (reuse indices/indptr)
        out = cp.zeros(self._struct_nnz, dtype=cp.float64)
        cp.add.at(out, self._coo2csr_pos_gpu, data)
        return out

    def _assemble_convection_matrix_gpu(self, phi_node_np: np.ndarray, prefactor: float):
        """
        Assemble C for term ∫ (prefactor * (∇v_i · ∇phi)) v_j dΩ on GPU.
        For linear tetrahedra, integral(phi_j) = Volume/4, so row i is constant across columns j.
        Returns CuPy CSR matrix.
        """
        phi_on_e = cp.asarray(phi_node_np, dtype=cp.float64)[self._elem_nodes_gpu]  # (E,4)
        # grad_phi_cell = sum_i phi_i * grad_i -> (E,3)
        grad_phi = cp.einsum('eik,ei->ek', self._grads_gpu, phi_on_e)
        # dot_i = grad_i · grad_phi -> (E,4)
        dot_i = cp.einsum('eik,ek->ei', self._grads_gpu, grad_phi)
        val_row = dot_i * (self._vols_gpu[:, None] * 0.25 * float(prefactor))  # (E,4)
        # Expand across columns j (repeat 4 times)
        val = cp.repeat(val_row[:, :, None], 4, axis=2)  # (E,4,4)
        data = val.reshape(-1)
        out = cp.zeros(self._struct_nnz, dtype=cp.float64)
        cp.add.at(out, self._coo2csr_pos_gpu, data)
        return out

    def _apply_bcs_gpu(self, J_gpu, residual_gpu, phi_gpu, c_gpu, electrode_indices, applied_voltages, k_reaction):
        """Apply Dirichlet on phi rows and first-order reaction on c diagonals directly on GPU CSR."""
        indptr = J_gpu.indptr
        indices = J_gpu.indices
        data = J_gpu.data
        N = self.num_nodes
        for n, elec_idx in enumerate(electrode_indices):
            v = applied_voltages[n]
            if np.isnan(v):
                continue
            # Dirichlet for phi row
            dof_phi = N + int(elec_idx)
            rs = indptr[dof_phi]
            re = indptr[dof_phi + 1]
            data[rs:re] = 0.0
            row_idx = indices[rs:re]
            mask = (row_idx == dof_phi)
            if cp.any(mask):
                data[rs:re][mask] = 1.0
            residual_gpu[dof_phi] = phi_gpu[int(elec_idx)] - (float(v) / float(self.phi_c))
            # Reaction on c diagonal and residual
            dof_c = int(elec_idx)
            rs_c = indptr[dof_c]
            re_c = indptr[dof_c + 1]
            row_idx_c = indices[rs_c:re_c]
            mask_c = (row_idx_c == dof_c)
            if cp.any(mask_c):
                data[rs_c:re_c][mask_c] += -float(k_reaction)
            residual_gpu[dof_c] += -float(k_reaction) * c_gpu[dof_c]
        return J_gpu, residual_gpu

    def step2(
        self,
        c_initial: np.ndarray,
        phi_initial: np.ndarray,
        electrode_indices: Sequence[int],
        applied_voltages: Sequence[float],
        k_reaction: float = 0.5,
        rtol: float = 1e-3,
        atol: float = 1e-14,
        max_iter: int = 50,
    ):
        c = np.asarray(c_initial, dtype=np.float64).copy()
        phi = np.asarray(phi_initial, dtype=np.float64).copy()
        if len(electrode_indices) != len(applied_voltages):
            raise ValueError("The number of electrode indices must match the number of applied voltages.")

        # Preallocate buffers for GPU path
        full_data_buf = None
        if self.gpu_assembly and self._gpu_ready:
            full_data_buf = cp.zeros(self._J_nnz, dtype=cp.float64)

        initial_residual_norm = -1.0
        for i in range(max_iter):
            t_asm0 = time.perf_counter()
            if self.gpu_assembly and self._gpu_ready:
                # Assemble on GPU entirely (reuse structure; only update data)
                c_gpu = cp.asarray(c, dtype=cp.float64)
                cprev_gpu = cp.asarray(c_initial, dtype=cp.float64)
                phi_gpu = cp.asarray(phi, dtype=cp.float64)
                # Variable block data
                J_cc_drift_data = self._assemble_convection_matrix_gpu(phi, self.D1_dim * self.z1)
                K_c_phi_data = self._assemble_coupling_matrix_gpu(c, self.D1_dim * self.z1)
                K_phi_phi_data = self._assemble_coupling_matrix_gpu(c, (self.D1_dim + self.D2_dim))
                # Compose full Jacobian data vector (reuse preallocated buffer)
                full_data = full_data_buf
                full_data.fill(0.0)
                # J11 = (1/dt)*M + D1*K + J_cc_drift
                if float(self.dt_dim) != 0.0:
                    cp.add.at(full_data, self._pos11_struct_gpu, (1.0 / float(self.dt_dim)) * self._M_struct_data_gpu)
                if float(self.D1_dim) != 0.0:
                    cp.add.at(full_data, self._pos11_struct_gpu, float(self.D1_dim) * self._K_struct_data_gpu)
                cp.add.at(full_data, self._pos11_struct_gpu, J_cc_drift_data)
                # J13 = K_c_phi
                cp.add.at(full_data, self._pos13_struct_gpu, K_c_phi_data)
                # J31 = -(D1 - D2) * K
                cp.add.at(full_data, self._pos31_struct_gpu, (-(float(self.D1_dim) - float(self.D2_dim))) * self._K_struct_data_gpu)
                # J33 = K_phi_phi
                cp.add.at(full_data, self._pos33_struct_gpu, K_phi_phi_data)
                Jacobian_gpu = cpx_csr_matrix((full_data, self._J_indices_gpu, self._J_indptr_gpu), shape=(2 * self._N, 2 * self._N))
                # Residual on GPU: reuse struct CSR for M/K; create temp CSR for variable blocks
                R_c_gpu = (self._M_struct_csr_gpu @ (c_gpu - cprev_gpu)) / float(self.dt_dim) + float(self.D1_dim) * (self._K_struct_csr_gpu @ c_gpu)
                K_c_phi_csr = cpx_csr_matrix((K_c_phi_data, self._struct_indices_gpu, self._struct_indptr_gpu), shape=(self._N, self._N))
                R_c_gpu = R_c_gpu + (K_c_phi_csr @ phi_gpu)
                K_phi_phi_csr = cpx_csr_matrix((K_phi_phi_data, self._struct_indices_gpu, self._struct_indptr_gpu), shape=(self._N, self._N))
                R_phi_gpu = (K_phi_phi_csr @ phi_gpu) + (-(float(self.D1_dim) - float(self.D2_dim))) * (self._K_struct_csr_gpu @ c_gpu)
                residual_gpu = cp.concatenate([R_c_gpu, R_phi_gpu])
            else:
                residual, Jacobian = self._assemble_residual_and_jacobian(c, phi, c_initial)
            t_asm1 = time.perf_counter()

            # Apply electrode BCs and first-order reaction
            t_bc0 = time.perf_counter()
            if self.gpu_assembly and self._gpu_ready:
                Jacobian_gpu, residual_gpu = self._apply_bcs_gpu(Jacobian_gpu, residual_gpu, phi_gpu, c_gpu,
                                                                 electrode_indices, applied_voltages, k_reaction)
            else:
                Jacobian, residual = self._apply_bcs_bulk(Jacobian, residual, phi, c,
                                                          electrode_indices, applied_voltages, k_reaction)
            t_bc1 = time.perf_counter()

            if self.gpu_assembly and self._gpu_ready:
                nrm = float(cp.asnumpy(cp.linalg.norm(residual_gpu)))
            else:
                nrm = float(np.linalg.norm(residual))
            if i == 0:
                initial_residual_norm = nrm if nrm > 0 else 1.0
            if nrm < (initial_residual_norm * rtol) + atol:
                break

            t_solve0 = time.perf_counter()
            if self.gpu_assembly and self._gpu_ready:
                rhs_gpu = -residual_gpu
                delta_gpu, linf = _gpu_linear_solve_gpu(Jacobian_gpu, rhs_gpu, rtol=rtol, max_iter=max_iter, profile=self.benchmark)
                delta = cp.asnumpy(delta_gpu)
            else:
                delta, linf = _gpu_linear_solve(Jacobian, -residual, rtol=rtol, max_iter=max_iter, profile=self.benchmark)
            t_solve1 = time.perf_counter()
            t_upd0 = time.perf_counter()
            c   += self.alpha     * delta[0 * self.num_nodes : 1 * self.num_nodes]
            phi += self.alpha_phi * delta[1 * self.num_nodes : 2 * self.num_nodes]
            t_upd1 = time.perf_counter()

            if self.benchmark:
                J_nnz = Jacobian.nnz if not (self.gpu_assembly and self._gpu_ready) else int(Jacobian_gpu.nnz)
                print(
                    f"[GPUNPEN it={i:02d}] asm={t_asm1-t_asm0:.4f}s bc={t_bc1-t_bc0:.4f}s "
                    f"solve={t_solve1-t_solve0:.4f}s (backend={'GPU' if linf.get('used_gpu') else 'CPU'}{' Fallback' if linf.get('fallback') else ''}) "
                    f"update={t_upd1-t_upd0:.4f}s resid={nrm:.3e} nnz={J_nnz}"
                )

        return c, phi


class PongSimulationNPENColab(PongSimulationNPEN):
    """
    Headless Colab-friendly runner. If use_gpu=True, swaps core to GPUNPENwithFOReaction.
    """

    def __init__(self, *args, use_gpu: bool = True, benchmark: bool = True, gpu_assembly: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_gpu = bool(use_gpu)
        if self._use_gpu:
            # Replace core with GPU variant
            self.sim = GPUNPENwithFOReaction(
                self.mesh, self.dt, self.D1, self.D2, self.D3, self.z1, self.z2,
                self.epsilon, self.R, self.T, self.L_c, self.c0,
                voltage=self.applied_voltage,
                alpha=0.5, alpha_phi=0.5,
                chemical_potential_terms=[],
                boundary_nodes=self.boundary_nodes,
                benchmark=benchmark,
                gpu_assembly=gpu_assembly,
            )

    def run(self, *args, **kwargs):
        # Headless display for Colab
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        return super().run(*args, **kwargs)
