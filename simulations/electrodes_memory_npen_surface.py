import os, numpy as np, h5py
from tqdm import tqdm
from typing import List, Optional, Sequence, Tuple
from utils.fem_mesh import create_structured_mesh_3d
from pong_simulation.hybrid_npen_simulation import HybridNPENwithFOReaction

def _create_ext_dataset(h5f, name, shape_tail, dtype=np.float64):
    maxshape=(None,)+tuple(shape_tail); chunks=(1,)+tuple(shape_tail)
    return h5f.create_dataset(name, shape=(0,)+tuple(shape_tail), maxshape=maxshape, chunks=chunks,
                              dtype=dtype, compression="gzip", compression_opts=4)

def init_h5_output(nodes, elements, meta, consts, out_path=None):
    os.makedirs("output", exist_ok=True)
    h5_path = out_path or os.path.join("output","pong_simulation_surface.h5")
    if os.path.exists(h5_path): os.remove(h5_path)
    f=h5py.File(h5_path,"w"); f.attrs.update({
        "Lx":meta["Lx"],"Ly":meta["Ly"],"Lz":meta["Lz"],"nx":meta["nx"],"ny":meta["ny"],"nz":meta["nz"],
        "num_nodes":nodes.shape[0],"num_cells":elements.shape[0],"dt":meta["dt"],"num_steps":meta["num_steps"],
        "experiment":meta.get("experiment","surface_memory")}); g=f.create_group("constants"); g.attrs.update(consts)
    f.create_dataset("mesh/nodes",data=nodes,compression="gzip",compression_opts=4)
    f.create_dataset("mesh/elements",data=elements,compression="gzip",compression_opts=4)
    N=nodes.shape[0]
    d={"c":_create_ext_dataset(f,"states/c",(N,),np.float64),
       "phi":_create_ext_dataset(f,"states/phi",(N,),np.float64),
       "ball":_create_ext_dataset(f,"game/ball_pos",(2,),np.float64),
       "platform":_create_ext_dataset(f,"game/platform_pos",tuple(),np.float64),
       "score":_create_ext_dataset(f,"game/score",tuple(),np.int32),
       "current":_create_ext_dataset(f,"measurements/measured_current",(3,),np.float64),
       "voltage":_create_ext_dataset(f,"electrodes/voltage_pattern",(18,),np.float64)}
    return f,d

def _append(ds,row): L=ds.shape[0]+1; ds.resize((L,)+ds.shape[1:]); ds[-1]=row

class MemoryElectrodesSurface:
    def __init__(self,Lx=1e-3,Ly=1e-3,Lz=0.25e-3,R=8.314,T=298.0,F=96485.33212,
                 epsilon=80*8.854e-12,D1=1.33e-9,D2=2.03e-9,D3=1e-9,z1=1,z2=-1,
                 c0=10.0,L_c=1e-2,dt=1e-2,nx=16,ny=16,nz=4,experiment="random",surface_radius=None):
        self.Lx,self.Ly,self.Lz=float(Lx),float(Ly),float(Lz)
        self.R,self.T,self.F=float(R),float(T),float(F); self.epsilon=float(epsilon)
        self.D1,self.D2,self.D3=float(D1),float(D2),float(D3); self.z1,self.z2=int(z1),int(z2)
        self.c0=float(c0); self.L_c=float(L_c); self.dt=float(dt)
        self.nx,self.ny,self.nz=int(nx),int(ny),int(nz); self.experiment=str(experiment)
        self.surface_radius = surface_radius if surface_radius is not None else 0.06*min(self.Ly,self.Lz)
        nodes, elements, _ = create_structured_mesh_3d(self.Lx, self.Ly, self.Lz, self.nx, self.ny, self.nz)
        self.nodes,self.elements=nodes,elements
        self.sim=HybridNPENwithFOReaction(mesh=type('M',(),{'nodes':nodes,'elements':elements}),dt=self.dt,
            D1=self.D1,D2=self.D2,D3=self.D3,z1=self.z1,z2=self.z2,epsilon=self.epsilon,R=self.R,T=self.T,L_c=self.L_c,c0=self.c0,
            voltage=0.0,alpha=1.0,alpha_phi=1.0)
        # Use Galerkin for a quick smoke run (switch to 'sg' after validation)
        self.sim.set_advection_scheme("sg")
        self.electrode_centers=self._make_centers(); self.electrode_radii=[self.surface_radius]*len(self.electrode_centers)
        # Precompute face sets once using the C++ mesh (faster than recomputing every step)
        self._face_sets = None
        try:
            # Build face sets by centroid proximity
            import numpy as _np
            cpp_mesh = getattr(self.sim, "_cpp_mesh", None)
            if cpp_mesh is not None:
                nodes_cpp = _np.array(cpp_mesh.getNodes())
                bfaces = _np.array(cpp_mesh.getBoundaryFaces(), dtype=_np.int32)
                tri = nodes_cpp[bfaces]            # (F,3,3)
                centroids = tri.mean(axis=1)       # (F,3)
                face_sets = []
                Lx = float(self.Lx)
                tol = max(1e-12, 1e-6 * Lx)
                for idx_e, (c, r) in enumerate(zip(self.electrode_centers, self.electrode_radii)):
                    c3 = _np.array(c, dtype=float)
                    # Plane filter by x-coordinate
                    if abs(c3[0]) < tol:  # near x=0
                        plane_mask = (tri[:, :, 0].mean(axis=1) <= tol)
                    elif abs(c3[0] - Lx) < tol:  # near x=Lx
                        plane_mask = (abs(tri[:, :, 0].mean(axis=1) - Lx) <= tol)
                    else:
                        plane_mask = _np.ones(centroids.shape[0], dtype=bool)
                    # Radius filter on the masked faces
                    d = _np.linalg.norm(centroids[plane_mask] - c3[None, :], axis=1)
                    idx_local = _np.where(d <= float(r))[0]
                    idx = _np.nonzero(plane_mask)[0][idx_local].astype(_np.int32)
                    face_sets.append(idx.tolist())
                self._face_sets = face_sets
                # face sets ready
            else:
                # Fallback to wrapper helper (Python 2D or if cpp mesh not exposed)
                self.sim.set_electrode_surfaces_from_centers(self.electrode_centers,self.electrode_radii,[0.0]*len(self.electrode_centers),[0.0]*len(self.electrode_centers))
        except Exception:
            # As a last resort, at least set something so step2 can run
            self.sim.set_electrode_surfaces_from_centers(self.electrode_centers,self.electrode_radii,[0.0]*len(self.electrode_centers),[0.0]*len(self.electrode_centers))
    def _make_centers(self)->List[Tuple[float,float,float]]:
        C=[]
        for p in range(6):
            y=(p+0.5)*(self.Ly/6.0); z=0.5*self.Lz; C.append((0.0,y,z)); C.append((self.Lx,y,z))
        return C
    def _init_conditions(self):
        N=self.nodes.shape[0]
        if self.experiment=="gaussian":
            cx,cy,cz=self.Lx/2,self.Ly/2,self.Lz/2; s=self.Lx/10
            c=0.05+0.04*np.exp(-((self.nodes[:,0]-cx)**2+(self.nodes[:,1]-cy)**2+(self.nodes[:,2]-cz)**2)/(2*s**2))
        elif self.experiment.startswith("gradient"):
            x=self.nodes[:,0]/max(self.Lx,1e-12); c=0.25+0.1*x
        elif self.experiment=="random":
            c=0.5+np.random.uniform(-0.1,0.1,N)
        else:
            c=np.full(N,0.5)
        phi=np.zeros(N); return c,phi
    def run(self,surface_voltages:Sequence[np.ndarray],applied_voltage:float,num_steps:int,k_reaction:float=0.5,
            output_path:Optional[str]=None,checkpoint:Optional[str]=None)->str:
        if len(surface_voltages)!=12: raise ValueError("surface_voltages must have length 12")
        for sv in surface_voltages:
            if len(sv)!=num_steps: raise ValueError("each surface voltage has length num_steps")
        c,phi=self._init_conditions()
        meta={"Lx":self.Lx,"Ly":self.Ly,"Lz":self.Lz,"nx":self.nx,"ny":self.ny,"nz":self.nz,
              "dt":self.dt,"num_steps":num_steps,"experiment":self.experiment}
        consts={"R":self.R,"T":self.T,"F":self.F,"epsilon":self.epsilon,"D1":self.D1,"D2":self.D2,"D3":self.D3,
                "z1":self.z1,"z2":self.z2,"c0":self.c0,"L_c":self.L_c}
        f,d=init_h5_output(self.nodes,self.elements,meta,consts,output_path)
        # initial state and placeholders
        _append(d["c"],c); _append(d["phi"],phi); _append(d["ball"],np.array([np.nan,np.nan])); _append(d["platform"],np.nan)
        _append(d["score"],0); _append(d["current"],np.array([np.nan,np.nan,np.nan])); _append(d["voltage"],np.full(18,np.nan))
        # Prepare internal surface maps once (already set in __init__). Update voltages per step.
        centers=self.electrode_centers
        for t in tqdm(range(num_steps), desc="Surface NPEN", leave=True):
            # Update electrode surface voltages: scale amplitudes by applied_voltage
            amps=[float(a[t]) if not np.isnan(a[t]) else np.nan for a in surface_voltages]
            volts=[(applied_voltage*amp if not np.isnan(amp) else np.nan) for amp in amps]
            try:
                if self._face_sets is not None:
                    self.sim.set_electrode_faces(self._face_sets, volts, [k_reaction]*len(self._face_sets))
                else:
                    self.sim.set_electrode_surfaces_from_centers(centers,self.electrode_radii,volts,[k_reaction]*len(centers))
            except Exception:
                pass
            # C++ expects node-based arrays for backward-compatibility of step2; pass empties (surfaces configured)
            c,phi=self.sim.step2(c,phi,np.array([],dtype=np.int32),np.array([],dtype=float),rtol=1e-4,atol=1e-10,max_iter=5)
            _append(d["c"],c); _append(d["phi"],phi)
            # placeholders for game/current and 18-slot voltage pattern
            _append(d["ball"],np.array([np.nan,np.nan])); _append(d["platform"],np.nan); _append(d["score"],0)
            _append(d["current"],np.array([np.nan,np.nan,np.nan])); _append(d["voltage"],np.full(18,np.nan))
        f.flush(); f.close(); return output_path or os.path.join("output","pong_simulation_surface.h5")
