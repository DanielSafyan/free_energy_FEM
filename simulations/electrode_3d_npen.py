import numpy as np
import os
from tqdm import tqdm
# Matplotlib is now used for the 2D shadow plot
import matplotlib.pyplot as plt


from utils.fem_mesh import TetrahedralMesh, create_structured_mesh_3d
from simulations.HybridNPENwithFOReaction import HybridNPENwithFOReaction
from utils.temporal_voltages import NPhasesVoltage, TemporalVoltage



