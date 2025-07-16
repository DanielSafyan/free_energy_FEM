# Free Energy Finite Element Method (FEM) Simulation

## Project Goal

This project aims to reproduce the experiments described in the paper "Electro-active polymer hydrogels exhibit emergent memory when embodied in a simulated game environment" by Yoshikatsu Hayashi et al. The simulation focuses on modeling electro-active polymer (EAP) hydrogels using the Finite Element Method (FEM) to demonstrate emergent memory functions.

## Research Context

Based on the paper's abstract:

The goal of artificial neural networks is to utilize the functions of biological brains to develop computational algorithms. However, these purely artificial implementations cannot achieve the adaptive behavior found in biological neural networks (BNNs) via their inherent memory. Alternative computing mediums that integrate biological neurons with computer hardware have shown similar emergent behavior via memory, as found in BNNs.

Our in silico reproduction will simulate these hydrogels using free energy principles and FEM to model the ion migration and emergent properties.

## Dependencies

- NumPy
- SciPy
- Matplotlib

## How to Run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python fickian_diffusion_fem.py
   ```

The script will generate a plot (`diffusion_profile.png`) showing the concentration profile at different time steps.
