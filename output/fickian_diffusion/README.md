# Fickian Diffusion from Free Energy Minimization

Fick's laws of diffusion describe the net movement of particles from an area of higher concentration to an area of lower concentration. This macroscopic observation is fundamentally a consequence of the system's tendency to maximize its entropy, or equivalently, minimize its free energy. The driving force for this particle transport is the gradient of the chemical potential (μ).

The flux of particles, `J`, can be expressed as being proportional to the gradient of the chemical potential:

$$ J = -M c \nabla \mu $$

Where:
- `J` is the diffusion flux.
- `M` is the particle mobility.
- `c` is the concentration.
- `μ` is the chemical potential.

For an ideal solution, the chemical potential is given by:

$$ \mu = \mu_0 + k_B T \ln(c) $$

Where:
- `μ_0` is the standard chemical potential.
- `k_B` is the Boltzmann constant.
- `T` is the absolute temperature.

Substituting this into the flux equation yields:

$$ J = -M c \nabla (\mu_0 + k_B T \ln(c)) = -M c (k_B T \frac{1}{c} \nabla c) $$

This simplifies to Fick's first law:

$$ J = -(M k_B T) \nabla c = -D \nabla c $$

Here, the diffusion coefficient `D` is identified through the Einstein relation as `D = M k_B T`.

This formulation demonstrates that the familiar concentration gradient-driven diffusion is an emergent phenomenon rooted in the more fundamental principle of free energy minimization.

## 2D Diffusion Simulation

The following animation shows a 2D simulation of Fickian diffusion, where an initial high concentration of particles in the center spreads out over time.

![2D Diffusion Simulation](../assets/diffusion_video_2D-ezgif.com-speed.gif)
