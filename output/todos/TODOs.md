# Equation implementation

## Fickian Diffusion **---done---**

## Flory-Huggins **---done---**

## Nernst-Planck-Poisson **---in progress---**

1. make algorithm work **---done---**
2. find suitible initial conditions: **---done---** 
    - look at characteristic scales: length, time, diffusion and other speed related scales **---done---**
3. write a README.md for the Nernst-Planck-Poisson simulation


# Questions

- Why does a change in dt not cause a perceived change in timestep? If i have a diffusion+Drift simulation and I want it to go slower and more step by step, dt does not seem to have any effect on the speed of the simulation. **---???---**


# Refactoring

- Refactor Fickian and FH to use chemical_potential.py and the new fem_mesh.py properly 
