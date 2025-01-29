# Investigating Phase Behaviour of Micro Solid Particles on a Liquid Surface


## Abstract (Draft)

This project will explore the phase behaviour of objects (e.g. representing micro/nanometer solid particles) floating on a liquid surface. We aim to keep account of a weak repulsion, the presence of gravity and surface tension.  

We expect various regimes to emerge as these three force parameters are varied (and compared to each other and to thermal noise). A condition of particular interest would be a phase separation (density as order parameter) coupled to sinking, i.e. formation of dense regions that gain energy by some sinking.     This would be a strongly non-additive effect, and has not been explored before.

 

The project will tackle this question through numerical simulations.    The first approach will be a lattice mean field (keeping track of two fileds: density and interface height) and finding equilibrium configurations and their properties via Metropolis MC sampling.    If possible within the time, the most interesting areas of parameter space will be explored by coding a Brownian Dynamics simulation where the particles are explicit objects (coupled to an interface height field) - this would allow to study properties of interfaces between particle dense and particle poor regions.

## Current File Structure:
### LatticeClass.py
Python class that simulates particle motion by performing Metropolis algorithm on a 2D lattice, with options to dynamically vary all key parameters, save/display results and track energy of system.

### AnimationCode.py
Skeleton code for creating animations of an instance of the lattice class updating

### AlteringCoefficients.py
Investigating how altering coefficients affects convergence and eqm structure of system.

### Standard_Imports.py
Collecting a list of all useful libraries and imports inc. the Lattice Class.


  
