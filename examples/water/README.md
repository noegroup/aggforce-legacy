# Toy system: Water Dimer

This directory contains a simple demonstration of noise-cancelling force matching on a toy system: two water molecules in a harmonic potential.


Run an unconstrained simulation:

    python simulate.py --unconstrained -dt 0.1

The 0.1 fs time step ensures that the fast O-H bond oscillations are resolved.

Run a constrained simulation:

    python simulate.py --platform cpu

Here we can use a 1 fs time step (the default).