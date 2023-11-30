"""Maps forces for coarse-graining molecular dynamics trajectories.

Coarse-grained force-fields can be created by matching the forces produced from a higher
resolution simulation. This process requires the higher resolution forces be mapped.
This module provides routines for performing this mapping in different ways, but does
not itself parameterize any force-fields.

The primary entry point is agg.project_forces.

See agg.py and the README for more information. Tests and examples are also available.
"""
