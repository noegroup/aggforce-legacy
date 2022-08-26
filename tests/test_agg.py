"""pytest unit tests for the agg module
"""


import os
import numpy as np
from aggforce import linearmap as lm
from aggforce import agg as ag


def test_agg_opt():
    """Test optimized force aggregation for a flexible water dimer"""

    dimerfile = os.path.join(os.path.dirname(__file__), "data/waterdimer.npz")
    dimerdata = np.load(dimerfile)
    forces = dimerdata["Fs"]

    # CG mapping: two oxygens
    inds = [[0], [3]]
    cmap = lm.LinearMap(inds, n_fg_sites=forces.shape[1])
    optim_results = ag.project_forces(
        xyz=None,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=set(),
        solver_args={"solver": "scs"},
    )

    # aggregation mapping: we expect that contributions from each water are added up
    # to cancel the intramolecular bond forces
    agg_mapping = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=float)
    assert np.allclose(optim_results["map"]._standard_matrix, agg_mapping, atol=5e-3)
