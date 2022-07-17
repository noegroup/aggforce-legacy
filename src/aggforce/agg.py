r"""Provides an interface for optimally aggregating forces from a given molecular
trajectory.

Methods are described in the following problem setting:

We have a fine grained system (with n_fg particles) which we map to
coarse-grained system (with n_cg particles) using a linear mapping function. The
configurational portion of this map is already set; methods here provide ways to
calculate the force map.
"""

import numpy as np
from . import linearmap
from . import constfinder


def project_forces(
    xyz,
    forces,
    config_mapping,
    constrained_inds='auto',
    method=linearmap.qp_linear_map,
    only_return_forces=False,
    **kwargs
):
    r"""Returns for an optimal force map.

    NOTE: Performs convenience operations (e.g., making sure the mapping matrix
    is in the correct form) so that internal methods can have strong assumptions
    about arguments.

    Arguments
    ---------
    xyz (np.ndarray):
        Three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        positions of the fg sites as a function of time.  Note that in the case
        of linear force maps, the content of this argument is ignored for
        finding forces. However, if constrained_inds is set to 'auto', it may
        still be used to find possible constraints.
    forces (np.ndarray):
        Three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        forces on the fg sites as a function of time.
    config_mapping (linearmap.LinearMap):
        LinearMap  characterizing the fg -> cg configurational map.
    constrained_inds (set of frozensets or 'auto'):
        If a set of frozensets, then each entry is a frozenset of indices, the
        group of which is constrained.  Currently, only bond constraints (frozen
        sets of 2 elements) are supported.  if 'auto', then
        guess_pairwise_constraints is used to generate a list of constrained
        atoms. All of xyz is passed to this function; if more flexibility is
        desired, call it externally and pass its output through this argument.
    method (callable):
        Specifies what method to use to find the optimal map.
    only_return_forces (boolean):
        If true, only the mapped forces are returned. If false, a dictionary
        with more results in returned.
    kwargs:
        Passed to method.

    Returns
    -------
    If only_return_forces, return an np.ndarray of shape
    (n_steps,n_cg_sites,n_dims) which contains the optimally mapped forces.
    If not only_return_forces, a dictionary with the following elements is
    returned:
        projected_force =
            np.ndarray of shape (n_steps,n_cg_sites,n_dims).
        map =
            LinearMap characterizing the optimal force map.
        residual =
            Force map residual calculated using force_smoothness. Note that this
            is not performed on a hold-out set, so be wary of overfitting.
        constraints =
            Set of frozensets characterizing the molecular constraints on the
            system. Useful if constrained_inds is set to 'auto'.
    """

    if constrained_inds == "auto":
        constrained_inds = constfinder.guess_pairwise_constraints(xyz)
    force_map = method(
        xyz=xyz,
        config_mapping=config_mapping,
        forces=forces,
        constrained_inds=constrained_inds,
        **kwargs
    )
    mapped_forces = force_map(forces)
    if only_return_forces:
        return mapped_forces
    to_return = {}
    to_return.update({"projected_forces": mapped_forces})
    to_return.update({"map": force_map})
    to_return.update({"residual": force_smoothness(mapped_forces)})
    to_return.update({"constraints": constrained_inds})
    return to_return


def force_smoothness(array):
    r"""Calculates the mean squared element of an array.

    This is proportional to a finite sum approximate of E[||x||^2_2], which
    is often used as a metric of quality for force-maps.
    """

    return np.mean(array**2)
