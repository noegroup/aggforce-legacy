"""Tools for inferring constrained bonds from molecular trajectories.

Useful for automatically obtaining a list of molecularly constrained atoms to
feed into mapping methods. Currently, only pairwise distance constraints are
considered.
"""

import numpy as np
from typing import Set, FrozenSet, Union

Constraints = Set[FrozenSet[int]]


def guess_pairwise_constraints(xyz: np.ndarray, threshold: float = 1e-3) -> Constraints:
    """Find pairs of sites which are likely constrained via fluctuations.

    Fluctuations are found by
    The pairwise distances for each frame are calculated; then, the standard
    deviation for each distance over time is calculated. If this standard
    deviation is lower than a threshold, the two atoms are considered
    constrained.

    Arguments:
    ---------
    xyz (numpy.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    threshold (positive float):
        Distances with standard deviations lower than this value are considered
        to be constrainted. Has units of xyz.

    Returns:
    -------
    A set of frozen sets, each of which contains a pair of indices of sites
    which are guessed to be pairwise constrained.
    """
    dists = distances(xyz)
    sds = np.sqrt(np.var(dists, axis=0))
    np.fill_diagonal(sds, threshold * 2)
    inds = np.nonzero(sds < threshold)
    return {frozenset(v) for v in zip(*inds)}


def distances(
    xyz: np.ndarray,
    cross_xyz: Union[None, np.ndarray] = None,
    return_matrix: bool = True,
    return_displacements: bool = False,
) -> np.ndarray:
    """Calculate the distances for each frame in a trajectory.

    Returns an array where each slice is the distance matrix of a single frame
    of an argument.

    Arguments:
    ---------
    xyz (np.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    cross_xyz (np.ndarray or None):
        An array describing the Cartesian coordinates of a different system over
        time or None; assumed to be of shape (n_steps,other_n_sites,n_dim). If
        present, then the returned distances are those between xyz and cross_xyz
        at each frame.  If present, return_matrix must be truthy.
    return_matrix (boolean):
        If true, then complete (symmetric) distance matrices are returned; if
        false, the upper half of each distance matrix is extracted, flattened,
        and then returned.
    return_displacements (boolean):
        If true, then instead of a distance array, an array of displacements is
        returned.

    Returns:
    -------
    Returns numpy.ndarrays, where the number of dimensions and size depend on
    the arguments.

    If return_displacements is False:
        If return_matrix and cross_xyz is None, returns a 3-dim numpy.ndarray of
        shape (n_steps,n_sites,n_sites), where the first index is the time step
        index and the second two are site indices. If return_matrix and
        cross_xyz is not None, then an array of shape
        (n_steps,other_n_sites,n_sites) is returned. If not return_matrix,
        return a 2-dim array (n_steps,n_distances), where n_distances indexes
        unique distances.
    else:
        return_matrix must be true, and a 4 dimensional array is returned,
        similar to the shapes above but with an additional terminal axis for
        dimension.
    """
    if cross_xyz is not None and not return_matrix:
        raise ValueError("Cross distances only supported when return_matrix is truthy.")
    if return_displacements and not return_matrix:
        raise ValueError("Displacements only supported when return_matrix is truthy.")

    if cross_xyz is None:
        displacement_matrix = xyz[:, None, :, :] - xyz[:, :, None, :]
    else:
        displacement_matrix = xyz[:, None, :, :] - cross_xyz[:, :, None, :]
    if return_displacements:
        return displacement_matrix
    distance_matrix = np.linalg.norm(displacement_matrix, axis=-1)
    if return_matrix:
        return distance_matrix
    n_sites = distance_matrix.shape[-1]
    indices0, indices1 = np.triu_indices(n_sites, k=1)
    subsetted_distances = distance_matrix[:, indices0, indices1]
    return subsetted_distances
