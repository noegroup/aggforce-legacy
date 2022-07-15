"""Provides tools for inferring constrained atoms from molecular trajectories.
Useful for automatically obtaining a list of molecularly constrained atoms to
feed into mapping methods. Currently, only pairwise distance constraints are
considered.
"""

import numpy as np


def guess_pairwise_constraints(xyz, threshold=1e-3):
    """Finds pairs of atoms which are likely constrained by considering how much
    their pairwise distance fluctuates over a set of frames.

    The pairwise distances for each frame are calculated; then, the standard
    deviation for each distance over time is calculated. If this standard
    deviation is lower than a threshold, the two atoms are considered
    constrained.

    Arguments
    ---------
    xyz (numpy.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    threshold (positive float):
        Distances with standard deviations lower than this value are considered
        to be constrainted. Has units of xyz.

    Returns
    -------
    A set of frozen sets, each of which contains a pair of indices of sites
    which are guessed to be pairwise constrained.
    """

    dists = distances(xyz)
    sds = np.sqrt(np.var(dists, axis=0))
    _ = np.fill_diagonal(sds, threshold * 2)
    inds = np.nonzero(sds < threshold)
    return set(frozenset(v) for v in zip(*inds))


def distances(xyz, return_matrix=True):
    """Calculates the distances for each frame in a trajectory.

    Returns an array where each slice is the distance matrix of a single frame
    of an argument.

    Arguments
    ---------
    xyz (np.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    return_matrix (boolean):
        If true, then complete (symmetric) distance matrices are returned; if
        false, the upper half of each distance matrix is extracted, flattened,
        and then returned.

    Returns
    -------
    If return_matrix, returns a 3-dim numpy.ndarray of shape
    (n_steps,n_sites,n_sites), where the first index is the time step index and
    the second two are site indices. If not return_matrix, return a 2-dim array
    (n_steps,n_distances), where n_distances indexes unique distances.
    """

    distance_matrix = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    if return_matrix:
        return distance_matrix
    n_sites = distance_matrix.shape[-1]
    indices0, indices1 = np.triu_indices(n_sites, k=1)
    subsetted_distances = distance_matrix[:, indices0, indices1]
    return subsetted_distances
