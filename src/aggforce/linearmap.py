r"""Provides routines for finding the optimal linear force aggregation map from a
given molecular trajectory.
"""

import copy
import numpy as np
from qpsolvers import solve_qp


class LinearMap:
    r"""Provides a unified interface for linear maps for transforming from
    fine-grained to coarse-grained systems.

    Allows multiple different representations of the same map to be used.
    Various representations are converted into a consistent internal
    representation, which is then transformed into other formats when needed.

    The primary map format is the "standard_matrix". Given that our linear map
    transforms configurations from the fine-grained (fg) resolution to the
    coarse-grained (cg) resolution, the standard matrix is a (num. of cg
    particles) x (num. of fg particles) where each element describes how a fg
    particle linearly contributes to a cg particle. This can be accessed using
    the standard_matrix attribute.

    Calling instances allows them to map trajectory arrays of the shape
    (n_steps,n_sites,n_dims).
    """

    def __init__(self, mapping, n_fg_sites=None):
        r"""Initializes a LinearMapping object from something describing a map.

        Arguments
        ---------
        mapping (list of lists of integers or 2-d numpy.ndarray):
            If a list of lists, then we assume the outer list iterates over
            various cg indices, and the inner lists describe the indices of
            which atoms contribute to that particular cg site. As this format
            does not make it clear how many total fg sites there are, n_fg_sites
            must be specified. If instead a 2-d numpy.ndarray, then the array is
            assumed to be of shape (num of cg,num of fg), which each element
            describing the coefficient of how the fg site contributes to the cg
            site. In this case n_fg_sites should not be specified.
        n_fg_sites (integer or None):
            Certain mapping descriptions make it ambiguous how many total
            fine-grained sites there are. This variable allows this ambiguity to
            be resolved.

        Example:
            [[0,2,3],[4]] with n_fg_sites=6 describes a 6 particle fg system
            and 2 particle cg system (from the length of the outer list).
            cg particle 0 (equally) depends on fg particles 0,2, and 3 as where
            cg particle 1 depends only on fg particle 4.

            The same information is given by the following 2-d matrix:
                [ 1/3 0   1/3 1/3 0   0  ]
                [ 0   0   0   0   1   0  ]
            Note that in the matrix case, we had to specify the normalization of
            the weights directly, as where in the list format it was done
            automatically.
        """

        if isinstance(mapping, np.ndarray) and len(mapping.shape) == 2:
            if n_fg_sites is not None:
                raise ValueError()
            self._standard_matrix = mapping
        elif hasattr(mapping, "__iter__"):
            # assume we are in the case of iterable of lists
            if n_fg_sites is None:
                raise ValueError()
            mapping = list(mapping)
            n_cg_sites = len(mapping)
            mapping_mat = np.zeros((n_cg_sites, n_fg_sites))
            for site, site_contents in enumerate(mapping):
                local_map = np.zeros(n_fg_sites)
                local_map[site_contents] = 1 / len(site_contents)
                mapping_mat[site, :] = local_map
            self._standard_matrix = mapping_mat
        else:
            raise ValueError()

    @property
    def standard_matrix(self):
        r"""The mapping in standard matrix format."""

        return self._standard_matrix

    @property
    def n_cg_sites(self):
        r"""The number of coarse-grained sites described by the output of the
        map.
        """

        return self._standard_matrix.shape[0]

    @property
    def n_fg_sites(self):
        r"""The number of fine-grained sites described by the input of the
        map.
        """

        return self._standard_matrix.shape[1]

    def __call__(self, to_map):
        r"""Applies map to a particular form of 3-dim array.

        Arguments
        ---------
        to_map (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).

        Returns
        -------
        Combines to_map along the n_sites dimension according to the internal
        map.
        """
        shape = to_map.shape
        reshaped_input = np.reshape(np.swapaxes(to_map, 0, 1), (shape[1], -1))
        reshaped_output = np.matmul(self.standard_matrix, reshaped_input)
        output = np.swapaxes(
            np.reshape(reshaped_output, (-1, shape[0], shape[2])), 0, 1
        )
        return output


def qp_linear_map(
    forces,
    config_mapping,
    constrained_inds=None,
    xyz=None,
    solver_args=dict(
        solver="osqp",
        eps_abs=1e-7,
        max_iter=int(1e3),
        polish=True,
        polish_refine_iter=10,
    ),
):
    r"""Searches for the linear force map which produces the average lowest mean
    square norm of the mapped force.

    Note: Uses a quadratic programming solver with equality constraints.

    Arguments
    ---------
    forces (np.ndarray):
        three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        forces of the FG sites as a function of time.
    config_mapping (np.ndarray):
        LinearMap object which characterizes configurational map.
    constrained_inds (set of frozensets):
        Each entry is a frozenset of indices, the group of which is constrained.
        Currently, only bond constraints (frozensets of size 2) are supported.
    xyz (None):
        Ignored. Included for compatibility with the interface of other methods.
    solver_args (dict):
        Passed as options to qp_solve to solve quadratic program.

    Returns
    -------
    LinearMap object characterizing force mapping.
    """

    # flatten force array
    reshaped_fs = qp_form(forces)
    # construct geom constraint matrix
    # prep matrices for solver
    con_mat = make_bond_constraint_matrix(config_mapping.n_fg_sites, constrained_inds)
    reg_mat = np.matmul(reshaped_fs, con_mat)
    qp_mat = np.matmul(reg_mat.T, reg_mat)
    zero_q = np.zeros(qp_mat.shape[0])
    per_site_maps = []
    # run solver
    for ind in range(config_mapping.n_cg_sites):
        sbasis = np.zeros(config_mapping.n_cg_sites)
        sbasis[ind] = 1
        constraint_mat = np.matmul(config_mapping.standard_matrix, con_mat)
        gen_map = solve_qp(
            P=qp_mat, q=zero_q, A=constraint_mat, b=sbasis, **solver_args
        )
        per_site_maps.append(np.matmul(con_mat, gen_map))
    return LinearMap(np.stack(per_site_maps))


def qp_form(target):
    r"""Transforms a 3 array (target) to a particular form of 2-array.

    e.g. target is (n_steps,n_sites,n_dims=3)
    output is (n_steps*n_dims,n_sites) where the rows are ordered as
        step=0, dim=0
        step=0, dim=1
        step=0, dim=2
        step=1, dim=0
    """

    mixed = np.swapaxes(target, 1, 2)
    reshaped = np.reshape(mixed, (mixed.shape[0] * mixed.shape[1], -1))
    return reshaped


def make_bond_constraint_matrix(n_sites, constraints):
    r"""Makes a molecular constraint matrix connective a generalized mapping
    coefficient to the expanded mapping coefficient.

    When creating optimal force maps, atoms which are molecularly constrained to
    each other are most easily handled by having them share mapping
    coefficients. We do so by creating a reduced mapping vector, that when
    multiplied by the matrix this function produces, creates a full sized
    mapping vector.  Optimization may then be performed over the reduced (or
    generalized) vector.

    This is done by creating a matrix that duplicates pertinent indices in the
    reduced vector over multiple atoms. This results in (as an example) a
    matrix like this:
        [1 0 0 0]
        [0 1 0 0]
        [0 1 0 0]
        [0 0 1 0]
        [0 0 0 1]
    when multiplied by a reduced vector [a b c d] on the right, this results in
    the vector [a b b c d], i.e., sites 1 and 2 are constrained to have the
    same value.

    Arguments
    ---------
    n_sites (integer):
        Total number of sites in the system. In the context of the quadratic
        programming in this module, usually the number of fine-grained
        particles in the system.
    constraints (set of frozensets of integers):
        Each member set contains fine-grained site indices of atoms which are
        constrained relative to each other (and should have identical mapping
        coefficients)

    Returns
    -------
    2-dim numpy.ndarray
    """
    # aggregate various constraints if needed
    rconstraints = reduce_constraint_sets(constraints)
    # get number of DOFs we will remove
    n_constrained_atoms = sum((len(x) for x in rconstraints))
    reduced_n_sites = n_sites - n_constrained_atoms + len(rconstraints)
    # make look up dictionary so that we know which site are dependent on which
    # atoms
    index_lookup = constraint_lookup_dict(rconstraints)
    mat = np.zeros((n_sites, reduced_n_sites))
    offset = 0
    # place all sites that don't depend on another site
    for site in range(n_sites):
        if site not in index_lookup:
            mat[site, offset] = 1
            offset += 1
    # place all sites that do depend on another site
    for site, anchor in index_lookup.items():
        mat[site, :] = mat[anchor, :]
    return mat


def reduce_constraint_sets(constraints):
    r"""Reduces a set of sets of constrained sites into a set of larger disjoint
    sets of constrained sites.

    If a single atom has a constrained bonds to two separate atoms, the list of
    bond constraints does not make it clear that all three of these atoms are
    (for the purpose of force mappings in this module ) all constrained
    together. This method replaces the two bond constraint entries with a 3 atom
    constraint entry, but does so for all atoms and all sized constrains such
    that the returned list of constraint's members are all disjoint.

    Arguments
    ---------
    constraints (set of frozensets of integers):
        Each member set contains indices of atoms which are constrained relative
        to each other

    Returns
    -------
    set of frozensets of integers.

    Example:
        {{1,2},{2,3},{4,5}}
        is transformed into
        {{1,2,3},{4,5}}

        In other words, {1,2} and {2,3} were combined because they both
        contained 2, while {4,5} was untouched as it did not share any elements
        with any other sets.

    NOTE: This function has complicated flow and is not proven to be correct. It
    seems to be a form of flood search using breadth first search should be
    revised.
    """

    constraints_copy = copy.copy(constraints)
    agged_constraints = set()

    if len(constraints) <= 1:
        return constraints_copy

    # this control flow is very bad, but we do not a good refactor yet.

    # We pop an element from constraints copy, and see if it has a non-empty
    # intersection with any other elements in constraints copy.  If so, we union
    # those sets into our selected element. The elements we had similarity with
    # in constraints copy are removed.  We repeat this process until we do not
    # see any candidates to add, at which point we add our selected (aggregated)
    # element to a new set, take the next element from the constraints copy, and
    # begin again. The new set is returned when we run out of elements in
    # constraints copy to process.

    new = set(constraints_copy.pop())
    second_try = False
    while True:
        to_add = [x for x in constraints_copy if new.intersection(x)]
        new = new.union(*to_add)
        constraints_copy.difference_update(to_add)
        if not to_add:
            agged_constraints.add(frozenset(new))
            if second_try:
                second_try = False
                try:
                    new = constraints_copy.pop()
                except KeyError:
                    break
            else:
                second_try = True
    return agged_constraints


def constraint_lookup_dict(constraints):
    r"""Transforms a set of frozensets of constraints to a dictionary connecting
    each set member to a master member.

    The smallest member of each member set is designated as the parent member.
    Each remaining element in that set is added to the return dictionary as a
    key which points to its parent member.

    Example:
        constraints = {{1,2,3},{4,5},{6,7}}
        is transformed into the following dictionary:
        {
            3:1
            2:1
            5:4
            7:6
        }
        In other words, 1 is the parent member of the first member set, so 2 and
        3 point to 1, etc.

    This is useful when setting up matrices for the quadratic programming
    problem when molecular constraint are present.
    """

    mapping = {}
    for group in constraints:
        sites = sorted(group)
        anchor = sites[0]
        _ = [mapping.update({s: anchor}) for s in sites[1:]]
    return mapping


def constraint_aware_uni_map(
    config_mapping,
    constrained_inds=None,
    xyz=None,
    forces=None,
):
    r"""Produces a uniform basic force map compatible with constraints.

    The given configurational map associates various fine-grained (fg) sites with
    each coarse grained (cg) site. This creates a force-map which:
        - aggregates forces from each fg site that contributes to a cg site
        - aggregates forces from atoms which are constrained with atoms included
          via the previous point

    No weighting is applied to the forces before aggregation.

    For example, if we use a carbon alpha slice configurational mapping, any
    carbon alphas which are constrained to hydrogens will have the forces from
    those hydrogens aggregated with their forces. Carbon alphas that are not
    connected to constrained atoms

    NOTE: The configurational map is not checked for any kind of correctness.

    Arguments
    ---------
    config_mapping (linearmap.LinearMap):
        LinearMap object characterizing the configurational map characterizing
        the connection between the fine-grained and coarse-grained systems.
    constrained_inds (None or set of frozen sets):
        Each set entry is a set of indices, the group of which is constrained.
        Currently, only bond constraints (frozen sets of 2 elements) are supported.
    xyz:
        Ignored. Included for compatibility with other mapping methods.
    forces:
        Ignored. Included for compatibility with other mapping methods.

    Returns
    -------
    LinearMap object describing a force-mapping.
    """

    # get which sites have nonzero contributions to each cg site
    cg_sets = [set(np.nonzero(row)[0]) for row in config_mapping.standard_matrix]
    constraints = reduce_constraint_sets(constrained_inds)
    # add atoms which are related by constraint to those already in cg sites
    for group in cg_sets:
        _ = [group.update(x) for x in constraints if group.intersection(x)]
    force_map_mat = np.zeros_like(config_mapping.standard_matrix)
    # place a 1 where all original or those pulled in by constraints are
    for cg_index, cg_contents in enumerate(cg_sets):
        force_map_mat[cg_index, list(cg_contents)] = 1.0
    return LinearMap(force_map_mat)
