r"""Provides routines for the optimal linear force maps."""

from typing import Union, Dict
from typing_extensions import TypedDict
from itertools import product
import copy
import numpy as np
from qpsolvers import solve_qp  # type: ignore [import-untyped]
from .map import LinearMap
from .constfinder import Constraints

SolverOptions = TypedDict(
    "SolverOptions",
    {
        "solver": str,
        "eps_abs": float,
        "max_iter": int,
        "polish": bool,
        "polish_refine_iter": int,
    },
)
DEFAULT_SOLVER_OPTIONS: SolverOptions = {
    "solver": "osqp",
    "eps_abs": 1e-7,
    "max_iter": int(1e3),
    "polish": True,
    "polish_refine_iter": 10,
}


def qp_linear_map(
    forces: np.ndarray,
    config_mapping: LinearMap,
    constraints: Union[None, Constraints] = None,
    l2_regularization: float = 0.0,
    xyz: Union[np.ndarray, None] = None,  # noqa: ARG001
    solver_args: SolverOptions = DEFAULT_SOLVER_OPTIONS,
) -> LinearMap:
    r"""Search for optimal linear force map.

    Optimally is determined via  average lowest mean square norm of the mapped force.

    Note: Uses a quadratic programming solver with equality constraints.

    Arguments:
    ---------
    forces (np.ndarray):
        three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        forces of the FG sites as a function of time.
    config_mapping (np.ndarray):
        LinearMap object which characterizes configurational map.
    constraints (set of frozensets):
        Each entry is a frozenset of indices, the group of which is constrained.
        Currently, only bond constraints (frozensets of size 2) are supported.
    l2_regularization (float):
        if positive, a l2 normalization of the (full) mapping vector is applied
        with this coefficient.
    xyz (None):
        Ignored. Included for compatibility with the interface of other methods.
    solver_args (dict):
        Passed as options to qp_solve to solve quadratic program.

    Returns:
    -------
    LinearMap object characterizing force mapping.
    """
    if constraints is None:
        constraints = set()
    # flatten force array
    reshaped_fs = qp_form(forces)
    # construct geom constraint matrix
    # prep matrices for solver
    con_mat = make_bond_constraint_matrix(config_mapping.n_fg_sites, constraints)
    reg_mat = np.matmul(reshaped_fs, con_mat)
    qp_mat = np.matmul(reg_mat.T, reg_mat)
    zero_q = np.zeros(qp_mat.shape[0])
    per_site_maps = []
    # since we want to penalize the norm of the expanded vector, we add
    # con_mat.t*con_mat
    if l2_regularization > 0.0:
        qp_mat += l2_regularization * np.matmul(con_mat.T, con_mat)
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


def qp_form(target: np.ndarray) -> np.ndarray:
    r"""Transform 3-array to a particular form of 2-array.

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


def make_bond_constraint_matrix(n_sites: int, constraints: Constraints) -> np.ndarray:
    r"""Make constraint matrix connecting a generalized maps to the expanded maps.

    This matrix connects a generalized mapping coefficient to the expanded
    mapping coefficient.

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

    Arguments:
    ---------
    n_sites (integer):
        Total number of sites in the system. In the context of the quadratic
        programming in this module, usually the number of fine-grained
        particles in the system.
    constraints (set of frozensets of integers):
        Each member set contains fine-grained site indices of atoms which are
        constrained relative to each other (and should have identical mapping
        coefficients)

    Returns:
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


def reduce_constraint_sets(constraints: Constraints) -> Constraints:
    r"""Reduces constraints to disjoint constraints.

    Reduces a set of frozensets of constrained sites into a set of larger disjoint
    frozensets of constrained sites.

    If a single atom has a constrained bonds to two separate atoms, the list of
    bond constraints does not make it clear that all three of these atoms are
    (for the purpose of force mappings in this module ) all constrained
    together. This method replaces the two bond constraint entries with a 3 atom
    constraint entry, but does so for all atoms and all sized constrains such
    that the returned list of constraint's members are all disjoint.

    Arguments:
    ---------
    constraints (set of frozensets of integers):
        Each member set contains indices of atoms which are constrained relative
        to each other

    Returns:
    -------
    set of frozensets of integers.

    Example:
    -------
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

    new = frozenset(constraints_copy.pop())
    second_try = False
    while True:
        to_add = [x for x in constraints_copy if new.intersection(x)]
        new = new.union(*to_add)
        constraints_copy.difference_update(to_add)
        if not to_add:
            agged_constraints.add(new)
            if second_try:
                second_try = False
                try:
                    new = frozenset(constraints_copy.pop())
                except KeyError:
                    break
            else:
                second_try = True
    return agged_constraints


def constraint_lookup_dict(constraints: Constraints) -> Dict[int, int]:
    r"""Transform constraints to a dictionary connecting each member to a parent.

    Transforms a set of frozensets of constraints to a dictionary connecting
    each set member to a master member.

    The smallest member of each member set is designated as the parent member.
    Each remaining element in that set is added to the return dictionary as a
    key which points to its parent member.

    Arguments:
    ---------
    constraints:
        Constraints to collapse

    Example:
    -------
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
    config_mapping: LinearMap,
    constraints: Union[None, Constraints] = None,
    xyz: Union[None, np.ndarray] = None,  # noqa: ARG001
    forces: Union[None, np.ndarray] = None,  # noqa: ARG001
) -> LinearMap:
    r"""Produce a uniform basic force map compatible with constraints.

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

    Arguments:
    ---------
    config_mapping (linearmap.LinearMap):
        LinearMap object characterizing the configurational map characterizing
        the connection between the fine-grained and coarse-grained systems.
    constraints (None or set of frozen sets):
        Each set entry is a set of indices, the group of which is constrained.
        Currently, only bond constraints (frozen sets of 2 elements) are supported.
    xyz:
        Ignored. Included for compatibility with other mapping methods.
    forces:
        Ignored. Included for compatibility with other mapping methods.

    Returns:
    -------
    LinearMap object describing a force-mapping.
    """
    if constraints is None:
        constraints = set()
    # get which sites have nonzero contributions to each cg site
    cg_sets = [set(np.nonzero(row)[0]) for row in config_mapping.standard_matrix]
    constraints = reduce_constraint_sets(constraints)
    # add atoms which are related by constraint to those already in cg sites
    for group, x in product(cg_sets, constraints):
        if group.intersection(x):
            group.update(x)
    force_map_mat = np.zeros_like(config_mapping.standard_matrix)
    # place a 1 where all original or those pulled in by constraints are
    for cg_index, cg_contents in enumerate(cg_sets):
        force_map_mat[cg_index, list(cg_contents)] = 1.0
    return LinearMap(force_map_mat)
