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
    constrained_inds="auto",
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
    config_mapping (map.LinearMap):
        LinearMap characterizing the fg -> cg configurational map.
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
            Map characterizing the optimal force map.
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
    mapped_forces = force_map(points=forces, copoints=xyz)
    if only_return_forces:
        return mapped_forces
    to_return = {}
    to_return.update({"projected_forces": mapped_forces})
    to_return.update({"map": force_map})
    to_return.update({"residual": force_smoothness(mapped_forces)})
    to_return.update({"constraints": constrained_inds})
    return to_return


def project_forces_cv(cv_arg_dict, forces, xyz=None, n_folds=5, *args, **kwargs):
    """A slim wrapper function to perform cross validation over project_forces.

    Note: this function does not choose an optimal model. Instead, it performs
    cross validation for each parameter listed in cv_arg_dict. You should use
    this to select an optimal hyperparameter and then train a production model.

    Note: xyz _is_ split up into folds like forces.

    Arguments
    ---------
    cv_arg_dict (dictionary):
        Contains arguments to run cross validation over. Must be of the
        following (limited) form: {<argument_name>:[arg_val_1,arg_val2,...]}.
        Each val is passed to project_forces as argument_name=arg_val_1.

        Note: Currently, it is only possible to scan over one parameter.
        Therefore, cv_arg_dict must of length 1.
    forces (numpy.ndarray):
        See project_forces; it is split into CV folds before being passed to
        project_forces.
    xyz (numpy.ndarray or None):
        See project_forces; it is split into CV folds before being passed to
        project_forces, unless it is None, in which case it is simply passed.
    n_folds (positive integer):
        Number of cross validation folds to use.
    *args/**kwargs:
        Passed to project_forces.

    Returns
    -------
    dictionary composed of <parameter>:<holdout score>, where parameter is each
    in cv_arg_dict's only value and holdout_score is force_smoothness evaluated
    each fold and then averaged.

    """
    # make fold indices
    n_frames = forces.shape[0]
    frames = np.arange(n_frames)
    _ = np.random.shuffle(frames)
    chunked_frame_inds = np.array_split(ary=frames, indices_or_sections=n_folds, axis=0)

    # create sequence of indices which are outside each fold (for training)
    compl_chunked_frame_inds = []
    for ind, _ in enumerate(chunked_frame_inds):
        outside_chunks = [x for i, x in enumerate(chunked_frame_inds) if i != ind]
        compl_chunked_frame_inds.append(np.concatenate(outside_chunks))

    cv_arg_name = list(cv_arg_dict.keys())[0]
    cv_results = {}
    # iterate over values of parameter
    for cv_arg in cv_arg_dict[cv_arg_name]:
        cv_fold_scores = []
        combined_kwargs = dict(kwargs, **{cv_arg_name: cv_arg})
        # iterate over folds
        for train_inds, val_inds in zip(compl_chunked_frame_inds, chunked_frame_inds):
            # make training data
            train_forces = forces[train_inds]
            if xyz is None:
                train_xyz = None
            else:
                train_xyz = xyz[train_inds]
            # use training data for parameterization
            trained_map = project_forces(
                forces=train_forces, xyz=train_xyz, *args, **combined_kwargs
            )["map"]
            # make validation data
            val_forces = forces[val_inds]
            if xyz is None:
                val_xyz = None
            else:
                val_xyz = xyz[val_inds]
            # use validation data
            cv_fold_scores.append(
                force_smoothness(trained_map(points=val_forces, copoints=val_xyz))
            )
        cv_results.update({cv_arg: sum(cv_fold_scores) / len(cv_fold_scores)})
    return cv_results


def force_smoothness(array):
    r"""Calculates the mean squared element of an array.

    This is proportional to a finite sum approximate of E[||x||^2_2], which
    is often used as a metric of quality for force-maps.
    """

    return np.mean(array**2)
