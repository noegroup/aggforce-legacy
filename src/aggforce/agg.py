r"""Provides an interface for optimally aggregating forces from a given molecular
trajectory.

Methods are described in the following problem setting:

We have a fine grained system (with n_fg particles) which we map to
coarse-grained system (with n_cg particles) using a linear mapping function. The
configurational portion of this map is already set; methods here provide ways to
calculate the force map.
"""

from gc import collect
from itertools import product
import numpy as np
from collections import namedtuple
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
        constraints=constrained_inds,
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


def project_forces_grid_cv(cv_arg_dict, forces, xyz=None, n_folds=5, *args, **kwargs):
    """Cross validation over project_forces using a grid of parameters.

    Note: this function does not choose an optimal model. Instead, it performs
    cross validation for each parameter listed in cv_arg_dict. You should use
    this to select an optimal hyperparameter and then train a production model.

    Arguments
    ---------
    cv_arg_dict (dictionary):
        Contains arguments to run cross validation over. Must be of the
        following (limited) form: {<argument_name>:[arg_val_1,arg_val2,...]}.
        Each val is passed to project_forces as argument_name=arg_val_1.
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
    dictionary composed of a series of dictionaries containing
    <parameters>:<holdout score> pairs, where parameter is each is
    force_smoothness evaluated each fold and then averaged. 'scores' indexes the
    for mean force fluctuation values, 'sds' indexes their sample standard
    deviations, and 'n_runs' indexes the number of optimization runs that
    completed successfuly. If no runs successfully finish, then the standard
    deviation and mean entries are set to None. <parameters> is represented by a
    custom namedtuple-derived instance.
    """

    # make fold indices
    n_frames = forces.shape[0]
    frames = np.arange(n_frames)
    np.random.shuffle(frames)
    chunked_frame_inds = np.array_split(ary=frames, indices_or_sections=n_folds, axis=0)

    # create sequence of indices which are outside each fold (for training)
    compl_chunked_frame_inds = []
    for ind, _ in enumerate(chunked_frame_inds):
        outside_chunks = [x for i, x in enumerate(chunked_frame_inds) if i != ind]
        compl_chunked_frame_inds.append(np.concatenate(outside_chunks))

    procced_cv_args = process_cvargs(cv_arg_dict)
    cv_results = dict(scores={}, sds={}, n_runs={})
    # iterate over values of parameter
    for cv_arg_label, cv_arg_dict in procced_cv_args:
        cv_fold_scores = []
        combined_kwargs = dict(kwargs, **cv_arg_dict)
        # iterate over folds
        for train_inds, val_inds in zip(compl_chunked_frame_inds, chunked_frame_inds):
            # make training data
            train_forces = forces[train_inds]
            if xyz is None:
                train_xyz = None
            else:
                train_xyz = xyz[train_inds]
            # use training data for parameterization
            try:
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
                del trained_map
            except ValueError as e:
                print(e)
            collect()
        if len(cv_fold_scores) > 0:
            mean = sum(cv_fold_scores) / len(cv_fold_scores)
            sd = sum([(o - mean) ** 2 for o in cv_fold_scores])
            sd /= len(cv_fold_scores) - 1
            sd = sd ** (0.5)
        else:
            sd = None
            mean = None
        cv_results["scores"].update({cv_arg_label: mean})
        cv_results["sds"].update({cv_arg_label: sd})
        cv_results["n_runs"].update({cv_arg_label: len(cv_fold_scores)})
    return cv_results


def process_cvargs(arg_dict):
    """Transforms a dictionary representing argument values into a reformatted list
    representing a grid of parameter combinations.

    Arguments
    ---------
    arg_dict (dictionary):
        Arguments to process. Assumed to have the form
        {
            key1: [key1_arg1,key1_arg2,...]
            key2: [key1_arg1,key1_arg2,...]
            ...
        }
        where key* are the names of the arguments, and key*_arg* are the
        argument values.

    Returns
    -------
    A list, where entries are tuples of the form (using the example above):
        (<namedtuple>(key1=key1_arg1, key2=key1_arg1),
                                    {key1:key1_arg1, key2:key2_arg_1})
        (<namedtuple>(key1=key1_arg2, key2=key2_arg1) :
                                    {key1:key1_arg1, key2:key1_arg_1})
                ...
    i.e., the first element of each tuple is a namedtable derived instance and
    the second element is a dictionary that can be passed containing command
    parametres.  An entry is given for every combination of parameters. The
    namedtuple entries are instances of a named tuple constructed to have a
    field for each parameter.
    """

    # the parameter names we are going to make a grid over
    param_names = list(arg_dict.keys())
    # values the parameters can take
    values = [content for _, content in arg_dict.items()]
    cross_values = product(*values)
    to_return = []
    C = namedtuple("CVArgs", param_names)
    for values in cross_values:
        key = C(**dict(zip(param_names, values)))
        sub_args = {}
        for name in param_names:
            sub_args.update({name: getattr(key, name)})
        to_return.append((key, sub_args))
    return to_return


def force_smoothness(array):
    r"""Calculates the mean squared element of an array.

    This is proportional to a finite sum approximate of E[||x||^2_2], which
    is often used as a metric of quality for force-maps.
    """

    return np.mean(array**2)
