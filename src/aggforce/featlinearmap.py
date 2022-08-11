r"""Provides data-based routines for finding the optimal force aggregation map
that is linear with respect to fixed molecular features.
"""

from copy import deepcopy
import numpy as np
from numpy.random import choice
import scipy.sparse as ss
from qpsolvers import solve_qp
from .map import CLAMap, smear_map
from .linearmap import reduce_constraint_sets
from .constfinder import distances
from queue import SimpleQueue, Empty


def qp_feat_linear_map(
    forces,
    xyz,
    config_mapping,
    featurizer,
    kbt,
    n_constraint_frames=20,
    constraints=set(),
    sparse=True,
    solver_args=dict(
        solver="osqp",
        eps_abs=1e-7,
        max_iter=int(1e3),
        polish=True,
        polish_refine_iter=10,
    ),
    l2_regularization=1e1,
):
    r"""Searches for the force map which produces the average lowest mean
    square norm of the mapped force. The produced map is linear with respect
    to user-provided features.

    Note: Uses a quadratic programming solver with equality constraints.

    Arguments
    ---------
    forces (numpy.ndarray):
        Three dimensional array of shape (n_frames,n_sites,n_dims). Contains the
        forces of the fine-grained sites as a function of time.
    xyz (numpy.ndarray):
        Three dimensional array of shape (n_frames,n_sites,n_dims). Contains the
        configurational coordinates of the fine-grained sites as a function of time.
    config_mapping (map.LinearMap):
        LinearMap object which characterizes configurational map.
    featurizer (callable):
        A callable for featurizing the system which uses the following as input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                copoints (positions) used for featurization
            config_mapping (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list/generator  of feature mats (n_frames, n_fg_sites, n_feats)
                (see below for more information)
            'divs': list/generator of divergence mats (n_frames, n_feats, n_dim)
                (see below for more information)
            'names': None or list of name strings (n_feat)
        where n_feat is the dimensionality of the features. Note that this must
        perform all of the featurizations for all sites simultaneously.
    kbt (float):
        Boltzmann constant times the temperature of the system. The units must
        match the energy portion of the forces units. Needed to account for the
        divergence correction.
    n_constraint_frames (positive integer):
        Number of frames to sample when constraining featurization index to obey
        orthogonality constraints between coarse-grained sites. A larger number
        provides a more exact constraint, but increases memory usage.
    constraints (set of frozensets of integers):
        Each entry is a frozenset of indices, the group of which is constrained.
        Currently, only bond constraints (frozensets of size 2) are supported.
    sparse (boolean):
        If true, then various matrices generated in the method are sparsified
        before being passed to the solver. This does not typically reduce the
        runtime memory usage but may remove warnings given by solvers.
    solver_args (dict):
        Passed as options to qp_solve to solve quadratic program via **.
    l2_regularization (float):
        Coefficient for penalizing the l2 norm of the parameter vector during
        quadratic programming. Note that this l2 penalty is not the same as
        that present in the simple linear map optimization: here, we penalize
        the size of the coefficient vector, as where there we penalize the
        resulting mapping. Doing so here is more complex as the mapping changes
        as a function of frame.

    NOTE: The featurization function output must conform to the following
    format. The entry for the 'feat' and 'div' keys is a list, and each
    element in each list corresponds to the features that are used for a
    particular coarse-grained site; e.g., the first entry in both the feat and
    div entries correspond to coarse-grained site index 0.

    Each feats list entry is of the shape (n_frames, n_fg_sites, n_feats), where
    n_frames is the number of trajectory frames, n_fg_sites is the number of
    fine-grained sites in the system, and n_feats is the number of features
    produced (each fine-grained site in each frame has multiple features).

    Each div list entry characterizes divergence and has the shape (n_frames,
    n_feats, n_dim). In this the feature for every fg_site and frame is d'd
    w.r.t each Cartesian component (this means 3 times as many values as the
    features, but is then summed over the fine-grained sites).

    See the documentation for exp2_feat for more information.

    Returns
    -------
    CLAMap object characterizing force mapping.
    """

    feat_results = featurizer(xyz, config_mapping, constraints)
    feats, divs, names = [feat_results[key] for key in ["feats", "divs", "names"]]

    per_site_feat_coef = []
    for ind, (feat, div) in enumerate(zip(feats, divs)):
        # make constraint arrays
        constr_mult, constr_target = _constr_arrays(
            features=feat,
            cg_ind=ind,
            config_mapping=config_mapping,
            n_frames=n_constraint_frames,
            sparse=sparse,
        )

        # create the force aggregation part of the matrix
        # <forces> * <dim mat> * <features> matrix
        force_features = np.einsum("...af,...ad->...fd", forces, feat)

        # <divergence> is already ready
        # mat1 + kbt mat 2
        # div is an inconvenient shape in the proofs
        reshaped_div = np.swapaxes(div, 1, 2)
        # combine agg'd forces and div term
        ms_reg_mat = force_features + kbt * reshaped_div
        reg_mat = np.reshape(ms_reg_mat, newshape=(-1, ms_reg_mat.shape[2]))
        qp_mat = np.matmul(reg_mat.T, reg_mat)
        if l2_regularization > 0:
            qp_mat += np.diag((l2_regularization,) * qp_mat.shape[0])
        if sparse:
            qp_mat = ss.csc_matrix(qp_mat)
        params = solve_qp(
            P=qp_mat,
            q=np.zeros(qp_mat.shape[0]),
            A=constr_mult,
            b=constr_target,
            **solver_args,
        )
        if params is None:
            raise ValueError("Map optimization failed.")
        per_site_feat_coef.append(params)

    force_mapping = _feat_linear_mapping(
        featurizer=featurizer,
        coefs=per_site_feat_coef,
        mapping=config_mapping,
        constraints=constraints,
        tags=dict(feat_names=names, coef_list=per_site_feat_coef),
    )

    return force_mapping


def _constr_arrays(features, cg_ind, config_mapping, n_frames, sparse=True):
    """Produces a constraint 2-array (A) and target 1-array (b) for
    later Ax=b constrained quadratic optimization.

    The returned 2-array is of shape (n_cg_sites*n_frames,n_features) and the
    1-array is of shape (n_cg_sites*n_frames,). Is sparse is true, the first
    array is a sparse csc matrix.

    Arguments
    ---------
    features (numpy.ndarray):
        Array characterizing the features for a single coarse-grained site.
        Assumed to be of shape (n_frames,n_fg_sites,n_features).
    cg_ind (positive integer):
        Later optimization is performed for each cg site individually. The
        particular cg site index used there is also passed here, and is used
        when creating the output target 1-array.
    config_mapping (map.LinearMap instance):
        Configurational mapping characterizing the coarse-grained resolution.
        Used when creating the output 2-array (including setting n_cg_sites).
    n_frames (positive integer):
        Number of frames to randomly sample from features when constructing
        the constraint matrix. Larger values use more memory, but provide a more
        exact constraint in later optimization.
    sparse (boolean):
        If true, the 2-array returned is a sparse csc array. This may
        avoid warnings from various external numerical solvers. Note that
        algebraic operations on this output may change as it is a matrix and not
        and array.

    Returns
    -------
    Ordered pair: first element is a (n_cg_sites*n_frames,n_features) sized 2-array
    (or scipy.sparse.csc_matrix, see below), second element is 1-dimensional array
    of shape (n_cg_sites*n_frames,).

    If sparse, a sparse csc_matrix is returned for the first array. Note that
    the algebraic operations change as it is a matrix and not and array.
    """

    # get random subset of calculated features
    frame_indices = choice(len(features), size=n_frames, replace=False)
    subsetted_features = features[frame_indices]

    mult_array = np.einsum(
        "ca,...af->...cf", config_mapping.standard_matrix, subsetted_features
    )
    target_array = np.zeros((n_frames, config_mapping.n_cg_sites))
    target_array[:, cg_ind] = 1

    cmshape = mult_array.shape
    mult = np.reshape(mult_array, (-1,) + cmshape[-1:])
    target = np.reshape(target_array, (-1,))
    if sparse:
        mult = ss.csc_matrix(mult)
    return (mult, target)


def _feat_linear_mapping(featurizer, coefs, mapping, constraints, **kwargs):
    """Creates a CLAMap instance for a map that is constructed from a
    fixed parameterization and linear coefficients.

    Arguments
    ---------
    featurizer (callable):
        A callable for featurizing the system which uses the following as input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                copoints (positions) used for featurization
            config_mapping (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list of feature mats (n_frames, n_fg_sites, n_feats)
                (see below for more information)
            'divs': list of divergence mats (n_frames, n_feats, n_dim)
                (see below for more information)
            'names': None or list of name strings (n_feat)
        where n_feat is the dimensionality of the features. Note that this must
        perform all of the featurizations for all sites simultaneously.
    coefs (list of 1-D numpy arrays):
        List of 1-D numpy arrays which are combined with the outputted features
        in the spirit of matmul(<feat>,<coefs>).  Note that these do not _need_
        to be the same sized arrays, but rather must be compatible with the
        output of featurizer (which may vary in dimensionality across
        coarse-grained sites).
    mapping (LinearMap instance):
        Passed to featurizer and used to extract number of coarse-grained sites.
    constraints (set of frozensets):
        Each entry is a frozenset of coarse-grained indices, the group of which
        is constrained. Passed to featurizer.

    Returns
    -------
    CLAMap instance.
    """

    def scale_f(copoints):
        feats = featurizer(copoints, mapping, constraints)["feats"]
        weights = [np.einsum("...ij,j->...i", f, c) for f, c in zip(feats, coefs)]
        return np.stack(weights, axis=1)

    def trans_f(copoints):
        divs = featurizer(copoints, mapping, constraints)["divs"]
        weights = [np.einsum("tij,i->tj", f, c) for f, c in zip(divs, coefs)]
        return np.stack(weights, axis=1)

    force_mapping = CLAMap(
        scale=scale_f,
        trans=trans_f,
        n_fg_sites=mapping.n_fg_sites,
        zeroes_check=True,
        **kwargs,
    )

    return force_mapping


def id_feat(points, cmap, constraints, return_ids=False):
    """Vector valued feature which associates a one-hotted label to each
    fine-grained site.  Labels are unique, except that they are shared between
    constraint groups.

    Arguments
    ---------
    points (numpy.ndarray):
        Points characterizing the positions of fine-grained sites as a function
        of time; assumed to be of shape (n_frames,n_sites,n_dims=3). Note that
        this is only used to find the number of frames of the system, but is
        included for compatibility with other featurization functions.
    cmap (map.LinearMap):
        Configurational map characterizing the relationship between the
        coarse-grained and fine-grained resolutions.
    constraints (set of frozensets of integers):
        Each frozenset describes fine-grained sites participating in a
        constraint. These constraint subsets may overlap.
    return_ids (boolean):
        If return_ids is True, return a numpy.ndarray containing the constraint
        d/group of each fine-grained site. Useful for creating other features
        which respect constraints, but not a feature in itself.

    Returns
    -------
    If return_ids is True, return a numpy.ndarray with the shape (n_fg_sites,).
    Else: A dict with the following key-value pairs:
        'feats': list of feature mats (n_frames, n_fg_sites, total_n_feats)
            The features do not changed between frames.
        'divs': list of div mats (n_frames, total_n_feats, n_dim).
            These are filled with zeros as the features do not change as a
            function of position.
        'names': None

    NOTE: While the features are not generated lazily (they are not returned via
    generators), the entries for each cg_site are views to the same numpy array,
    so the memory footprint is only that of a single cg_site.
    """

    # get list of groups of fine-grained sites which have to share id features
    # because of constraints
    groups = deepcopy(constraints)
    groups = groups.union(frozenset([x]) for x in range(cmap.n_fg_sites))
    reduced_groups = sorted(reduce_constraint_sets(groups))
    places = []
    if return_ids:
        ids = np.zeros(cmap.n_fg_sites, dtype=np.int32)
        for label, fg_set in enumerate(reduced_groups):
            indices = list(fg_set)
            ids[indices] = label
        return ids
    else:
        for label, fg_set in enumerate(reduced_groups):
            _ = [places.append([fg_ind, label]) for fg_ind in fg_set]

    n_frames = points.shape[0]
    n_fg_sites = cmap.n_fg_sites
    n_cg_sites = cmap.n_cg_sites
    n_types = len(reduced_groups)
    n_dim = cmap.n_dim

    inds = list(zip(*places))
    feats = np.zeros((n_frames, n_fg_sites, n_types), dtype=np.float32)
    feats[:, inds[0], inds[1]] = 1

    divs = np.zeros((n_frames, n_types, n_dim), dtype=np.float32)

    return dict(
        feats=[feats] * n_cg_sites,
        divs=[divs] * n_cg_sites,
        names=None,
    )


def multifeaturize(featurizers):
    """Combines multiple featurization functions into a single featurization function.

    NOTE: Output names are not supported and are set to None.

    Arguments
    ---------
    featurizers (list of callables):
        A list, each member of which must use the following as callable input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                points used for featurization
            config_mapping (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list of feature mats (n_frames, n_fg_sites, n_feats)
            'divs': list of divergence mats (n_frames, n_feats, n_dim)
            'names': None or list of name strings (n_feat)

    Returns
    -------
    Callable which returns the output expected of a featurization function: a
    dictionary with the following key value pairs:
       'feats': list of feature mats (n_frames, n_fg_sites, total_n_feats)
       'divs': list of divergence mats (n_frames, total_n_feats, n_dim)
       'names': None
    This output is created by combining the pertinent arrays along the n_feat
    dimensions.  Input is the same as it was for each individual featurization
    function.
    """

    def composite(*args, **kwargs):
        output = [feat(*args, **kwargs) for feat in featurizers]
        return FeatZipper(content=output)

    return composite


class FeatZipper:
    r"""Lazily combines the output of multiple featurizers.

    NOTE: This function does not combine _featurizers_; it combines their
    output. To combine featurizers see multifeaturize.

    Featurizers output a dict with keys for features ("feats"), divergences
    ("divs"), and names ("names"). This class combines the _output_ of multiple
    featurizers, providing dictionary-like interface for accessing features and
    divergences aggregated across all the results. This class can be used with
    lazy featurizers (those providing generators) and is itself lazy: "feats" and
    "divs" index generators.

    In other words, instances of this class are objects which can be indexed
    using the same keys as featurizer output. Indexing returns generators that
    iterate over the data aggregated from the featurized content provided at
    initialization.

    The information for the produced feature and divergence generators comes
    from shared data present in each instance. The original data are iterated
    over as needed (in the case of lazy features, this implies that memory and
    computation is performed only when needed by the aggregate output) and
    served through the provided generators.

    Note that when new data is generated (when the generators are iterated),
    features and divergences are both generated and cached, even if only one of
    the two is used at that particular moment.
    """

    # these keys provide understanding into the feature dictionary format
    feat_key = "feats"
    div_key = "divs"
    generator_keys = set([feat_key, div_key])
    name_key = "names"

    # joiners has the particular functions for combining arrays from each
    # content member's iteration. They are indexed by the appropriate key.
    # e.g., "feats"'s callable combines iterations from "feats" entries.

    # members cannot be static methods and in this dictionary, so they are
    # lambdas.
    joiners = {
        feat_key: lambda args: np.concatenate(args, axis=2),
        div_key: lambda args: np.concatenate(args, axis=1),
    }

    def __init__(self, content):
        r"""Initialize a FeatZipper from list of content dictionaries.

        Arguments
        ---------
        content (list of dictionaries):
            List of the dictionaries that  we want to aggregate and iterate
            over. Each dictionary should be the output of a featurizer; that is,
            it should have "feats", "divs", and "names" as keys.  "feats" and
            "divs" should index an iterable of numpy arrays and "names" should
            be a list of strings.
        """

        self.reset(content)
        # no current support for names
        self.names = None

    def dictzip(self, dictionary):
        r"""Takes a dictionary of iterables and returns an generator that serves
        a dictionary with a single iteration's contents, similar to zip.

        Arguments
        ---------
        dictionary (dict):
            dictionary with entries for each key in self.generator_keys. Each
            key must index an iterable. Other keys are ignored.

        Yields
        ------
        A dict with keys equal to self.generator_keys. Each key indexes a
        single value, produced by iterating over the corresponding sequences
        in dictionary.

        EXAMPLE: Here we use dummy keys ('a', 'b').
            Input is {'a':[1,2,3],'b':[5,6,7]}.
            Output's first iteration is:
                {'a':1,'b':5}
            The second iteration is
                {'a':2,'b':6}
        """

        # make sure content are iterators
        iter_dictionary = dict(
            [(key, iter(dictionary[key])) for key in self.generator_keys]
        )
        while True:
            to_fill = dict()
            for key in self.generator_keys:
                value = iter_dictionary[key]
                try:
                    to_fill[key] = next(value)
                except StopIteration:
                    return
            yield to_fill

    def keys(self):
        r"""Returns a set of all viable keys for indexing.

        Returns
        -------
        Set of all feasible keys.
        """

        return self.generator_keys.union(set([self.name_key]))

    def reset(self, content):
        r"""Prepares internal state.

        Internal state includes setting up iterators and queues. The queues are
        used to temporarily aggregate the output of content's iterators when
        necessary. The queues are used to aggregate results as we (indirectly)
        iterate over the input iterables in content.

        Arguments
        ---------
        content (list of dicts):
            list of dictionaries, each of which is the output of a
            featurization function.  See __init__'s content argument for more
            details.
        """

        self.iterators = [self.dictzip(x) for x in content]
        queues = [SimpleQueue() for _ in self.generator_keys]
        self._queues = dict(zip(self.generator_keys, queues))

    def _makegenerator(self, key):
        r"""Creates a generator for a specified key.

        These generators query the internal queues for results from the input
        dictionaries. If the queues are empty, they try to repopulate them; if
        that fails, they return.

        Arguments
        ---------
        key (hashable):
            The key used to extract the series from the content dictionaries for
            aggregation.

        Yields
        ------
        Aggregated numpy.ndarrays formed from results under key in content dictionaries.

        EXAMPLE: if "feats" is passed, we return a generator; at each iteration
        this generator grabs an iteration from the iterator under "feats" from
        each content dictionary, combines them into a single array, and returns them.
        """

        while True:
            try:
                item = self._queues[key].get(block=False)
            except Empty:
                try:
                    self._populate(exception=False)
                    item = self._queues[key].get(block=False)
                except Empty:
                    return
            yield item

    def __getitem__(self, key):
        r"""Implements indexing. Returns generator associated with key or (in
        the case of names) None.

        Arguments
        ---------
        key (hashable):
            A member of self.generator_keys or \{self.name_key\}. If
            name_key, then self.names is returned. If a member of
            generator_keys, a generator which iterates over the aggregated
            form of that key's content in self.content. See the class
            description for more details.

        Returns
        -------
        Generator or self.names
        """

        if key in self.generator_keys:
            return self._makegenerator(key)
        if key == self.name_key:
            return self.names
        raise KeyError("Invalid key; valid keys are {}".format(self.keys()))

    def _populate(self, exception=True):
        r"""Adds an item to internal queues by continuing source generators one
        step.

        The internal queues' purposes are to store the output of the input
        feature data for processing. This method is called when other methods
        realize the queues are empty.

        Arguments
        ---------
        exception (boolean):
            If true, StopIteration exceptions are not caught; as a result, if
            the internal iterators are depleted StopIteration will be thrown. If
            false, this situation is caught and the method simply leaves the
            queues untouched and returns.
        """

        if exception:
            outs = [next(x) for x in self.iterators]
        else:
            try:
                outs = [next(x) for x in self.iterators]
            except StopIteration:
                return
        for key in self.generator_keys:
            joiner = self.joiners[key]
            agg = joiner([x[key] for x in outs])
            self._queues[key].put(agg)
        return


def curry(func, *args, **kwargs):
    """Curries a function using named and keyword arguments.

    That is: for f(x,y), curry(f,y=a) returns a function g, where g(b) =
    f(x=b,y=a). Non-keyword arguments also work--- they are passed after any
    non-keyword arguments passed to g.  Useful when creating a featurization
    function with certain options set.

    Arguments
    ---------
    func (callable):
        Function to be curried.
    args/kwargs:
        Used to curry func.

    Returns
    -------
    Callable which evaluates func appending args and kwargs to any passed
    arguments.
    """

    def curried_f(*sub_args, **sub_kwargs):
        return func(*sub_args, *args, **sub_kwargs, **kwargs)

    return curried_f


def flatten(nested_list):
    """Flattens a nested list.

    Arguments
    ---------
    nested_list (list of lists):
        List of the form [[a,b...],[h,g,...],...]

    Returns
    -------
    Returns a list where the items are the subitems of nested_list. For example,
    [[1,2],[3,4] would be transformed into [1,2,3,4].
    """

    return [item for sublist in nested_list for item in sublist]
