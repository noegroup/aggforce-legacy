"""Featurization functions which use the deeptime and JAX libraries. Used for
configuration dependent maps.
"""

import numpy as np
import jax
import jax.numpy as jnp
from deeptime.decomposition import TICA
from .jaxfeat import Whiten, _id, gaussian_dist_basis, channel_allocate, id_feat, abatch


def gbcv_feat(
    points,
    cmap,
    constraints,
    cv,
    outer,
    inner,
    n_basis=10,
    width=1.0,
    dist_power=1,
    batch_size=None,
    lazy=True,
    div_method="reorder",
):
    """Featurizes each fine-grained site by using a provided single dimensional
    collective variable callable.

    NOTE: This function uses JAX for acceleration.

    At each frame, the cv argument is used to produce a scalar that
    characterizes the global state of the system.  This global state is "binned"
    by applying a series of Gaussians at multiple points. This binned cvs is
    copied and then associated separately to each atom (this is done using a
    one-hot encoding in the feature matrix).

    In effect, this feature should allow for force maps which are a function
    global state.

    Arguments
    ---------
    points (jnp.DeviceArray):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims).
    cmap (map.LinearMap):
        Configurational map that links the fine-grained and coarse-grained
        resolutions.
    constraints (set of frozensets):
        Set of frozensets, each of which contains a set of fine-grained
        sites which have a molecular constraint applied. Constrained
        groups may overlap.
    cv (callable):
        Callable that produces collective variables. Must apply to a trajectory
        array (n_steps,n_sites,n_dims) and produce output of shape (n_steps,1).
        Must apply to and produce jnp.DeviceArray objects.
    outer (None or positive float):
        The largest cv value to consider when making the grid of Gaussians. If
        None, it is set to the maximum cv value found.
        NOTE: It is likely that None will _not_ do what you want: if applied to a
        train and hold-out set, these two sets may very well have different
        automatic outers calculated.
    inner (None or non-negative float):
        The smallest cv value to consider when making the grid of Gaussians. If
        None, it is set to the minimum cv value found.
        NOTE: It is likely that None will _not_ do what you want: if applied to a
        train and hold-out set, these two sets may very well have different
        automatic inners calculated.
    n_basis (positive integer):
        Number of Gaussian bins to use. Higher numbers are more expressive, but
        increase memory usage.
    width (positive integer):
        Controls the width of each Gaussian. Gaussians are roughly calculated as
        exp(-d**2/width), where d is the cv.
    dist_power (positive float):
        Controls the spacing and scaling of the Gaussians. Values greater than 1
        concentrate Gaussians towards outer, values between 0 and 1 concentrate
        Gaussians towards inner. Areas with more concentrated Gaussians also
        have Gaussians of less variance. See gaussian_dist_basis for more
        information.
    batch_size (positive integer):
        Number of trajectory frames to feed into JAX at once. Larger values are
        faster but use more memory.
    lazy (boolean):
        If truthy, generators of features and divs are returned; else, lists are
        returned.
    div_method (string):
        Determines how the divergence will be calculated; passed to
        gb_subfeat_jac as method.

    Returns
    -------
    A dictionary with two key elements pairs:
        'feats': list/generator of feature mats (n_frames, n_fg_sites, total_n_feats)
        'divs': list/generator of div mats (n_frames, total_n_feats, n_dim).
        'names': None
    Each element of these lists corresponds to features for a single CG site,
    except for 'names', which may have names which are shared for each cg site
    (or None).
    """

    # prep information needed for featurization

    # mapped CG points
    points = jnp.asarray(points)
    ids = tuple(id_feat(points, cmap, constraints, return_ids=True))
    max_channels = max(ids)

    # shared option dict for featurization and div calls
    f_kwargs = dict(
        cv=cv,
        channels=ids,
        max_channels=max_channels,
        inner=inner,
        outer=outer,
        width=width,
        n_basis=n_basis,
        dist_power=dist_power,
    )

    # we use abatch to break down computation. In order to do so, we make
    # wrapped callables that take simpler arguments
    def subfeater(arg_inds):
        sub_points = points[arg_inds]
        feat = gbcv_subfeat(
            points=sub_points,
            **f_kwargs,
        )
        return feat

    # subsetted by abatch in feater and divver to mark where to evaluate
    inds = np.arange(len(points))

    feat = np.asarray(abatch(func=subfeater, arr=inds, chunk_size=batch_size))
    feats = [feat] * cmap.n_cg_sites

    # now do the same for divergences

    # this function takes a set of indices for subsetting, this makes it
    # compatible with abatch
    def subdivver(arg_inds):
        sub_points = points[arg_inds]
        div = gbcv_subfeat_jac(
            points=sub_points,
            method=div_method,
            **f_kwargs,
        )
        return div

    div = np.asarray(abatch(func=subdivver, arr=inds, chunk_size=batch_size))
    divs = [div] * cmap.n_cg_sites

    return dict(
        feats=feats,
        divs=divs,
        names=None,
    )


def gbcv_subfeat(
    points,
    cv,
    channels,
    max_channels,
    collapse=False,
    channelize=True,
    **kwargs,
):
    """Creates features (not divergences) using Gaussian bins and a collective
    variable function.

    Note that the cv callable should output a scalar for every frame; this
    single scalar is then used for every site.

    Arguments
    ---------
    points (jnp.DeviceArray):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims) or (n_fg_sites,n_dims); in the
        latter case, a dummy n_frames index is added during computation.
    cv (callable):
        Callable that returns a time series (n_steps,1) of collective variable
        values when applied to a (n_steps,n_sites,n_dims) input ndarray.
        Must apply to and produce jnp.DeviceArray objects.
    channels (tuple of positive integers):
        Tuple of integers with the length of the number of fine-grained sites
        in the trajectory. Each integer assigns that fine-grained site to a
        constraint group. So, if two atoms have a constrained bond connecting
        them, they should both have the same integer. The integers do not have
        to be consecutive, but max_channels must as big as the largest channel.
    max_channels (positive integer):
        Maximum value of channels. Included as argument due to JAX constraints.
        Larger values increase memory usage, so the most memory efficient
        (max_channels,channels) pair has channels starting at 0 with maximum value
        at max_channels, with no unused index in between.
    collapse (boolean):
        Trace over indices corresponding to frames and fine-grained sites in the
        output. Useful for some later gradient calculations.  If collapse=True
        and points is 2-dimensional, the output may not make sense.
    channelize (boolean):
        Whether to distribute the Gaussian features over one-hot-like channels
        to make them specific to various groups of atoms.
    kwargs:
        Passed to gaussian_dist_basis.

    Returns
    -------
    If collapse, an array of shape (n_features,) is returned; else,
    jnp.DeviceArray of ether shape (n_frames,n_fg_sites,n_features) or
    (n_fg_sites,n_features) is returned, with the latter occurring when points
    only has two dimensions.  n_features is set via kwargs, max_channels,
    and gaussian_dist_basis.
    """

    # if our input has no frame axis, add dummy
    if len(points.shape) == 2:
        points = points[None, ...]
        dummy_axis = True
    else:
        dummy_axis = False

    cvs = jnp.repeat(cv(points), points.shape[1], axis=1)
    if kwargs.get("inner", None) is None:
        kwargs["inner"] = cvs.min()
    if kwargs.get("outer", None) is None:
        kwargs["outer"] = cvs.max()
    gauss = gaussian_dist_basis(cvs, **kwargs)
    if channelize:
        channelized = channel_allocate(gauss, channels, max_channels)
    else:
        channelized = gauss
    if collapse:
        collapsed = channelized.sum(axis=(0, 1))
    else:
        collapsed = channelized
        # if we collapse, then this index removal doesn't make sense
        if dummy_axis:
            return collapsed[0, ...]
    return collapsed


def gbcv_subfeat_jac(
    points,
    cv,
    channels,
    max_channels,
    method="reorder",
    **kwargs,
):
    """Calculates per frame (collapsed) divergences for gbcv_subfeat.

    Most arguments are passed to gbcv_subfeat; see that function for more details.
    However, note that not all the arguments are the same (see, for example, the
    allowed shaped of points and where kwargs goes).

    NOTE: Be sure to pass the same arguments to this and gbcv_subfeat if using
    their results in tandem (even if this function's internal call to gbcv_subfeat
    changes certain arguments).

    Arguments
    ---------
    points (jnp.DeviceArray):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims).
    cv (callable):
        Callable that produces collective variables. Must apply to a trajectory
        array (n_steps,n_sites,n_dims) and produce output of shape (n_steps,1).
        Must apply to and produce jnp.DeviceArray objects.
    channels (tuple of positive integers):
        Tuple of integers with the length of the number of fine-grained sites
        in the trajectory. Each integer assigns that fine-grained site to a
        constraint group. So, if two atoms have a constrained bond connecting
        them, they should both have the same integer. The integers to not have
        to be consecutive, but max_channels must as big as the largest channel.
    max_channels (positive integer):
        Maximum value of channels. Included as argument due to JAX constraints.
        Larger values increase memory usage, so the most memory efficient
        (max_channels,channels) pair has channels starting at 0 with maximum value
        at max_channels, with no unused index in between.
    method (string):
        if method=="basic":
            A direct Jacobian is calculated using a full gb_subfeat call with
            collapse=True.
        elif method=="reorder":
            Jacobian is calculated before one-hot-like vectors are created, and
            then itself one-hotted.
    kwargs:
        Passed to gb_subfeat.

    Returns
    -------
    jnp.DeviceArray of shape (n_frames, n_features, n_dims=3) containing the per
    frame Jacobian values summed over the fine grained particles.
    """

    if method == "basic":
        # collapse=True-> sums features over all atoms and frames to that
        # jacobian calculation avoids trivial zero entries.
        to_jac = lambda x: gbcv_subfeat(
            x,
            cv=cv,
            channels=channels,
            max_channels=max_channels,
            collapse=True,
            **kwargs,
        )
        jac = jax.jacfwd(to_jac)(points)
        # sum over fine-grained sites
        traced_jac = jac.sum(axis=(2,))
        reshaped_jac = jnp.swapaxes(traced_jac, 0, 1)
        return reshaped_jac
    elif method == "reorder":
        to_jac = lambda x: gbcv_subfeat(
            x,
            cv=cv,
            channels=channels,
            max_channels=max_channels,
            collapse=True,
            channelize=False,
            **kwargs,
        )
        # jac is (n_feat, n_frame, n_fg_sites, n_dim)
        jac = jax.jacrev(to_jac)(points)
        # ch_jac is (exp_n_feat, n_frame, n_fg_sites, n_dim)
        ch_jac = channel_allocate(jac, channels, max_channels, jac_shape=True)
        # sum over fine-grained sites
        traced_ch_jac = ch_jac.sum(axis=(2,))
        reshaped_jac = jnp.swapaxes(traced_ch_jac, 0, 1)
        return reshaped_jac
    else:
        raise ValueError("Unknown method for jacobian calculation.")


class JaxTICA:
    """Time independent coordinate analysis with JAX'd fit and transform
    methods.

    This class wraps the TICA class from deeptime, but introduces some
    modifications. Featurization and whitening is handled internally, and only
    then is the result analyzed via TICA. The singular values are extracted from
    the TICA results and then directly applied when transforming new data.

    These modifications allow the fit and transform methods to apply to JAX
    arrays, and allows transform to be differentiated using JAX tools.

    NOTE: Whitening transform is trained on data provided to fit; the resulting
    offset and scaling are help constant for future transform calls.

    NOTE: For any future gradient calculations, the whitening offsets and the
    TICA parameters are treated as static and not a function of the input
    arrays.
    """

    def __init__(self, featurizer=_id, dim=None, **kwargs):
        """Creates JaxTICA object.

        Arguments
        ---------
        featurizer (callable):
            Function to applied to input in array form before subsequent
            whitening and TICA analysis. Note that featurizer is applied
            _before_ any whitening and should be written to accept arrays (if
            list input is provided, it is run across each element).
        dim (positive integer):
            Number of dimensions of TICA output to return. Note that lowering
            this number does not reduce the computational cost in the current
            implementation.
        kwargs:
            Passed to deeptime.decomposition.TICA.
        """

        self._tica = TICA(**kwargs)
        self.is_fit = False
        self.featurizer = featurizer
        self.dim = dim

    def fit(self, data):
        """Fits whitening and TICA transforms on data.

        NOTE: data should be an individual or list of jax.DeviceArray(s).

        Data is first featurized. Featurized data is then used to train the
        whitener; this whitener is then used to whiten the featurized data. The
        resulting output is then fed into the internal TICA engine for fitting.

        Arguments
        ---------
        data (jax.DeviceArray or list of jax.DeviceArray):
            Data used to parameterize the model. May be a list of arrays or an
            individual array of shape (n_frames,n_dim). Note that mdtraj-shaped
            (n_frames,n_sites,n_dims) arrays are not accepted.
        """

        self.is_fit = True
        if not isinstance(data, list):
            data = [data]
        feats = [self.featurizer(x) for x in data]
        self.whiten = Whiten(axis=0).fit(feats)
        white_feats = self.whiten(feats)

        self._tica.fit([np.asarray(x) for x in white_feats])
        model = self._tica.fetch_model()
        self.transform_matrix = jnp.asarray(model.singular_vectors_left)

        return self

    def transform(self, data):
        """Transforms data through the whitened featurized TICA transform.

        NOTE: fit must be called before transform or a ValueError will be
        raised.

        NOTE: data should be an individual or list of jax.DeviceArray(s).

        Arguments
        ---------
        data (jax.DeviceArray or list of jax.DeviceArray):
            Data used to be transformed. May be a list of arrays or an
            individual array of shape (n_frames,n_dim). Note that mdtraj-shaped
            (n_frames,n_sites,n_dims) arrays are not accepted.
        """

        if not self.is_fit:
            raise ValueError("self.fit must be called before self.transform.")
        if isinstance(data, list):
            naked = False
        else:
            naked = True
            data = [data]
        feats = [self.featurizer(x) for x in data]
        white_feats = self.whiten(feats)
        all_coords = [jnp.matmul(x, self.transform_matrix) for x in white_feats]
        if self.dim is None:
            filtered = all_coords
        else:
            filtered = [x[:, 0 : self.dim] for x in all_coords]
        if naked:
            return filtered[0]
        else:
            return filtered

    def __call__(self, data):
        """Transforms data through the whitened featurized TICA transform.

        NOTE: fit must be called before transform or a ValueError will be
        raised.

        NOTE: data should be an individual or list of jax.DeviceArray(s).

        Arguments
        ---------
        data (jax.DeviceArray or list of jax.DeviceArray):
            Data used to be transformed. May be a list of arrays or an
            individual array of shape (n_frames,n_dim). Note that mdtraj-shaped
            (n_frames,n_sites,n_dims) arrays are not accepted.
        """

        return self.transform(data)
