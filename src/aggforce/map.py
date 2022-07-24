r"""Provides objects describing maps transforming fine-grained points to
coarse-grained points. Points may be positions or forces.
"""

from abc import ABC, abstractmethod
import numpy as np


class Map(ABC):
    r"""Provides a abstact interface for maps for transforming
    fine-grained to coarse-grained systems.

    This is an abstract class. You must instantiate a subclass to use it. It
    mostly useful to remind what content can be assumed to be present for a
    Map object.

    Note that Map objects act on systems of particles, each of which has n
    dimensions (and not flat vectors).
    """

    n_dim = 3

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def n_cg_sites(self):
        r"""The number of coarse-grained sites present in the output of the
        map.
        """

    @property
    @abstractmethod
    def n_fg_sites(self):
        r"""The number of fine-grained sites expected in the input to the
        map.
        """

    @abstractmethod
    def __call__(self, points, copoints):
        r"""Applies map to a particular form of 3-dim array.

        Arguments
        ---------
        points (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
        copoints (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
            Some mapping methods require an additional set of points. For
            example, positions must sometimes be given in order to apply a
            position-dependent force map.

        Returns
        -------
        Mapped version of points as a three dimensional nd.numpy array.
        """


class LinearMap(Map):
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

    def __call__(self, points, copoints=None):
        r"""Applies map to a particular form of 3-dim array.

        Arguments
        ---------
        points (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
        copoints:
            Ignored. Included for compatibility with parent class.

        Returns
        -------
        Combines points along the n_sites dimension according to the internal
        map.
        """
        shape = points.shape
        reshaped_input = np.reshape(np.swapaxes(points, 0, 1), (shape[1], -1))
        reshaped_output = np.matmul(self.standard_matrix, reshaped_input)
        output = np.swapaxes(
            np.reshape(reshaped_output, (-1, shape[0], shape[2])), 0, 1
        )
        return output
