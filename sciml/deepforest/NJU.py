# deepforest/__init__.py

from .cascade import CascadeForestClassifier, CascadeForestRegressor
from .forest import RandomForestClassifier, RandomForestRegressor
from .forest import ExtraTreesClassifier, ExtraTreesRegressor
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .tree import ExtraTreeClassifier, ExtraTreeRegressor


__all__ = [
    "CascadeForestClassifier",
    "CascadeForestRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]


# deepforest/_binner.py
"""
Implementation of the Binner class in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_hist_gradient_boosting/binning.py
"""


__all__ = ["Binner"]

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array

from . import _cutils as _LIB


X_DTYPE = np.float64
X_BINNED_DTYPE = np.uint8
ALMOST_INF = 1e300


def _find_binning_thresholds_per_feature(
    col_data, n_bins, bin_type="percentile"
):
    """
    Private function used to find midpoints for samples along a
    specific feature.
    """
    if len(col_data.shape) != 1:

        msg = (
            "Per-feature data should be of the shape (n_samples,), but"
            " got {}-dims instead."
        )
        raise RuntimeError(msg.format(len(col_data.shape)))

    missing_mask = np.isnan(col_data)
    if missing_mask.any():
        col_data = col_data[~missing_mask]
    col_data = np.ascontiguousarray(col_data, dtype=X_DTYPE)
    distinct_values = np.unique(col_data)
    # Too few distinct values
    if len(distinct_values) <= n_bins:
        midpoints = distinct_values[:-1] + distinct_values[1:]
        midpoints *= 0.5
    else:
        # Equal interval in terms of percentile
        if bin_type == "percentile":
            percentiles = np.linspace(0, 100, num=n_bins + 1)
            percentiles = percentiles[1:-1]
            midpoints = np.percentile(
                col_data, percentiles, interpolation="midpoint"
            ).astype(X_DTYPE)
            assert midpoints.shape[0] == n_bins - 1
            np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
        # Equal interval in terms of value
        elif bin_type == "interval":
            min_value, max_value = np.min(col_data), np.max(col_data)
            intervals = np.linspace(min_value, max_value, num=n_bins + 1)
            midpoints = intervals[1:-1]
            assert midpoints.shape[0] == n_bins - 1
        else:
            raise ValueError("Unknown binning type: {}.".format(bin_type))

    return midpoints


def _find_binning_thresholds(
    X, n_bins, bin_subsample=2e5, bin_type="percentile", random_state=None
):
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    if n_samples > bin_subsample:
        subset = rng.choice(np.arange(n_samples), bin_subsample, replace=False)
        X = X.take(subset, axis=0)

    binning_thresholds = []
    for f_idx in range(n_features):
        threshold = _find_binning_thresholds_per_feature(
            X[:, f_idx], n_bins, bin_type
        )
        binning_thresholds.append(threshold)

    return binning_thresholds


class Binner(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        random_state=None,
    ):
        self.n_bins = n_bins + 1  # + 1 for missing values
        self.bin_subsample = int(bin_subsample)
        self.bin_type = bin_type
        self.random_state = random_state
        self._is_fitted = False

    def _validate_params(self):

        if not 2 <= self.n_bins - 1 <= 255:
            msg = (
                "`n_bins` should be in the range [2, 255], bug got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_bins - 1))

        if not self.bin_subsample > 0:
            msg = (
                "The number of samples used to construct the Binner"
                " should be strictly positive, but got {} instead."
            )
            raise ValueError(msg.format(self.bin_subsample))

        if self.bin_type not in ("percentile", "interval"):
            msg = (
                "The type of binner should be one of {{percentile, interval"
                "}}, bug got {} instead."
            )
            raise ValueError(msg.format(self.bin_type))

    def fit(self, X):

        self._validate_params()

        self.bin_thresholds_ = _find_binning_thresholds(
            X,
            self.n_bins - 1,
            self.bin_subsample,
            self.bin_type,
            self.random_state,
        )

        self.n_bins_non_missing_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32,
        )

        self.missing_values_bin_idx_ = self.n_bins - 1
        self._is_fitted = True

        return self

    def transform(self, X):

        if not self._is_fitted:
            msg = (
                "The binner has not been fitted yet when calling `transform`."
            )
            raise RuntimeError(msg)

        if not X.shape[1] == self.n_bins_non_missing_.shape[0]:
            msg = (
                "The binner was fitted with {} features but {} features got"
                " passed to `transform`."
            )
            raise ValueError(
                msg.format(self.n_bins_non_missing_.shape[0], X.shape[1])
            )

        X = check_array(X, dtype=X_DTYPE, force_all_finite=False)
        X_binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order="F")

        _LIB._map_to_bins(
            X, self.bin_thresholds_, self.missing_values_bin_idx_, X_binned
        )

        return X_binned


# deepforest/_cutils.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport isnan

ctypedef np.npy_bool BOOL
ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float64 X_DTYPE_C
ctypedef np.npy_uint8 X_BINNED_DTYPE_C

np.import_array()


cpdef void _c_merge_proba(np.ndarray[X_DTYPE_C, ndim=2] probas,
                          SIZE_t n_outputs,
                          np.ndarray[X_DTYPE_C, ndim=2] out):
    cdef:
        SIZE_t n_features = probas.shape[1]
        SIZE_t start = 0
        SIZE_t count = 0

    while start < n_features:
        out += probas[:, start : (start + n_outputs)]
        start += n_outputs
        count += 1

    out /= count


cpdef np.ndarray _c_sample_mask(const INT32_t [:] indices,
                                int n_samples):
    """
    Generate the sample mask given indices without resorting to `np.unique`."""
    cdef:
        SIZE_t i
        SIZE_t n = indices.shape[0]
        SIZE_t sample_id
        np.ndarray[BOOL, ndim=1] sample_mask = np.zeros((n_samples,),
                                                        dtype=bool)

    with nogil:
        for i in range(n):
            sample_id = indices[i]
            if not sample_mask[sample_id]:
                sample_mask[sample_id] = True

    return sample_mask


# Modified from HGBDT in Scikit-Learn
cpdef _map_to_bins(object X,
                   list binning_thresholds,
                   const unsigned char missing_values_bin_idx,
                   X_BINNED_DTYPE_C [::1, :] binned):
    """Bin numerical values to discrete integer-coded levels.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        The numerical data to bin.
    binning_thresholds : list of arrays
        For each feature, stores the increasing numeric values that are
        used to separate the bins.
    binned : ndarray, shape (n_samples, n_features)
        Output array, must be fortran aligned.
    """
    cdef:
        const X_DTYPE_C[:, :] X_ndarray = X
        SIZE_t n_features = X.shape[1]
        SIZE_t feature_idx

    for feature_idx in range(n_features):
        _map_num_col_to_bins(X_ndarray[:, feature_idx],
                             binning_thresholds[feature_idx],
                             missing_values_bin_idx,
                             binned[:, feature_idx])


cdef void _map_num_col_to_bins(const X_DTYPE_C [:] data,
                               const X_DTYPE_C [:] binning_thresholds,
                               const unsigned char missing_values_bin_idx,
                               X_BINNED_DTYPE_C [:] binned):
    """Binary search to find the bin index for each value in the data."""
    cdef:
        SIZE_t i
        SIZE_t left
        SIZE_t right
        SIZE_t middle

    for i in range(data.shape[0]):

        if isnan(data[i]):
            binned[i] = missing_values_bin_idx
        else:
            # for known values, use binary search
            left, right = 0, binning_thresholds.shape[0]
            while left < right:
                middle = (right + left - 1) // 2
                if data[i] <= binning_thresholds[middle]:
                    right = middle
                else:
                    left = middle + 1
            binned[i] = left


# deepforest/_estimator.py
"""A wrapper on the base estimator for the naming consistency."""


__all__ = ["Estimator"]

import numpy as np
from .forest import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier as sklearn_RandomForestClassifier,
    ExtraTreesClassifier as sklearn_ExtraTreesClassifier,
    RandomForestRegressor as sklearn_RandomForestRegressor,
    ExtraTreesRegressor as sklearn_ExtraTreesRegressor,
)


def make_classifier_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="custom",
    n_jobs=None,
    random_state=None,
):
    # RandomForestClassifier
    if name == "rf":
        if backend == "custom":
            estimator = RandomForestClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_RandomForestClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    # ExtraTreesClassifier
    elif name == "erf":
        if backend == "custom":
            estimator = ExtraTreesClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_ExtraTreesClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


def make_regressor_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="custom",
    n_jobs=None,
    random_state=None,
):
    # RandomForestRegressor
    if name == "rf":
        if backend == "custom":
            estimator = RandomForestRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_RandomForestRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    # ExtraTreesRegressor
    elif name == "erf":
        if backend == "custom":
            estimator = ExtraTreesRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_ExtraTreesRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):
    def __init__(
        self,
        name,
        criterion,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        n_jobs=None,
        random_state=None,
        is_classifier=True,
    ):

        self.backend = backend
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.estimator_ = make_classifier_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )
        else:
            self.estimator_ = make_regressor_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )

    @property
    def oob_decision_function_(self):
        # Scikit-Learn uses `oob_prediction_` for ForestRegressor
        if self.backend == "sklearn" and not self.is_classifier:
            oob_prediction = self.estimator_.oob_prediction_
            if len(oob_prediction.shape) == 1:
                oob_prediction = np.expand_dims(oob_prediction, 1)
            return oob_prediction
        return self.estimator_.oob_decision_function_

    @property
    def feature_importances_(self):
        """Return the impurity-based feature importances from the estimator."""

        return self.estimator_.feature_importances_

    def fit_transform(self, X, y, sample_weight=None):
        self.estimator_.fit(X, y, sample_weight)
        return self.oob_decision_function_

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict(X)

    def predict(self, X):
        if self.is_classifier:
            return self.estimator_.predict_proba(X)
        pred = self.estimator_.predict(X)
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, 1)
        return pred

# deepforest/_forest.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from numpy import uint8 as DTYPE
from numpy import float64 as DOUBLE

from .tree._tree cimport DTYPE_t
from .tree._tree cimport DOUBLE_t
from .tree._tree cimport SIZE_t

cdef SIZE_t _TREE_LEAF = -1


cpdef np.ndarray predict(object data,
                         const SIZE_t [:] feature,
                         const DTYPE_t [:] threshold,
                         const SIZE_t [:, ::1] children,
                         np.ndarray[DOUBLE_t, ndim=2] value):
    """Predict the class distributions or values for samples in ``data``.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The numerical data to predict.
    feature : ndarray of shape (n_internal_nodes,)
        Stores the splitting feature for all internal nodes in the forest.
    threshold : ndarray of shape (n_internal_nodes,)
        Store the splitting threshold for all internal nodes in the forest.
    children : ndarray of shape (n_internal_nodes, 2)
        Store the IDs of left and right child for all internal nodes in the
        forest. Negative values indicate that the corresponding node is a
        leaf node.
    value : ndarray of shape (n_leaf_nodes, n_outputs)
        Store the prediction for all leaf nodes in the forest. The layout of
        ``children`` should be C-aligned. It is declared as ``np.ndarray``
        instead f typed memoryview to support splicing.

    Returns
    -------
    out : ndarray of shape (n_samples, n_outputs)
        The predicted class probabilities or values.
    """
    cdef:
        SIZE_t n_samples = data.shape[0]
        SIZE_t n_outputs = value.shape[1]
        SIZE_t n_indices
        np.ndarray[SIZE_t, ndim=1] indice = np.empty((n_samples,),
                                                     dtype=np.int32)
        np.ndarray[DOUBLE_t, ndim=2] out = np.zeros((n_samples, n_outputs),
                                                    dtype=DOUBLE)

    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)

    _apply_region(data, feature, threshold, children, indice)
    out += value.take(indice, axis=0, mode='clip')

    return out


cdef void _apply_region(const DTYPE_t [:, :] data,
                        const SIZE_t [:] feature,
                        const DTYPE_t [:] threshold,
                        const SIZE_t [:, ::1] children,
                        SIZE_t [:] out):
    """
    Find the terminal region (i.e., leaf node ID) for each sample in ``data``.
    """
    cdef:
        SIZE_t n_samples = data.shape[0]
        SIZE_t n_internal_nodes = feature.shape[0]
        SIZE_t i
        SIZE_t node_id
        SIZE_t node_feature
        DTYPE_t node_threshold
        SIZE_t left_child
        SIZE_t right_child

    with nogil:
        for i in range(n_samples):

            # Skip the corner case where the root node is a leaf node
            if n_internal_nodes == 0:
                out[i] = 0
                continue

            node_id = 0
            node_feature = feature[node_id]
            node_threshold = threshold[node_id]
            left_child = children[node_id, 0]
            right_child = children[node_id, 1]

            # While one of the two child of the current node is not a leaf node
            while left_child > 0 or right_child > 0:

                # If the left child is a leaf node
                if left_child <= 0:

                    # If X[sample_id] should be assigned to the left child
                    if data[i, node_feature] <= node_threshold:
                        out[i] = <SIZE_t>(_TREE_LEAF * left_child)
                        break
                    else:
                        node_id = right_child
                        node_feature = feature[node_id]
                        node_threshold = threshold[node_id]
                        left_child = children[node_id, 0]
                        right_child = children[node_id, 1]

                # If the right child is a leaf node
                elif right_child <= 0:

                    # If X[sample_id] should be assigned to the right child
                    if data[i, node_feature] > node_threshold:
                        out[i] = <SIZE_t>(_TREE_LEAF * right_child)
                        break
                    else:
                        node_id = left_child
                        node_feature = feature[node_id]
                        node_threshold = threshold[node_id]
                        left_child = children[node_id, 0]
                        right_child = children[node_id, 1]

                # If the left and right child are both internal nodes
                else:
                    if data[i, node_feature] <= node_threshold:
                        node_id = left_child
                    else:
                        node_id = right_child

                    node_feature = feature[node_id]
                    node_threshold = threshold[node_id]
                    left_child = children[node_id, 0]
                    right_child = children[node_id, 1]

            # If the left and child child are both leaf nodes
            if left_child <= 0 and right_child <= 0:
                if data[i, node_feature] <= node_threshold:
                    out[i] = <SIZE_t>(_TREE_LEAF * left_child)
                else:
                    out[i] = <SIZE_t>(_TREE_LEAF * right_child)

# deepforest/_io.py
"""
Implement methods on dumping and loading large objects using joblib. This
class is designed to support the partial mode in deep forest.
"""


__all__ = ["Buffer"]

import os
import shutil
import warnings
import tempfile
from joblib import load, dump


class Buffer(object):
    """
    The class of dumping and loading large array objects including the data
    and estimators.

    Parameters
    ----------
    partial_mode : bool

        - If ``True``, a temporary buffer on the local disk is created to
          cache objects such as data and estimators.
        - If ``False``, all objects are directly stored in memory without
          extra processing.
    store_est : bool, default=True
        Whether to cache the estimators to the local buffer.
    store_pred : bool, default=True
        Whether to cache the predictor to the local buffer.
    store_data : bool, default=False
        Whether to cache the intermediate data to the local buffer.
    """

    def __init__(
        self,
        use_buffer,
        buffer_dir=None,
        store_est=True,
        store_pred=True,
        store_data=False,
    ):

        self.use_buffer = use_buffer
        self.store_est = store_est and use_buffer
        self.store_pred = store_pred and use_buffer
        self.store_data = store_data and use_buffer
        self.buffer_dir = os.getcwd() if buffer_dir is None else buffer_dir

        # Create buffer
        if self.use_buffer:
            self.buffer = tempfile.TemporaryDirectory(
                prefix="buffer_", dir=self.buffer_dir
            )

            if store_data:
                self.data_dir_ = tempfile.mkdtemp(
                    prefix="data_", dir=self.buffer.name
                )

            if store_est or store_pred:
                self.model_dir_ = tempfile.mkdtemp(
                    prefix="model_", dir=self.buffer.name
                )
                self.pred_dir_ = os.path.join(self.model_dir_, "predictor.est")

    @property
    def name(self):
        """Return the buffer name."""
        if self.use_buffer:
            return self.buffer.name
        else:
            return None

    def cache_data(self, layer_idx, X, is_training_data=True):
        """
        When ``X`` is a large array, it is not recommended to directly pass the
        array to all processors because the array will be copied multiple
        times and cause extra overheads. Instead, dumping the array to the
        local buffer and reading it as the ``numpy.memmap`` mode across
        processors is able to speed up the training and evaluating process.

        Parameters
        ----------
        layer_idx : int
            The index of the cascade layer that utilizes ``X``.
        X : ndarray of shape (n_samples, n_features)
            The training / testing data to be cached.
        is_training_data : bool, default=True
            Whether ``X`` is the training data.

        Returns
        -------
        X: {ndarray, ndarray in numpy.memmap mode}

            - If ``self.store_data`` is ``True``, return the memory-mapped
            object of `X` cached to the local buffer.
            - If ``self.store_data`` is ``False``, return the original ``X``.
        """
        if not self.store_data:
            return X

        if is_training_data:
            cache_dir = os.path.join(
                self.data_dir_, "joblib_train_{}.mmap".format(layer_idx)
            )
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)
        else:
            cache_dir = os.path.join(
                self.data_dir_, "joblib_test_{}.mmap".format(layer_idx)
            )
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)

        # Dump and reload data in the numpy.memmap mode
        dump(X, cache_dir)
        X_mmap = load(cache_dir, mmap_mode="r+")

        return X_mmap

    def cache_estimator(self, layer_idx, est_idx, est_name, est):
        """
        Dumping the fitted estimator to the buffer is highly recommended,
        especially when the python version is below 3.8. When the size of
        estimator is large, for instance, several gigabytes in the memory,
        sending it back from each processor will cause the struct error.

        Reference:
            https://bugs.python.org/issue17560

        Parameters
        ----------
        layer_idx : int
            The index of the cascade layer that contains the estimator to be
            cached.
        est_idx : int
            The index of the estimator in the cascade layer to be cached.
        est_name : {"rf", "erf", "custom"}
            The name of the estimator to be cached.
        est : object
            The object of base estimator.

        Returns
        -------
        cache_dir : {string, object}

            - If ``self.store_est`` is ``True``, return the absolute path to
              the location of the cached estimator.
            - If ``self.store_est`` is ``False``, return the estimator.
        """
        if not self.store_est:
            return est

        filename = "{}-{}-{}.est".format(layer_idx, est_idx, est_name)
        cache_dir = os.path.join(self.model_dir_, filename)
        dump(est, cache_dir)

        return cache_dir

    def cache_predictor(self, predictor):
        """
        Please refer to `cache_estimator`.

        Parameters
        ----------
        predictor : object
            The object of the predictor.

        Returns
        -------
        pred_dir : {string, object}

            - If ``self.store_pred`` is ``True``, return the absolute path to
              the location of the cached predictor.
            - If ``self.store_pred`` is ``False``, return the predictor.
        """
        if not self.store_pred:
            return predictor

        dump(predictor, self.pred_dir_)

        return self.pred_dir_

    def load_estimator(self, estimator_path):
        if not os.path.exists(estimator_path):
            msg = "Missing estimator in the path: {}."
            raise FileNotFoundError(msg.format(estimator_path))

        estimator = load(estimator_path)

        return estimator

    def load_predictor(self, predictor):
        # Since this function is always called from `cascade.py`, the input
        # `predictor` could be the actual predictor object. If so, this
        # function will directly return the predictor.
        if not isinstance(predictor, str):
            return predictor

        if not os.path.exists(predictor):
            msg = "Missing predictor in the path: {}."
            raise FileNotFoundError(msg.format(predictor))

        predictor = load(predictor)

        return predictor

    def del_estimator(self, layer_idx):
        """Used for the early stopping stage in deep forest."""
        for est_name in os.listdir(self.model_dir_):
            if est_name.startswith(str(layer_idx)):
                try:
                    os.unlink(os.path.join(self.model_dir_, est_name))
                except OSError:
                    msg = (
                        "Permission denied when deleting the dumped"
                        " estimators during the early stopping stage."
                    )
                    warnings.warn(msg, RuntimeWarning)

    def close(self):
        """Clean up the buffer."""
        try:
            self.buffer.cleanup()
        except OSError:
            msg = "Permission denied when cleaning up the local buffer."
            warnings.warn(msg, RuntimeWarning)


def model_mkdir(dirname):
    """Make the directory for saving the model."""
    if os.path.isdir(dirname):
        msg = "The directory to be created already exists {}."
        raise RuntimeError(msg.format(dirname))

    os.mkdir(dirname)
    os.mkdir(os.path.join(dirname, "estimator"))


def model_saveobj(dirname, obj_type, obj, partial_mode=False):
    """Save objects of the deep forest according to the specified type."""

    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}. Please create it first."
        raise RuntimeError(msg.format(dirname))

    if obj_type in ("param", "binner"):
        if not isinstance(obj, dict):
            msg = "{} to be saved should be in the form of dict."
            raise RuntimeError(msg.format(obj_type))
        dump(obj, os.path.join(dirname, "{}.pkl".format(obj_type)))

    elif obj_type == "layer":
        if not isinstance(obj, dict):
            msg = "The layer to be saved should be in the form of dict."
            raise RuntimeError(msg)

        est_path = os.path.join(dirname, "estimator")
        if not os.path.isdir(est_path):
            msg = "Cannot find the target directory: {}."
            raise RuntimeError(msg.format(est_path))

        # If `partial_mode` is True, each base estimator in the model is the
        # path to the dumped estimator, and we only need to move it to the
        # target directory.
        if partial_mode:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    shutil.move(estimator, dest)
        # Otherwise, we directly use `joblib.dump` to save the estimator to
        # the target directory.
        else:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    dump(estimator, dest)
    elif obj_type == "predictor":
        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        # Same as `layer`
        if partial_mode:
            shutil.move(obj, pred_path)
        else:
            dump(obj, pred_path)
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))


def model_loadobj(dirname, obj_type, d=None):
    """Load objects of the deep forest from the given directory."""

    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}."
        raise RuntimeError(msg.format(dirname))

    if obj_type in ("param", "binner"):
        obj = load(os.path.join(dirname, "{}.pkl".format(obj_type)))
        return obj
    elif obj_type == "layer":
        from ._layer import (
            ClassificationCascadeLayer,
            RegressionCascadeLayer,
            CustomCascadeLayer,
        )

        if not isinstance(d, dict):
            msg = "Loading layers requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        n_estimators = d["n_estimators"]
        n_layers = d["n_layers"]
        layers = {}

        for layer_idx in range(n_layers):

            if not d["use_custom_estimator"]:
                if d["is_classifier"]:
                    layer_ = ClassificationCascadeLayer(
                        layer_idx=layer_idx,
                        n_outputs=d["n_outputs"],
                        criterion=d["criterion"],
                        n_estimators=d["n_estimators"],
                        partial_mode=d["partial_mode"],
                        buffer=d["buffer"],
                        verbose=d["verbose"],
                    )
                else:
                    layer_ = RegressionCascadeLayer(
                        layer_idx=layer_idx,
                        n_outputs=d["n_outputs"],
                        criterion=d["criterion"],
                        n_estimators=d["n_estimators"],
                        partial_mode=d["partial_mode"],
                        buffer=d["buffer"],
                        verbose=d["verbose"],
                    )

                for est_type in ("rf", "erf"):
                    for est_idx in range(n_estimators):
                        est_key = "{}-{}-{}".format(
                            layer_idx, est_idx, est_type
                        )
                        dest = os.path.join(
                            dirname, "estimator", est_key + ".est"
                        )

                        if not os.path.isfile(dest):
                            msg = "Missing estimator in the path: {}."
                            raise RuntimeError(msg.format(dest))

                        if d["partial_mode"]:
                            layer_.estimators_.update(
                                {est_key: os.path.abspath(dest)}
                            )
                        else:
                            est = load(dest)
                            layer_.estimators_.update({est_key: est})
            else:

                layer_ = CustomCascadeLayer(
                    layer_idx=layer_idx,
                    n_splits=1,  # will not be used
                    n_outputs=d["n_outputs"],
                    estimators=[None] * n_estimators,  # will not be used
                    partial_mode=d["partial_mode"],
                    buffer=d["buffer"],
                    verbose=d["verbose"],
                )

                for est_idx in range(n_estimators):
                    est_key = "{}-{}-custom".format(layer_idx, est_idx)
                    dest = os.path.join(dirname, "estimator", est_key + ".est")

                    if not os.path.isfile(dest):
                        msg = "Missing estimator in the path: {}."
                        raise RuntimeError(msg.format(dest))

                    if d["partial_mode"]:
                        layer_.estimators_.update({est_key: dest})
                    else:
                        est = load(dest)
                        layer_.estimators_.update({est_key: est})

            layer_key = "layer_{}".format(layer_idx)
            layers.update({layer_key: layer_})
        return layers
    elif obj_type == "predictor":

        if not isinstance(d, dict):
            msg = "Loading the predictor requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        if not os.path.isfile(pred_path):
            msg = "Missing predictor in the path: {}."
            raise RuntimeError(msg.format(pred_path))

        if d["partial_mode"]:
            return os.path.abspath(pred_path)
        else:
            predictor = load(pred_path)
            return predictor
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))


# deepforest/_layer.py
"""Implementation of the cascade layer in deep forest."""


__all__ = [
    "BaseCascadeLayer",
    "ClassificationCascadeLayer",
    "RegressionCascadeLayer",
    "CustomCascadeLayer",
]

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from . import _utils
from ._estimator import Estimator
from .utils.kfoldwrapper import KFoldWrapper


def _build_estimator(
    X,
    y,
    layer_idx,
    estimator_idx,
    estimator_name,
    estimator,
    oob_decision_function,
    partial_mode=True,
    buffer=None,
    verbose=1,
    sample_weight=None,
):
    """Private function used to fit a single estimator."""
    if verbose > 1:
        msg = "{} - Fitting estimator = {:<5} in layer = {}"
        key = estimator_name + "_" + str(estimator_idx)
        print(msg.format(_utils.ctime(), key, layer_idx))

    X_aug_train = estimator.fit_transform(X, y, sample_weight)
    oob_decision_function += estimator.oob_decision_function_

    if partial_mode:
        # Cache the fitted estimator in out-of-core mode
        buffer_path = buffer.cache_estimator(
            layer_idx, estimator_idx, estimator_name, estimator
        )
        return X_aug_train, buffer_path
    else:
        return X_aug_train, estimator


class BaseCascadeLayer(BaseEstimator):
    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        self.layer_idx = layer_idx
        self.n_outputs = n_outputs
        self.criterion = criterion
        self.n_estimators = n_estimators * 2  # internal conversion
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.backend = backend
        self.partial_mode = partial_mode
        self.buffer = buffer
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        # Internal container
        self.estimators_ = {}

    @property
    def n_trees_(self):
        return self.n_estimators * self.n_trees

    @property
    def feature_importances_(self):
        feature_importances_ = np.zeros((self.n_features,))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            # Partial mode
            if isinstance(estimator, str):
                estimator_ = self.buffer.load_estimator(estimator)
                feature_importances_ += estimator_.feature_importances_
            # In-memory mode
            else:
                feature_importances_ += estimator.feature_importances_

        return feature_importances_ / len(self.estimators_)

    def _make_estimator(self, estimator_idx, estimator_name):
        """Make and configure a copy of the estimator."""
        # Set the non-overlapped random state
        if self.random_state is not None:
            random_state = (
                self.random_state + 10 * estimator_idx + 100 * self.layer_idx
            )
        else:
            random_state = None

        estimator = Estimator(
            name=estimator_name,
            criterion=self.criterion,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            backend=self.backend,
            n_jobs=self.n_jobs,
            random_state=random_state,
            is_classifier=is_classifier(self),
        )

        return estimator

    def _validate_params(self):

        if not self.n_estimators > 0:
            msg = "`n_estimators` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_estimators))

        if not self.n_trees > 0:
            msg = "`n_trees` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_trees))

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict_full(X)

    def predict_full(self, X):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_outputs * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_outputs * idx, self.n_outputs * (idx + 1)
            pred[:, left:right] += estimator.predict(X)

        return pred


class ClassificationCascadeLayer(BaseCascadeLayer, ClassifierMixin):
    """Implementation of the cascade forest layer for classification."""

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit_transform(self, X, y, sample_weight=None):

        self._validate_params()
        n_samples, self.n_features = X.shape

        X_aug = []
        oob_decision_function = np.zeros((n_samples, self.n_outputs))

        # A random forest and an extremely random forest will be fitted
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "rf",
                self._make_estimator(estimator_idx, "rf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "rf")
            self.estimators_.update({key: _estimator})

        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "erf",
                self._make_estimator(estimator_idx, "erf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "erf")
            self.estimators_.update({key: _estimator})

        # Set the OOB estimations and validation accuracy
        self.oob_decision_function_ = oob_decision_function / self.n_estimators
        y_pred = np.argmax(oob_decision_function, axis=1)
        self.val_performance_ = accuracy_score(
            y, y_pred, sample_weight=sample_weight
        )

        X_aug = np.hstack(X_aug)
        return X_aug


class RegressionCascadeLayer(BaseCascadeLayer, RegressorMixin):
    """Implementation of the cascade forest layer for regression."""

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit_transform(self, X, y, sample_weight=None):

        self._validate_params()
        n_samples, self.n_features = X.shape

        X_aug = []
        oob_decision_function = np.zeros((n_samples, self.n_outputs))

        # A random forest and an extremely random forest will be fitted
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "rf",
                self._make_estimator(estimator_idx, "rf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "rf")
            self.estimators_.update({key: _estimator})

        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "erf",
                self._make_estimator(estimator_idx, "erf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "erf")
            self.estimators_.update({key: _estimator})

        # Set the OOB estimations and validation mean squared error
        self.oob_decision_function_ = oob_decision_function / self.n_estimators
        y_pred = self.oob_decision_function_
        self.val_performance_ = mean_squared_error(
            y, y_pred, sample_weight=sample_weight
        )

        X_aug = np.hstack(X_aug)
        return X_aug


class CustomCascadeLayer(object):
    """Implementation of the cascade layer for customized base estimators."""

    def __init__(
        self,
        layer_idx,
        n_splits,
        n_outputs,
        estimators,
        partial_mode=False,
        buffer=None,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):
        self.layer_idx = layer_idx
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.n_estimators = len(estimators)
        self.dummy_estimators_ = estimators
        self.partial_mode = partial_mode
        self.buffer = buffer
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = {}

    def fit_transform(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        X_aug = []

        # Parameters were already validated by upstream methods
        for estimator_idx, estimator in enumerate(self.dummy_estimators_):
            kfold_estimator = KFoldWrapper(
                estimator,
                self.n_splits,
                self.n_outputs,
                self.random_state,
                self.verbose,
                self.is_classifier,
            )

            if self.verbose > 1:
                msg = "{} - Fitting estimator = custom_{} in layer = {}"
                print(
                    msg.format(_utils.ctime(), estimator_idx, self.layer_idx)
                )

            kfold_estimator.fit_transform(X, y, sample_weight)
            X_aug.append(kfold_estimator.oob_decision_function_)
            key = "{}-{}-custom".format(self.layer_idx, estimator_idx)

            if self.partial_mode:
                # Cache the fitted estimator in out-of-core mode
                buffer_path = self.buffer.cache_estimator(
                    self.layer_idx, estimator_idx, "custom", kfold_estimator
                )
                self.estimators_.update({key: buffer_path})
            else:
                self.estimators_.update({key: kfold_estimator})

        # Set the OOB estimations and validation performance
        oob_decision_function = np.zeros_like(X_aug[0])
        for estimator_oob_decision_function in X_aug:
            oob_decision_function += (
                estimator_oob_decision_function / self.n_estimators
            )

        if self.is_classifier:  # classification
            y_pred = np.argmax(oob_decision_function, axis=1)
            self.val_performance_ = accuracy_score(
                y, y_pred, sample_weight=sample_weight
            )
        else:  # regression
            y_pred = oob_decision_function
            self.val_performance_ = mean_squared_error(
                y, y_pred, sample_weight=sample_weight
            )

        X_aug = np.hstack(X_aug)
        return X_aug

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict_full(X)

    def predict_full(self, X):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_outputs * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = custom_{} in layer = {}"
                print(msg.format(_utils.ctime(), idx, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_outputs * idx, self.n_outputs * (idx + 1)
            pred[:, left:right] += estimator.predict(X)

        return pred

# deepforest/_utils.py
"""Implement utilities used in deep forest."""


import numpy as np
from datetime import datetime

from . import _cutils as _LIB


def merge_proba(probas, n_outputs):
    """
    Merge an array that stores multiple class distributions from all estimators
    in a cascade layer into a final class distribution."""

    n_samples, n_features = probas.shape

    if n_features % n_outputs != 0:
        msg = "The dimension of probas = {} does not match n_outputs = {}."
        raise RuntimeError(msg.format(n_features, n_outputs))

    proba = np.zeros((n_samples, n_outputs))
    _LIB._c_merge_proba(probas, n_outputs, proba)

    return proba


def init_array(X, n_aug_features):
    """
    Initialize a array that stores the intermediate data used for training
    or evaluating the model."""
    if X.dtype != np.uint8:
        msg = "The input `X` when creating the array should be binned."
        raise ValueError(msg)

    # Create the global array that stores both X and X_aug
    n_samples, n_features = X.shape
    n_dims = n_features + n_aug_features
    X_middle = np.zeros((n_samples, n_dims), dtype=np.uint8)
    X_middle[:, :n_features] += X

    return X_middle


def merge_array(X_middle, X_aug, n_features):
    """
    Update the array created by `init_array`  with additional checks on the
    layout."""

    if X_aug.dtype != np.uint8:
        msg = "The input `X_aug` when merging the array should be binned."
        raise ValueError(msg)

    assert X_middle.shape[0] == X_aug.shape[0]  # check n_samples
    assert X_middle.shape[1] == n_features + X_aug.shape[1]  # check n_features
    X_middle[:, n_features:] = X_aug

    return X_middle


def ctime():
    """A formatter on current time used for printing running status."""
    ctime = "[" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    return ctime

# deepforest/cascade.py
"""Implementation of Deep Forest."""


__all__ = ["CascadeForestClassifier", "CascadeForestRegressor"]

import numbers
import time
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    is_classifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import type_of_target

from . import _io, _utils
from ._binner import Binner
from ._layer import (
    ClassificationCascadeLayer,
    CustomCascadeLayer,
    RegressionCascadeLayer,
)


def _get_predictor_kwargs(predictor_kwargs, **kwargs) -> dict:
    """Overwrites default args if predictor_kwargs is supplied."""
    for key, value in kwargs.items():
        if key not in predictor_kwargs.keys():
            predictor_kwargs[key] = value
    return predictor_kwargs


def _build_classifier_predictor(
    predictor_name,
    criterion,
    n_estimators,
    n_outputs,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
    predictor_kwargs={},
):
    """Build the predictor concatenated to the deep forest."""
    predictor_name = predictor_name.lower()

    # Random Forest
    if predictor_name == "forest":
        from .forest import RandomForestClassifier

        predictor = RandomForestClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # XGBoost
    elif predictor_name == "xgboost":
        try:
            xgb = __import__("xgboost.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module XGBoost when building the predictor."
                " Please make sure that XGBoost is installed."
            )
            raise ModuleNotFoundError(msg)

        # The argument `tree_method` is always set as `hist` for XGBoost,
        # because the exact mode of XGBoost is too slow.
        objective = "multi:softmax" if n_outputs > 2 else "binary:logistic"
        predictor = xgb.sklearn.XGBClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # LightGBM
    elif predictor_name == "lightgbm":
        try:
            lgb = __import__("lightgbm.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module LightGBM when building the predictor."
                " Please make sure that LightGBM is installed."
            )
            raise ModuleNotFoundError(msg)

        objective = "multiclass" if n_outputs > 2 else "binary"
        predictor = lgb.LGBMClassifier(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    else:
        msg = (
            "The name of the predictor should be one of {{forest, xgboost,"
            " lightgbm}}, but got {} instead."
        )
        raise NotImplementedError(msg.format(predictor_name))

    return predictor


def _build_regressor_predictor(
    predictor_name,
    criterion,
    n_estimators,
    n_outputs,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
    predictor_kwargs={},
):
    """Build the predictor concatenated to the deep forest."""
    predictor_name = predictor_name.lower()

    # Random Forest
    if predictor_name == "forest":
        from .forest import RandomForestRegressor

        predictor = RandomForestRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # XGBoost
    elif predictor_name == "xgboost":
        try:
            xgb = __import__("xgboost.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module XGBoost when building the predictor."
                " Please make sure that XGBoost is installed."
            )
            raise ModuleNotFoundError(msg)

        # The argument `tree_method` is always set as `hist` for XGBoost,
        # because the exact mode of XGBoost is too slow.
        objective = "reg:squarederror"
        predictor = xgb.sklearn.XGBRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    # LightGBM
    elif predictor_name == "lightgbm":
        try:
            lgb = __import__("lightgbm.sklearn")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module LightGBM when building the predictor."
                " Please make sure that LightGBM is installed."
            )
            raise ModuleNotFoundError(msg)

        objective = "regression"
        predictor = lgb.LGBMRegressor(
            **_get_predictor_kwargs(
                predictor_kwargs,
                objective=objective,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    else:
        msg = (
            "The name of the predictor should be one of {{forest, xgboost,"
            " lightgbm}}, but got {} instead."
        )
        raise NotImplementedError(msg.format(predictor_name))

    return predictor


__classifier_model_doc = """
    Parameters
    ----------
    n_bins : :obj:`int`, default=255
        The number of bins used for non-missing values. In addition to the
        ``n_bins`` bins, one more bin is reserved for missing values. Its
        value must be no smaller than 2 and no greater than 255.
    bin_subsample : :obj:`int`, default=200,000
        The number of samples used to construct feature discrete bins. If
        the size of training set is smaller than ``bin_subsample``, then all
        training samples will be used.
    bin_type : :obj:`{"percentile", "interval"}`, default= :obj:`"percentile"`
        The type of binner used to bin feature values into integer-valued bins.

        - If ``"percentile"``, each bin will have approximately the same
          number of distinct feature values.
        - If ``"interval"``, each bin will have approximately the same size.
    max_layers : :obj:`int`, default=20
        The maximum number of cascade layers in the deep forest. Notice that
        the actual number of layers can be smaller than ``max_layers`` because
        of the internal early stopping stage.
    criterion : :obj:`{"gini", "entropy"}`, default= :obj:`"gini"`
        The function to measure the quality of a split. Supported criteria 
        are ``gini`` for the Gini impurity and ``entropy`` for the information 
        gain. Note: this parameter is tree-specific.
    n_estimators : :obj:`int`, default=2
        The number of estimator in each cascade layer. It will be multiplied
        by 2 internally because each estimator contains a
        :class:`RandomForestClassifier` and a :class:`ExtraTreesClassifier`,
        respectively.
    n_trees : :obj:`int`, default=100
        The number of trees in each estimator.
    max_depth : :obj:`int`, default=None
        The maximum depth of each tree. ``None`` indicates no constraint.
    min_samples_split : :obj:`int`, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : :obj:`int`, default=1
        The minimum number of samples required to be at a leaf node.
    use_predictor : :obj:`bool`, default=False
        Whether to build the predictor concatenated to the deep forest. Using
        the predictor may improve the performance of deep forest.
    predictor : :obj:`{"forest", "xgboost", "lightgbm"}`, default= :obj:`"forest"`
        The type of the predictor concatenated to the deep forest. If
        ``use_predictor`` is False, this parameter will have no effect.
    predictor_kwargs : :obj:`dict`, default={}
        The configuration of the predictor concatenated to the deep forest.
        Specifying this will extend/overwrite the original parameters inherit
        from deep forest. If ``use_predictor`` is False, this parameter will
        have no effect.
    backend : :obj:`{"custom", "sklearn"}`, default= :obj:`"custom"`
        The backend of the forest estimator. Supported backends are ``custom``
        for higher time and memory efficiency and ``sklearn`` for additional
        functionality.
    n_tolerant_rounds : :obj:`int`, default=2
        Specify when to conduct early stopping. The training process
        terminates when the validation performance on the training set does
        not improve compared against the best validation performance achieved
        so far for ``n_tolerant_rounds`` rounds.
    delta : :obj:`float`, default=1e-5
        Specify the threshold on early stopping. The counting on
        ``n_tolerant_rounds`` is triggered if the performance of a fitted
        cascade layer does not improve by ``delta`` compared against the best
        validation performance achieved so far.
    partial_mode : :obj:`bool`, default=False
        Whether to train the deep forest in partial mode. For large
        datasets, it is recommended to use the partial mode.

        - If ``True``, the partial mode is activated and all fitted
          estimators will be dumped in a local buffer;
        - If ``False``, all fitted estimators are directly stored in the
          memory.
    n_jobs : :obj:`int` or ``None``, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. None means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    random_state : :obj:`int` or ``None``, default=None

        - If :obj:`int`, ``random_state`` is the seed used by the random
          number generator;
        - If ``None``, the random number generator is the RandomState
          instance used by :mod:`np.random`.
    verbose : :obj:`int`, default=1
        Controls the verbosity when fitting and predicting.

        - If ``<= 0``, silent mode, which means no logging information will
          be displayed;
        - If ``1``, logging information on the cascade layer level will be
          displayed;
        - If ``> 1``, full logging information will be displayed.
"""


__classifier_fit_doc = """

    .. note::

        Deep forest supports two kinds of modes for training:

        - **Full memory mode**, in which the training / testing data and
          all fitted estimators are directly stored in the memory.
        - **Partial mode**, in which after fitting each estimator using
          the training data, it will be dumped in the buffer. During the
          evaluating stage, the dumped estimators are reloaded into the
          memory sequentially to evaluate the testing data.

        By setting the ``partial_mode`` to ``True``, the partial mode is
        activated, and a local buffer will be created at the current
        directory. The partial mode is able to reduce the running memory
        cost when training the deep forest.

    Parameters
    ----------
    X : :obj: array-like of shape (n_samples, n_features)
        The training data. Internally, it will be converted to
        ``np.uint8``.
    y : :obj:`numpy.ndarray` of shape (n_samples,)
        The class labels of input samples.
    sample_weight : :obj:`numpy.ndarray` of shape (n_samples,), default=None
        Sample weights. If ``None``, then samples are equally weighted.
"""

__regressor_model_doc = """
    Parameters
    ----------
    n_bins : :obj:`int`, default=255
        The number of bins used for non-missing values. In addition to the
        ``n_bins`` bins, one more bin is reserved for missing values. Its
        value must be no smaller than 2 and no greater than 255.
    bin_subsample : :obj:`int`, default=200,000
        The number of samples used to construct feature discrete bins. If
        the size of training set is smaller than ``bin_subsample``, then all
        training samples will be used.
    bin_type : :obj:`{"percentile", "interval"}`, default= :obj:`"percentile"`
        The type of binner used to bin feature values into integer-valued bins.

        - If ``"percentile"``, each bin will have approximately the same
          number of distinct feature values.
        - If ``"interval"``, each bin will have approximately the same size.
    max_layers : :obj:`int`, default=20
        The maximum number of cascade layers in the deep forest. Notice that
        the actual number of layers can be smaller than ``max_layers`` because
        of the internal early stopping stage.
    criterion : :obj:`{"mse", "mae"}`, default= :obj:`"mse"`
        The function to measure the quality of a split. Supported criteria are 
        ``mse`` for the mean squared error, which is equal to variance reduction 
        as feature selection criterion, and ``mae`` for the mean absolute error.
    n_estimators : :obj:`int`, default=2
        The number of estimator in each cascade layer. It will be multiplied
        by 2 internally because each estimator contains a
        :class:`RandomForestRegressor` and a :class:`ExtraTreesRegressor`,
        respectively.
    n_trees : :obj:`int`, default=100
        The number of trees in each estimator.
    max_depth : :obj:`int`, default=None
        The maximum depth of each tree. ``None`` indicates no constraint.
    min_samples_split : :obj:`int`, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : :obj:`int`, default=1
        The minimum number of samples required to be at a leaf node.
    use_predictor : :obj:`bool`, default=False
        Whether to build the predictor concatenated to the deep forest. Using
        the predictor may improve the performance of deep forest.
    predictor : :obj:`{"forest", "xgboost", "lightgbm"}`, default= :obj:`"forest"`
        The type of the predictor concatenated to the deep forest. If
        ``use_predictor`` is False, this parameter will have no effect.
    predictor_kwargs : :obj:`dict`, default={}
        The configuration of the predictor concatenated to the deep forest.
        Specifying this will extend/overwrite the original parameters inherit
        from deep forest.
        If ``use_predictor`` is False, this parameter will have no effect.
    backend : :obj:`{"custom", "sklearn"}`, default= :obj:`"custom"`
        The backend of the forest estimator. Supported backends are ``custom``
        for higher time and memory efficiency and ``sklearn`` for additional
        functionality.
    n_tolerant_rounds : :obj:`int`, default=2
        Specify when to conduct early stopping. The training process
        terminates when the validation performance on the training set does
        not improve compared against the best validation performance achieved
        so far for ``n_tolerant_rounds`` rounds.
    delta : :obj:`float`, default=1e-5
        Specify the threshold on early stopping. The counting on
        ``n_tolerant_rounds`` is triggered if the performance of a fitted
        cascade layer does not improve by ``delta`` compared against the best
        validation performance achieved so far.
    partial_mode : :obj:`bool`, default=False
        Whether to train the deep forest in partial mode. For large
        datasets, it is recommended to use the partial mode.

        - If ``True``, the partial mode is activated and all fitted
          estimators will be dumped in a local buffer;
        - If ``False``, all fitted estimators are directly stored in the
          memory.
    n_jobs : :obj:`int` or ``None``, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. None means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    random_state : :obj:`int` or ``None``, default=None

        - If :obj:`int`, ``random_state`` is the seed used by the random
          number generator;
        - If ``None``, the random number generator is the RandomState
          instance used by :mod:`np.random`.
    verbose : :obj:`int`, default=1
        Controls the verbosity when fitting and predicting.

        - If ``<= 0``, silent mode, which means no logging information will
          be displayed;
        - If ``1``, logging information on the cascade layer level will be
          displayed;
        - If ``> 1``, full logging information will be displayed.
"""

__regressor_fit_doc = """

    .. note::

        Deep forest supports two kinds of modes for training:

        - **Full memory mode**, in which the training / testing data and
          all fitted estimators are directly stored in the memory.
        - **Partial mode**, in which after fitting each estimator using
          the training data, it will be dumped in the buffer. During the
          evaluating stage, the dumped estimators are reloaded into the
          memory sequentially to evaluate the testing data.

        By setting the ``partial_mode`` to ``True``, the partial mode is
        activated, and a local buffer will be created at the current
        directory. The partial mode is able to reduce the running memory
        cost when training the deep forest.

    Parameters
    ----------
    X : :obj: array-like of shape (n_samples, n_features)
        The training data. Internally, it will be converted to
        ``np.uint8``.
    y : :obj:`numpy.ndarray` of shape (n_samples,) or (n_samples, n_outputs)
        The target values of input samples.
    sample_weight : :obj:`numpy.ndarray` of shape (n_samples,), default=None
        Sample weights. If ``None``, then samples are equally weighted.
"""


def deepforest_model_doc(header, item):
    """
    Decorator on obtaining documentation for deep forest models.

    Parameters
    ----------
    header: string
       Introduction to the decorated class or method.
    item : string
       Type of the docstring item.
    """

    def get_doc(item):
        """Return the selected item."""
        __doc = {
            "regressor_model": __regressor_model_doc,
            "regressor_fit": __regressor_fit_doc,
            "classifier_model": __classifier_model_doc,
            "classifier_fit": __classifier_fit_doc,
        }

        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc


class BaseCascadeForest(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        self.n_bins = n_bins
        self.bin_subsample = bin_subsample
        self.bin_type = bin_type
        self.max_layers = max_layers
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.predictor_kwargs = predictor_kwargs
        self.backend = backend
        self.n_tolerant_rounds = n_tolerant_rounds
        self.delta = delta
        self.partial_mode = partial_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Utility variables
        self.n_layers_ = 0
        self.is_fitted_ = False

        # Internal containers
        self.layers_ = {}
        self.binners_ = {}
        self.buffer_ = _io.Buffer(partial_mode)

        # Predictor
        self.use_predictor = use_predictor
        self.predictor = predictor

    def __len__(self):
        return self.n_layers_

    def __getitem__(self, index):
        return self._get_layer(index)

    def _get_n_output(self, y):
        """Return the number of output inferred from the training labels."""
        if is_classifier(self):
            n_output = np.unique(y).shape[0]  # classification
            return n_output
        return y.shape[1] if len(y.shape) > 1 else 1  # regression

    def _make_layer(self, **layer_args):
        """Make and configure a cascade layer."""
        if not hasattr(self, "use_custom_estimator"):
            # Use built-in cascade layers
            if is_classifier(self):
                layer = ClassificationCascadeLayer(**layer_args)
            else:
                layer = RegressionCascadeLayer(**layer_args)
        else:
            # Use customized cascade layers
            layer = CustomCascadeLayer(
                layer_args["layer_idx"],
                self.n_splits,
                layer_args["n_outputs"],
                self.dummy_estimators,
                layer_args["partial_mode"],
                layer_args["buffer"],
                layer_args["random_state"],
                layer_args["verbose"],
                is_classifier(self),
            )

        return layer

    def _get_layer(self, layer_idx):
        """Get the layer from the internal container according to the index."""
        if not 0 <= layer_idx < self.n_layers_:
            msg = (
                "The layer index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise IndexError(msg.format(self.n_layers_ - 1, layer_idx))

        layer_key = "layer_{}".format(layer_idx)

        return self.layers_[layer_key]

    def _set_layer(self, layer_idx, layer):
        """
        Register a layer into the internal container with the given index."""
        layer_key = "layer_{}".format(layer_idx)
        if layer_key in self.layers_:
            msg = (
                "Layer with the key {} already exists in the internal"
                " container."
            )
            raise RuntimeError(msg.format(layer_key))

        self.layers_.update({layer_key: layer})

    def _get_binner(self, binner_idx):
        """Get the binner from the internal container with the given index."""
        if not 0 <= binner_idx <= self.n_layers_:
            msg = (
                "The binner index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise ValueError(msg.format(self.n_layers_, binner_idx))

        binner_key = "binner_{}".format(binner_idx)

        return self.binners_[binner_key]

    def _set_binner(self, binner_idx, binner):
        """
        Register a binner into the internal container with the given index."""
        binner_key = "binner_{}".format(binner_idx)
        if binner_key in self.binners_:
            msg = (
                "Binner with the key {} already exists in the internal"
                " container."
            )
            raise RuntimeError(msg.format(binner_key))

        self.binners_.update({binner_key: binner})

    def _set_n_trees(self, layer_idx):
        """
        Set the number of decision trees for each estimator in the cascade
        layer with `layer_idx` using the pre-defined rules.
        """
        # The number of trees for each layer is fixed as `n_trees`.
        if isinstance(self.n_trees, numbers.Integral):
            if not self.n_trees > 0:
                msg = "n_trees = {} should be strictly positive."
                raise ValueError(msg.format(self.n_trees))
            return self.n_trees
        # The number of trees for the first 5 layers grows linearly with
        # `layer_idx`, while that for remaining layers is fixed to `500`.
        elif self.n_trees == "auto":
            n_trees = 100 * (layer_idx + 1)
            return n_trees if n_trees <= 500 else 500
        else:
            msg = (
                "Invalid value for n_trees. Allowed values are integers or"
                " 'auto'."
            )
            raise ValueError(msg)

    def _check_input(self, X, y=None):
        """
        Check the input data and set the attributes if X is training data."""
        is_training_data = y is not None

        if is_training_data:
            _, self.n_features_ = X.shape
            self.n_outputs_ = self._get_n_output(y)

    def _validate_params(self):
        """
        Validate parameters, those passed to the sub-modules will not be
        checked here."""
        if not self.n_outputs_ > 0:
            msg = "n_outputs = {} should be strictly positive."
            raise ValueError(msg.format(self.n_outputs_))

        if not self.max_layers > 0:
            msg = "max_layers = {} should be strictly positive."
            raise ValueError(msg.format(self.max_layers))

        if not self.backend in ("custom", "sklearn"):
            msg = "backend = {} should be one of {{custom, sklearn}}."
            raise ValueError(msg.format(self.backend))

        if not self.n_tolerant_rounds > 0:
            msg = "n_tolerant_rounds = {} should be strictly positive."
            raise ValueError(msg.format(self.n_tolerant_rounds))

        if not self.delta >= 0:
            msg = "delta = {} should be no smaller than 0."
            raise ValueError(msg.format(self.delta))

    def _bin_data(self, binner, X, is_training_data=True):
        """
        Bin data X. If X is training data, the bin mapper is fitted first."""
        description = "training" if is_training_data else "testing"

        tic = time.time()
        if is_training_data:
            X_binned = binner.fit_transform(X)
        else:
            X_binned = binner.transform(X)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time.time()
        binning_time = toc - tic

        if self.verbose > 1:
            msg = (
                "{} Binning {} data: {:.3f} MB => {:.3f} MB |"
                " Elapsed = {:.3f} s"
            )
            print(
                msg.format(
                    _utils.ctime(),
                    description,
                    X.nbytes / (1024 * 1024),
                    X_binned.nbytes / (1024 * 1024),
                    binning_time,
                )
            )

        return X_binned

    def _handle_early_stopping(self):
        """
        Remove cascade layers temporarily added, along with dumped objects on
        the local buffer if `partial_mode` is True."""
        for layer_idx in range(
            self.n_layers_ - 1, self.n_layers_ - self.n_tolerant_rounds, -1
        ):
            self.layers_.pop("layer_{}".format(layer_idx))
            self.binners_.pop("binner_{}".format(layer_idx))

            if self.partial_mode:
                self.buffer_.del_estimator(layer_idx)

        # The last layer temporarily added only requires dumped estimators on
        # the local buffer to be removed.
        if self.partial_mode:
            self.buffer_.del_estimator(self.n_layers_)

        self.n_layers_ -= self.n_tolerant_rounds - 1

        if self.verbose > 0:
            msg = "{} The optimal number of layers: {}"
            print(msg.format(_utils.ctime(), self.n_layers_))

    def _if_improved(self, new_pivot, pivot, delta):
        """
        Return true if new validation result is better than previous"""
        if is_classifier(self):
            return new_pivot >= pivot + delta
        return new_pivot <= pivot - delta

    @abstractmethod
    def _repr_performance(self, pivot):
        """Format the printting information on training performance."""

    @abstractmethod
    def predict(self, X):
        """
        Predict class labels or regression values for X.
        For classification, the predicted class for each sample in X is
        returned. For regression, the predicted value based on X is returned.
        """

    @property
    def n_aug_features_(self):
        if not hasattr(self, "use_custom_estimator"):
            return 2 * self.n_estimators * self.n_outputs_
        else:
            return self.n_estimators * self.n_outputs_

    # flake8: noqa: E501
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(
            X,
            y,
            multi_output=True
            if type_of_target(y)
            in ("continuous-multioutput", "multiclass-multioutput")
            else False,
        )

        self._check_input(X, y)
        self._validate_params()
        n_counter = 0  # a counter controlling the early stopping

        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )

        # Bin the training data
        X_train_ = self._bin_data(binner_, X, is_training_data=True)
        X_train_ = self.buffer_.cache_data(0, X_train_, is_training_data=True)

        # =====================================================================
        # Training Stage
        # =====================================================================

        if self.verbose > 0:
            print("{} Start to fit the model:".format(_utils.ctime()))

        # Build the first cascade layer
        layer_ = self._make_layer(
            layer_idx=0,
            n_outputs=self.n_outputs_,
            criterion=self.criterion,
            n_estimators=self.n_estimators,
            n_trees=self._set_n_trees(0),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            backend=self.backend,
            partial_mode=self.partial_mode,
            buffer=self.buffer_,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        if self.verbose > 0:
            print("{} Fitting cascade layer = {:<2}".format(_utils.ctime(), 0))

        tic = time.time()
        X_aug_train_ = layer_.fit_transform(
            X_train_, y, sample_weight=sample_weight
        )
        toc = time.time()
        training_time = toc - tic

        # Set the reference performance
        pivot = layer_.val_performance_

        if self.verbose > 0:
            msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
            print(
                msg.format(
                    _utils.ctime(),
                    0,
                    self._repr_performance(pivot),
                    training_time,
                )
            )

        # Copy the snapshot of `X_aug_train_` for training the predictor.
        if self.use_predictor:
            snapshot_X_aug_train_ = np.copy(X_aug_train_)

        # Add the first cascade layer, binner
        self._set_layer(0, layer_)
        self._set_binner(0, binner_)
        self.n_layers_ += 1

        # Pre-allocate the global array on storing training data
        X_middle_train_ = _utils.init_array(X_train_, self.n_aug_features_)

        # ====================================================================
        # Main loop on the training stage
        # ====================================================================

        while self.n_layers_ < self.max_layers:

            # Set the binner
            binner_ = Binner(
                n_bins=self.n_bins,
                bin_subsample=self.bin_subsample,
                bin_type=self.bin_type,
                random_state=self.random_state,
            )

            X_binned_aug_train_ = self._bin_data(
                binner_, X_aug_train_, is_training_data=True
            )

            X_middle_train_ = _utils.merge_array(
                X_middle_train_, X_binned_aug_train_, self.n_features_
            )

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = self._make_layer(
                layer_idx=layer_idx,
                n_outputs=self.n_outputs_,
                criterion=self.criterion,
                n_estimators=self.n_estimators,
                n_trees=self._set_n_trees(layer_idx),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                backend=self.backend,
                partial_mode=self.partial_mode,
                buffer=self.buffer_,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            X_middle_train_ = self.buffer_.cache_data(
                layer_idx, X_middle_train_, is_training_data=True
            )

            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            X_aug_train_ = layer_.fit_transform(
                X_middle_train_, y, sample_weight=sample_weight
            )
            toc = time.time()
            training_time = toc - tic

            new_pivot = layer_.val_performance_

            if self.verbose > 0:
                msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        self._repr_performance(new_pivot),
                        training_time,
                    )
                )

            # Check on early stopping: If the performance of the fitted
            # cascade layer does not improved by `delta` compared to the best
            # performance achieved so far for `n_tolerant_rounds`, the
            # training stage will terminate before reaching the maximum number
            # of layers.

            if self._if_improved(new_pivot, pivot, self.delta):

                # Update the cascade layer
                self._set_layer(layer_idx, layer_)
                self._set_binner(layer_idx, binner_)
                self.n_layers_ += 1

                # Performance calibration
                n_counter = 0
                pivot = new_pivot

                if self.use_predictor:
                    snapshot_X_aug_train_ = np.copy(X_aug_train_)
            else:
                n_counter += 1

                if self.verbose > 0:
                    msg = "{} Early stopping counter: {} out of {}"
                    print(
                        msg.format(
                            _utils.ctime(), n_counter, self.n_tolerant_rounds
                        )
                    )

                # Activate early stopping if reaching `n_tolerant_rounds`
                if n_counter == self.n_tolerant_rounds:

                    if self.verbose > 0:
                        msg = "{} Handling early stopping"
                        print(msg.format(_utils.ctime()))

                    self._handle_early_stopping()
                    break

                # Add the fitted layer, and binner temporarily
                self._set_layer(layer_idx, layer_)
                self._set_binner(layer_idx, binner_)
                self.n_layers_ += 1

        if self.n_layers_ == self.max_layers and self.verbose > 0:
            msg = "{} Reaching the maximum number of layers: {}"
            print(msg.format(_utils.ctime(), self.max_layers))

        # Build the predictor if `self.use_predictor` is True
        if self.use_predictor:
            # Use built-in predictors
            if self.predictor in ("forest", "xgboost", "lightgbm"):
                if is_classifier(self):
                    self.predictor_ = _build_classifier_predictor(
                        self.predictor,
                        self.criterion,
                        self.n_trees,
                        self.n_outputs_,
                        self.max_depth,
                        self.min_samples_split,
                        self.min_samples_leaf,
                        self.n_jobs,
                        self.random_state,
                        self.predictor_kwargs,
                    )
                else:
                    self.predictor_ = _build_regressor_predictor(
                        self.predictor,
                        self.criterion,
                        self.n_trees,
                        self.n_outputs_,
                        self.max_depth,
                        self.min_samples_split,
                        self.min_samples_leaf,
                        self.n_jobs,
                        self.random_state,
                        self.predictor_kwargs,
                    )
            elif self.predictor == "custom":
                if not hasattr(self, "predictor_"):
                    msg = "Missing predictor after calling `set_predictor`"
                    raise RuntimeError(msg)

            binner_ = Binner(
                n_bins=self.n_bins,
                bin_subsample=self.bin_subsample,
                bin_type=self.bin_type,
                random_state=self.random_state,
            )

            X_binned_aug_train_ = self._bin_data(
                binner_, snapshot_X_aug_train_, is_training_data=True
            )

            X_middle_train_ = _utils.merge_array(
                X_middle_train_, X_binned_aug_train_, self.n_features_
            )

            if self.verbose > 0:
                msg = "{} Fitting the concatenated predictor: {}"
                print(msg.format(_utils.ctime(), self.predictor))

            tic = time.time()
            self.predictor_.fit(X_middle_train_, y, sample_weight)
            toc = time.time()

            if self.verbose > 0:
                msg = "{} Finish building the predictor | Elapsed = {:.3f} s"
                print(msg.format(_utils.ctime(), toc - tic))

            self._set_binner(self.n_layers_, binner_)
            self.predictor_ = self.buffer_.cache_predictor(self.predictor_)

        self.is_fitted_ = True

        return self

    def set_estimator(self, estimators, n_splits=5):
        """
        Specify the custom base estimators for cascade layers.

        Parameters
        ----------
        estimators : :obj:`list`
            A list of your base estimators, will be used in all cascade layers.
        n_splits : :obj:`int`, default=5
            The number of folds, must be at least 2.
        """
        # Validation check
        if not isinstance(estimators, list):
            msg = (
                "estimators should be a list that stores instantiated"
                " objects of your base estimator."
            )
            raise ValueError(msg)

        for idx, estimator in enumerate(estimators):
            if not callable(getattr(estimator, "fit", None)):
                msg = "The `fit` method of estimator = {} is not callable."
                raise AttributeError(msg.format(idx))

            if is_classifier(self) and not callable(
                getattr(estimator, "predict_proba", None)
            ):
                msg = (
                    "The `predict_proba` method of estimator = {} is not"
                    " callable."
                )
                raise AttributeError(msg.format(idx))

            if not is_classifier(self) and not callable(
                getattr(estimator, "predict", None)
            ):
                msg = "The `predict` method of estimator = {} is not callable."
                raise AttributeError(msg.format(idx))

        if not n_splits >= 2:
            msg = "n_splits = {} should be at least 2."
            raise ValueError(msg.format(n_splits))

        self.dummy_estimators = estimators
        self.n_splits = n_splits
        self.use_custom_estimator = True

        # Update attributes
        self.n_estimators = len(estimators)

    def set_predictor(self, predictor):
        """
        Specify the custom predictor concatenated to deep forest.

        Parameters
        ----------
        predictor : :obj:`object`
            The instantiated object of your predictor.
        """
        # Validation check
        if not callable(getattr(predictor, "fit", None)):
            msg = "The `fit` method of the predictor is not callable."
            raise AttributeError(msg)

        if is_classifier(self) and not callable(
            getattr(predictor, "predict_proba", None)
        ):
            msg = (
                "The `predict_proba` method of the predictor is not"
                " callable."
            )
            raise AttributeError(msg)

        if not is_classifier(self) and not callable(
            getattr(predictor, "predict", None)
        ):
            msg = "The `predict` method of the predictor is not callable."
            raise AttributeError(msg)

        # Set related attributes
        self.predictor = "custom"
        self.predictor_ = predictor
        self.use_predictor = True

    def get_layer_feature_importances(self, layer_idx):
        """
        Return the feature importances of ``layer_idx``-th cascade layer.

        Parameters
        ----------
        layer_idx : :obj:`int`
            The index of the cascade layer, should be in the range
            ``[0, self.n_layers_-1]``.

        Returns
        -------
        feature_importances_: :obj:`numpy.ndarray` of shape (n_features,)
            The impurity-based feature importances of the cascade layer.
            Notice that the number of input features are different between the
            first cascade layer and remaining cascade layers.


        .. note::
            - This method is only applicable when deep forest is built using
              the ``sklearn`` backend
            - The functionality of this method is not available when using
              customized estimators in deep forest.
        """
        if self.backend == "custom":
            msg = (
                "Please use the sklearn backend to get the feature"
                " importances property for each cascade layer."
            )
            raise RuntimeError(msg)
        layer = self._get_layer(layer_idx)
        return layer.feature_importances_

    def get_estimator(self, layer_idx, est_idx, estimator_type):
        """
        Get estimator from a cascade layer in the deep forest.

        Parameters
        ----------
        layer_idx : :obj:`int`
            The index of the cascade layer, should be in the range
            ``[0, self.n_layers_-1]``.
        est_idx : :obj:`int`
            The index of the estimator, should be in the range
            ``[0, self.n_estimators]``.
        estimator_type : :obj:`{"rf", "erf", "custom"}`
            Specify the forest type.

            - If ``rf``, return the random forest.
            - If ``erf``, return the extremely random forest.
            - If ``custom``, return the customized estimator, only applicable
              when using customized estimators in deep forest via
              :meth:`set_estimator`.

        Returns
        -------
        estimator : Estimator with the given index.
        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")

        # Check the given index
        if not 0 <= layer_idx < self.n_layers_:
            msg = (
                "`layer_idx` should be in the range [0, {}), but got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_layers_, layer_idx))

        if not 0 <= est_idx < self.n_estimators:
            msg = (
                "`est_idx` should be in the range [0, {}), but got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_estimators, est_idx))

        if estimator_type not in ("rf", "erf", "custom"):
            msg = (
                "`estimator_type` should be one of {{rf, erf, custom}},"
                " but got {} instead."
            )
            raise ValueError(msg.format(estimator_type))

        if estimator_type == "custom" and not self.use_custom_estimator:
            msg = (
                "`estimator_type` = {} is only applicable when using"
                "customized estimators in deep forest."
            )
            raise ValueError(msg.format(estimator_type))

        layer = self._get_layer(layer_idx)
        est_key = "{}-{}-{}".format(layer_idx, est_idx, estimator_type)
        estimator = layer.estimators_[est_key]

        # Load the model if in partial mode
        if self.partial_mode:
            estimator = self.buffer_.load_estimator(estimator)

        return estimator.estimator_

    def save(self, dirname="model"):
        """
        Save the model to the directory ``dirname``.

        Parameters
        ----------
        dirname : :obj:`str`, default="model"
            The name of the output directory.


        .. warning::
            Other methods on model serialization such as :mod:`pickle` or
            :mod:`joblib` are not recommended, especially when ``partial_mode``
            is set to True.
        """
        # Create the output directory
        _io.model_mkdir(dirname)

        # Save each object sequentially
        d = {}
        d["n_estimators"] = self.n_estimators
        d["criterion"] = self.criterion
        d["n_layers"] = self.n_layers_
        d["n_features"] = self.n_features_
        d["n_outputs"] = self.n_outputs_
        d["partial_mode"] = self.partial_mode
        d["buffer"] = self.buffer_
        d["verbose"] = self.verbose
        d["use_predictor"] = self.use_predictor
        d["is_classifier"] = is_classifier(self)
        d["use_custom_estimator"] = (
            True if hasattr(self, "use_custom_estimator") else False
        )

        if self.use_predictor:
            d["predictor"] = self.predictor

        # Save label encoder if labels are encoded.
        if hasattr(self, "labels_are_encoded"):
            d["labels_are_encoded"] = self.labels_are_encoded
            d["label_encoder"] = self.label_encoder_

        _io.model_saveobj(dirname, "param", d)
        _io.model_saveobj(dirname, "binner", self.binners_)
        _io.model_saveobj(dirname, "layer", self.layers_, self.partial_mode)

        if self.use_predictor:
            _io.model_saveobj(
                dirname, "predictor", self.predictor_, self.partial_mode
            )

    def load(self, dirname):
        """
        Load the model from the directory ``dirname``.

        Parameters
        ----------
        dirname : :obj:`str`
            The name of the input directory.


        .. note::
            The dumped model after calling :meth:`load_model` is not exactly
            the same as the model before saving, because many objects
            irrelevant to model inference will not be saved.
        """
        d = _io.model_loadobj(dirname, "param")

        # Set parameter
        self.n_estimators = d["n_estimators"]
        self.n_layers_ = d["n_layers"]
        self.n_features_ = d["n_features"]
        self.n_outputs_ = d["n_outputs"]
        self.partial_mode = d["partial_mode"]
        self.buffer_ = d["buffer"]
        self.verbose = d["verbose"]
        self.use_predictor = d["use_predictor"]
        if d["use_custom_estimator"]:
            self.use_custom_estimator = True

        # Load label encoder if labels are encoded.
        if "labels_are_encoded" in d:
            self.labels_are_encoded = d["labels_are_encoded"]
            self.label_encoder_ = d["label_encoder"]

        # Load internal containers
        self.binners_ = _io.model_loadobj(dirname, "binner")
        self.layers_ = _io.model_loadobj(dirname, "layer", d)
        if self.use_predictor:
            self.predictor_ = _io.model_loadobj(dirname, "predictor", d)

        # Some checks after loading
        if len(self.layers_) != self.n_layers_:
            msg = (
                "The size of the loaded dictionary of layers {} does not"
                " match n_layers_ {}."
            )
            raise RuntimeError(msg.format(len(self.layers_), self.n_layers_))

        self.is_fitted_ = True

    def clean(self):
        """Clean the buffer created by the model."""
        if self.partial_mode:
            self.buffer_.close()


@deepforest_model_doc(
    """Implementation of the deep forest for classification.""",
    "classifier_model",
)
class CascadeForestClassifier(BaseCascadeForest, ClassifierMixin):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        # Used to deal with classification labels
        self.labels_are_encoded = False
        self.type_of_target_ = None
        self.label_encoder_ = None

    def _encode_class_labels(self, y):
        """
        Fit the internal label encoder and return encoded labels.
        """
        self.type_of_target_ = type_of_target(y)
        if self.type_of_target_ in ("binary", "multiclass"):
            self.labels_are_encoded = True
            self.label_encoder_ = LabelEncoder()
            encoded_y = self.label_encoder_.fit_transform(y)
        else:
            msg = (
                "CascadeForestClassifier is used for binary and multiclass"
                " classification, wheras the training labels seem not to"
                " be any one of them."
            )
            raise ValueError(msg)

        return encoded_y

    def _decode_class_labels(self, y):
        """
        Transform the predicted labels back to original encoding.
        """
        if self.labels_are_encoded:
            decoded_y = self.label_encoder_.inverse_transform(y)
        else:
            decoded_y = y

        return decoded_y

    def _repr_performance(self, pivot):
        msg = "Val Acc = {:.3f} %"
        return msg.format(pivot * 100)

    @deepforest_model_doc(
        """Build a deep forest using the training data.""", "classifier_fit"
    )
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(
            X,
            y,
            multi_output=True
            if type_of_target(y)
            in ("continuous-multioutput", "multiclass-multioutput")
            else False,
        )
        # Check the input for classification
        y = self._encode_class_labels(y)

        super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : :obj: array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        proba : :obj:`numpy.ndarray` of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = check_array(X)

        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        binner_ = self._get_binner(0)
        X_test = self._bin_data(binner_, X, is_training_data=False)
        X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)

            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                X_aug_test_ = layer.transform(X_test)
            elif layer_idx < self.n_layers_ - 1:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )
                X_aug_test_ = layer.transform(X_middle_test_)
            else:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                # Skip calling the `transform` if not using the predictor
                if self.use_predictor:
                    X_aug_test_ = layer.transform(X_middle_test_)

        if self.use_predictor:

            if self.verbose > 0:
                print("{} Evaluating the predictor".format(_utils.ctime()))

            binner_ = self._get_binner(self.n_layers_)
            X_aug_test_ = self._bin_data(
                binner_, X_aug_test_, is_training_data=False
            )
            X_middle_test_ = _utils.merge_array(
                X_middle_test_, X_aug_test_, self.n_features_
            )

            predictor = self.buffer_.load_predictor(self.predictor_)
            proba = predictor.predict_proba(X_middle_test_)
        else:
            if self.n_layers_ > 1:
                proba = layer.predict_full(X_middle_test_)
                proba = _utils.merge_proba(proba, self.n_outputs_)
            else:
                # Directly merge results with one cascade layer only
                proba = _utils.merge_proba(X_aug_test_, self.n_outputs_)

        return proba

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : :obj: array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        y : :obj:`numpy.ndarray` of shape (n_samples,)
            The predicted classes.
        """
        X = check_array(X)

        proba = self.predict_proba(X)
        y = self._decode_class_labels(np.argmax(proba, axis=1))
        return y


@deepforest_model_doc(
    """Implementation of the deep forest for regression.""", "regressor_model"
)
class CascadeForestRegressor(BaseCascadeForest, RegressorMixin):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="mse",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        # Used to deal with target values
        self.type_of_target_ = None

    def _check_target_values(self, y):
        """Check the input target values for regressor."""
        self.type_of_target_ = type_of_target(y)

        if not self._check_array_numeric(y):
            msg = (
                "CascadeForestRegressor only accepts numeric values as"
                " valid target values."
            )
            raise ValueError(msg)

        if self.type_of_target_ not in (
            "continuous",
            "continuous-multioutput",
            "multiclass",
            "multiclass-multioutput",
        ):
            msg = (
                "CascadeForestRegressor is used for univariate or"
                " multi-variate regression, but the target values seem not"
                " to be one of them."
            )
            raise ValueError(msg)

    def _check_array_numeric(self, y):
        """Check the input numpy array y is all numeric."""
        numeric_types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        if y.dtype.kind in numeric_types:
            return True
        else:
            return False

    def _repr_performance(self, pivot):
        msg = "Val MSE = {:.5f}"
        return msg.format(pivot)

    @deepforest_model_doc(
        """Build a deep forest using the training data.""", "regressor_fit"
    )
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(
            X,
            y,
            multi_output=True
            if type_of_target(y)
            in ("continuous-multioutput", "multiclass-multioutput")
            else False,
        )

        # Check the input for regression
        self._check_target_values(y)

        super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : :obj: array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        y : :obj:`numpy.ndarray` of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        X = check_array(X)

        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        binner_ = self._get_binner(0)
        X_test = self._bin_data(binner_, X, is_training_data=False)
        X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)

            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                X_aug_test_ = layer.transform(X_test)
            elif layer_idx < self.n_layers_ - 1:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )
                X_aug_test_ = layer.transform(X_middle_test_)
            else:
                binner_ = self._get_binner(layer_idx)
                X_aug_test_ = self._bin_data(
                    binner_, X_aug_test_, is_training_data=False
                )
                X_middle_test_ = _utils.merge_array(
                    X_middle_test_, X_aug_test_, self.n_features_
                )

                # Skip calling the `transform` if not using the predictor
                if self.use_predictor:
                    X_aug_test_ = layer.transform(X_middle_test_)

        if self.use_predictor:

            if self.verbose > 0:
                print("{} Evaluating the predictor".format(_utils.ctime()))

            binner_ = self._get_binner(self.n_layers_)
            X_aug_test_ = self._bin_data(
                binner_, X_aug_test_, is_training_data=False
            )
            X_middle_test_ = _utils.merge_array(
                X_middle_test_, X_aug_test_, self.n_features_
            )

            predictor = self.buffer_.load_predictor(self.predictor_)
            _y = predictor.predict(X_middle_test_)
        else:
            if self.n_layers_ > 1:
                _y = layer.predict_full(X_middle_test_)
                _y = _utils.merge_proba(_y, self.n_outputs_)
            else:
                # Directly merge results with one cascade layer only
                _y = _utils.merge_proba(X_aug_test_, self.n_outputs_)

        return _y

# deepforest/forest.py
"""
Implementation of the forest model for classification in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
"""


__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

import numbers
from warnings import warn
import threading
from typing import List

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from joblib import Parallel, delayed
from joblib import effective_n_jobs

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import is_classifier
from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args

from . import _cutils as _LIB
from . import _forest as _C_FOREST

from .tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from .tree._tree import DOUBLE


MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_mask(random_state, n_samples, n_samples_bootstrap):
    """Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    sample_indices = sample_indices.astype(np.int32)
    sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)

    return sample_mask


def _parallel_build_trees(
    tree,
    X,
    y,
    n_samples_bootstrap,
    sample_weight,
    out,
    mask,
    is_classifier,
    lock,
):
    """
    Private function used to fit a single tree in parallel."""
    n_samples = X.shape[0]

    sample_mask = _generate_sample_mask(
        tree.random_state, n_samples, n_samples_bootstrap
    )

    # Fit the tree on the bootstrapped samples
    if sample_weight is not None:
        sample_weight = sample_weight[sample_mask]
    feature, threshold, children, value = tree.fit(
        X[sample_mask],
        y[sample_mask],
        sample_weight=sample_weight,
        check_input=False,
    )

    if not children.flags["C_CONTIGUOUS"]:
        children = np.ascontiguousarray(children)

    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)

    if is_classifier:
        value = np.squeeze(value, axis=1)
        value /= value.sum(axis=1)[:, np.newaxis]
    else:
        if len(value.shape) == 3:
            value = np.squeeze(value, axis=2)

    # Set the OOB predictions
    oob_prediction = _C_FOREST.predict(
        X[~sample_mask, :], feature, threshold, children, value
    )
    with lock:

        mask += ~sample_mask
        out[~sample_mask, :] += oob_prediction

    return feature, threshold, children, value


# [Source] Sklearn.ensemble._base.py
def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int or RandomState, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


# [Source] Sklearn.ensemble._base.py
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(
        n_jobs, n_estimators // n_jobs, dtype=int
    )
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_prediction(feature, threshold, children, value, X, out, lock):
    """This is a utility function for joblib's Parallel."""
    prediction = _C_FOREST.predict(X, feature, threshold, children, value)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


# [Source] Sklearn.ensemble._base.py
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(
        self, base_estimator, *, n_estimators=10, estimator_params=tuple()
    ):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.

        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                "n_estimators must be an integer, "
                "got {0}.".format(type(self.n_estimators))
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {0}.".format(self.n_estimators)
            )

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(
            **{p: getattr(self, p) for p in self.estimator_params}
        )

        # Pass the inferred class information to avoid redudant finding.
        if is_classifier(estimator):
            estimator.classes_ = self.classes_
            estimator.n_classes_ = np.array(self.n_classes_, dtype=np.int32)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_samples = max_samples

        # Internal containers
        self.features = []
        self.thresholds = []
        self.childrens = []
        self.values = []

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for i in range(self.n_estimators)
        ]

        # Pre-allocate OOB estimations
        if is_classifier(self):
            oob_decision_function = np.zeros(
                (n_samples, self.classes_[0].shape[0])
            )
        else:
            oob_decision_function = np.zeros((n_samples, self.n_outputs_))
        mask = np.zeros(n_samples)

        lock = threading.Lock()
        rets = Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            prefer="threads",
            require="sharedmem",
        )(
            delayed(_parallel_build_trees)(
                t,
                X,
                y,
                n_samples_bootstrap,
                sample_weight,
                oob_decision_function,
                mask,
                is_classifier(self),
                lock,
            )
            for i, t in enumerate(trees)
        )
        # Collect newly grown trees
        for feature, threshold, children, value in rets:

            # No check on feature and threshold since 1-D array is always
            # C-aligned and F-aligned.
            self.features.append(feature)
            self.thresholds.append(threshold)
            self.childrens.append(children)
            self.values.append(value)

        # Check the OOB predictions
        if (
            is_classifier(self)
            and (oob_decision_function.sum(axis=1) == 0).any()
        ):
            warn(
                "Some inputs do not have OOB predictions. "
                "This probably means too few trees were used "
                "to compute any reliable oob predictions."
            )
        if is_classifier(self):
            prediction = (
                oob_decision_function
                / oob_decision_function.sum(axis=1)[:, np.newaxis]
            )
        else:
            prediction = oob_decision_function / mask.reshape(-1, 1)

        self.oob_decision_function_ = prediction

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

    def _validate_y_class_weight(self, y):

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        'Given "%s".' % self.class_weight
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(
                    class_weight, y_original
                )

        return y, expanded_class_weight

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem",)(
            delayed(_accumulate_prediction)(
                self.features[i],
                self.thresholds[i],
                self.childrens[i],
                self.values[i],
                X,
                all_proba,
                lock,
            )
            for i in range(self.n_estimators)
        )

        for proba in all_proba:
            proba /= len(self.features)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


class RandomForestClassifier(ForestClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesClassifier(ForestClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None
    ):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem",)(
            delayed(_accumulate_prediction)(
                self.features[i],
                self.thresholds[i],
                self.childrens[i],
                self.values[i],
                X,
                [y_hat],
                lock,
            )
            for i in range(self.n_estimators)
        )

        y_hat /= self.n_estimators
        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred


class RandomForestRegressor(ForestRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesRegressor(ForestRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
            ),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

# deepforest/setup.py
import os
import numpy
from distutils.version import LooseVersion
from numpy.distutils.misc_util import Configuration


CYTHON_MIN_VERSION = "0.24"


def configuration(parent_package="", top_path=None):

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("deepforest", parent_package, top_path)
    config.add_subpackage("tree")

    config.add_extension(
        "_forest",
        sources=["_forest.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    config.add_extension(
        "_cutils",
        sources=["_cutils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    msg = (
        "Please install cython with a version >= {} in order to build a"
        " deepforest development version."
    )
    msg = msg.format(CYTHON_MIN_VERSION)

    try:
        import Cython

        if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
            msg += " Your version of Cython is {}.".format(Cython.__version__)
            raise ValueError(msg)
        from Cython.Build import cythonize
    except ImportError as exc:
        exc.args += (msg,)
        raise

    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())

# deepforest/tree/__init__.py
from .tree import BaseDecisionTree
from .tree import DecisionTreeClassifier
from .tree import DecisionTreeRegressor
from .tree import ExtraTreeClassifier
from .tree import ExtraTreeRegressor

__all__ = [
    "BaseDecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]

# deepforest/tree/_criterion.pxd
# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pxd


import numpy as np
cimport numpy as np

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for counters, child, and feature ID
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef const DOUBLE_t[:, ::1] y        # Values of y
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int reverse_reset(self) except -1 nogil
    cdef int update(self, SIZE_t new_pos) except -1 nogil
    cdef double node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil
    cdef double proxy_impurity_improvement(self) nogil

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride

cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total

# deepforest/tree/_criterion.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This class is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx


from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport WeightedMedianCalculator

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of the samples being considered
        samples : array-like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node

        """

        pass

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split

        Return
        ------
        double : improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left /
                             self.weighted_n_node_samples * impurity_left)))


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """

        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end) except -1 nogil:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """

        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """

        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride


cdef class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

            sum_total += self.sum_stride

        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(ClassificationCriterion):
    r"""Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion."""


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node to
        impurity_right : double pointer
            The memory address to save the impurity of the right node to
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) except -1 nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

cdef class MAE(RegressionCriterion):
    r"""Mean absolute error impurity criterion

       MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
       value and f_i is the predicted value."""
    def __dealloc__(self):
        """Destructor."""
        free(self.node_medians)

    cdef np.ndarray left_child
    cdef np.ndarray right_child
    cdef DOUBLE_t* node_medians

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.node_medians = NULL

        # Allocate memory for the accumulators
        safe_realloc(&self.node_medians, n_outputs)

        self.left_child = np.empty(n_outputs, dtype='object')
        self.right_child = np.empty(n_outputs, dtype='object')
        # initialize WeightedMedianCalculators
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) except -1 nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""

        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0

        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef void** left_child
        cdef void** right_child

        left_child = <void**> self.left_child.data
        right_child = <void**> self.right_child.data

        for k in range(self.n_outputs):
            (<WeightedMedianCalculator> left_child[k]).reset()
            (<WeightedMedianCalculator> right_child[k]).reset()

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # push method ends up calling safe_realloc, hence `except -1`
                # push all values to the right side,
                # since pos = start initially anyway
                (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

            self.weighted_n_node_samples += w
        # calculate the node medians
        for k in range(self.n_outputs):
            self.node_medians[k] = (<WeightedMedianCalculator> right_child[k]).get_median()

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef SIZE_t i, k
        cdef DOUBLE_t value
        cdef DOUBLE_t weight

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        # reset the WeightedMedianCalculators, left should have no
        # elements and right should have all elements.

        for k in range(self.n_outputs):
            # if left has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> left_child[k]).size()):
                # remove everything from left and put it into right
                (<WeightedMedianCalculator> left_child[k]).pop(&value,
                                                               &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> right_child[k]).push(value,
                                                                 weight)
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        cdef DOUBLE_t value
        cdef DOUBLE_t weight
        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        # reverse reset the WeightedMedianCalculators, right should have no
        # elements and left should have all elements.
        for k in range(self.n_outputs):
            # if right has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> right_child[k]).size()):
                # remove everything from right and put it into left
                (<WeightedMedianCalculator> right_child[k]).pop(&value,
                                                                &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> left_child[k]).push(value,
                                                                weight)
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # We are going to update right_child and left_child
        # from the direction that require the least amount of
        # computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from right and add to left
                    (<WeightedMedianCalculator> right_child[k]).remove(self.y[i, k], w)
                    # push method ends up calling safe_realloc, hence except -1
                    (<WeightedMedianCalculator> left_child[k]).push(self.y[i, k], w)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from left and add to right
                    (<WeightedMedianCalculator> left_child[k]).remove(self.y[i, k], w)
                    (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.pos = new_pos
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Computes the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        for k in range(self.n_outputs):
            dest[k] = <double> self.node_medians[k]

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]"""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t impurity = 0.0

        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity += fabs(self.y[i, k] - self.node_medians[k]) * w

        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    cdef void children_impurity(self, double* p_impurity_left,
                                double* p_impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).
        """

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef SIZE_t i, p, k
        cdef DOUBLE_t median
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t impurity_left = 0.0
        cdef DOUBLE_t impurity_right = 0.0

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> left_child[k]).get_median()
            for p in range(start, pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity_left += fabs(self.y[i, k] - median) * w
        p_impurity_left[0] = impurity_left / (self.weighted_n_left *
                                              self.n_outputs)

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> right_child[k]).get_median()
            for p in range(pos, end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity_right += fabs(self.y[i, k] - median) * w
        p_impurity_right[0] = impurity_right / (self.weighted_n_right *
                                                self.n_outputs)


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))

# deepforest/tree/_splitter.pxd
# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pxd


import numpy as np
cimport numpy as np

from ._criterion cimport Criterion

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for counters, child, and feature ID
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    DTYPE_t threshold      # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(self, object X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=*) except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) except -1 nogil

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil

# deepforest/tree/_splitter.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This class is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pyx


from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0
    self.improvement = -INFINITY

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X,
                   const DOUBLE_t[:, ::1] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : DOUBLE_t*
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.

        X_idx_sorted : ndarray, default=None
            The indexes of the sorted training input samples
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = y

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

    cdef np.ndarray X_idx_sorted
    cdef SIZE_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):

        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        self.X = X
        return 0


cdef class BestSplitter(BaseDenseSplitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil:
        """Find the best split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SIZE_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]

                # Sort samples along that feature; by
                # copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]

                sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start]:
                    features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p]):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves is used to avoid infinite value
                                current.threshold = <DTYPE_t>(Xf[p - 1] / 2.0 + Xf[p] / 2.0)

                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    samples[p], samples[partition_end] = samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef class RandomSplitter(BaseDenseSplitter):
    """Splitter for finding the best random split."""
    def __reduce__(self):
        return (RandomSplitter, (self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.min_weight_leaf,
                                 self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil:
        """Find the best random split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Draw random splits and pick the best
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t partition_end
        cdef SIZE_t feature_stride
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Find min, max
                min_feature_value = self.X[samples[start], current.feature]
                max_feature_value = min_feature_value
                Xf[start] = min_feature_value

                for p in range(start + 1, end):
                    current_feature_value = self.X[samples[p], current.feature]
                    Xf[p] = current_feature_value

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value:
                    features[f_j], features[n_total_constants] = features[n_total_constants], current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = <DTYPE_t>(rand_uniform(min_feature_value,
                                                              max_feature_value,
                                                              random_state))

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    p, partition_end = start, end
                    while p < partition_end:
                        if Xf[p] <= current.threshold:
                            p += 1
                        else:
                            partition_end -= 1

                            Xf[p], Xf[partition_end] = Xf[partition_end], Xf[p]
                            samples[p], samples[partition_end] = samples[partition_end], samples[p]

                    current.pos = partition_end

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                p, partition_end = start, end

                while p < partition_end:
                    if self.X[samples[p], best.feature] <= best.threshold:
                        p += 1
                    else:
                        partition_end -= 1

                        samples[p], samples[partition_end] = samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


cdef class BaseSparseSplitter(Splitter):
    # The sparse splitter works only with csc sparse matrix format
    cdef DTYPE_t* X_data
    cdef SIZE_t* X_indices
    cdef SIZE_t* X_indptr

    cdef SIZE_t n_total_samples

    cdef SIZE_t* index_to_samples
    cdef SIZE_t* sorted_samples

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        # Parent __cinit__ is automatically called

        self.X_data = NULL
        self.X_indices = NULL
        self.X_indptr = NULL

        self.n_total_samples = 0

        self.index_to_samples = NULL
        self.sorted_samples = NULL

    def __dealloc__(self):
        """Deallocate memory."""
        free(self.index_to_samples)
        free(self.sorted_samples)

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        if not isinstance(X, csc_matrix):
            raise ValueError("X should be in csc format")

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t n_samples = self.n_samples

        # Initialize X
        cdef np.ndarray[dtype=DTYPE_t, ndim=1] data = X.data
        cdef np.ndarray[dtype=SIZE_t, ndim=1] indices = X.indices
        cdef np.ndarray[dtype=SIZE_t, ndim=1] indptr = X.indptr
        cdef SIZE_t n_total_samples = X.shape[0]

        self.X_data = <DTYPE_t*> data.data
        self.X_indices = <SIZE_t*> indices.data
        self.X_indptr = <SIZE_t*> indptr.data
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        safe_realloc(&self.index_to_samples, n_total_samples)
        safe_realloc(&self.sorted_samples, n_samples)

        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t p
        for p in range(n_total_samples):
            index_to_samples[p] = -1

        for p in range(n_samples):
            index_to_samples[samples[p]] = p
        return 0

    cdef inline SIZE_t _partition(self, double threshold,
                                  SIZE_t end_negative, SIZE_t start_positive,
                                  SIZE_t zero_pos) nogil:
        """Partition samples[start:end] based on threshold."""

        cdef SIZE_t p
        cdef SIZE_t partition_end

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* index_to_samples = self.index_to_samples

        if threshold < 0.:
            p = self.start
            partition_end = end_negative
        elif threshold > 0.:
            p = start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            if Xf[p] <= threshold:
                p += 1

            else:
                partition_end -= 1

                Xf[p], Xf[partition_end] = Xf[partition_end], Xf[p]
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline void extract_nnz(self, SIZE_t feature,
                                 SIZE_t* end_negative, SIZE_t* start_positive,
                                 bint* is_samples_sorted) nogil:
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        Xf[start:end_negative[0]] and positive values Xf[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : SIZE_t,
            Index of the feature we want to extract non zero value.


        end_negative, start_positive : SIZE_t*, SIZE_t*,
            Return extracted non zero values in self.samples[start:end] where
            negative values are in self.feature_values[start:end_negative[0]]
            and positive values are in
            self.feature_values[start_positive[0]:end].

        is_samples_sorted : bint*,
            If is_samples_sorted, then self.sorted_samples[start:end] will be
            the sorted version of self.samples[start:end].

        """
        cdef SIZE_t indptr_start = self.X_indptr[feature],
        cdef SIZE_t indptr_end = self.X_indptr[feature + 1]
        cdef SIZE_t n_indices = <SIZE_t>(indptr_end - indptr_start)
        cdef SIZE_t n_samples = self.end - self.start

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if ((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < 0.1 * n_indices):
            extract_nnz_binary_search(self.X_indices, self.X_data,
                                      indptr_start, indptr_end,
                                      self.samples, self.start, self.end,
                                      self.index_to_samples,
                                      self.feature_values,
                                      end_negative, start_positive,
                                      self.sorted_samples, is_samples_sorted)

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(self.X_indices, self.X_data,
                                         indptr_start, indptr_end,
                                         self.samples, self.start, self.end,
                                         self.index_to_samples,
                                         self.feature_values,
                                         end_negative, start_positive)


cdef int compare_SIZE_t(const void* a, const void* b) nogil:
    """Comparison function for sort."""
    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])


cdef inline void binary_search(SIZE_t* sorted_array,
                               SIZE_t start, SIZE_t end,
                               SIZE_t value, SIZE_t* index,
                               SIZE_t* new_start) nogil:
    """Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef SIZE_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(SIZE_t* X_indices,
                                              DTYPE_t* X_data,
                                              SIZE_t indptr_start,
                                              SIZE_t indptr_end,
                                              SIZE_t* samples,
                                              SIZE_t start,
                                              SIZE_t end,
                                              SIZE_t* index_to_samples,
                                              DTYPE_t* Xf,
                                              SIZE_t* end_negative,
                                              SIZE_t* start_positive) nogil:
    """Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    cdef SIZE_t k
    cdef SIZE_t index
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(SIZE_t* X_indices,
                                           DTYPE_t* X_data,
                                           SIZE_t indptr_start,
                                           SIZE_t indptr_end,
                                           SIZE_t* samples,
                                           SIZE_t start,
                                           SIZE_t end,
                                           SIZE_t* index_to_samples,
                                           DTYPE_t* Xf,
                                           SIZE_t* end_negative,
                                           SIZE_t* start_positive,
                                           SIZE_t* sorted_samples,
                                           bint* is_samples_sorted) nogil:
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef SIZE_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(sorted_samples + start, samples + start,
               n_samples * sizeof(SIZE_t))
        qsort(sorted_samples + start, n_samples, sizeof(SIZE_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef SIZE_t p = start
    cdef SIZE_t index
    cdef SIZE_t k
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
             # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(SIZE_t* index_to_samples, SIZE_t* samples,
                             SIZE_t pos_1, SIZE_t pos_2) nogil:
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] =  samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef class BestSparseSplitter(BaseSparseSplitter):
    """Splitter for finding the best split, using the sparse data."""

    def __reduce__(self):
        return (BestSparseSplitter, (self.criterion,
                                     self.max_features,
                                     self.min_samples_leaf,
                                     self.min_weight_leaf,
                                     self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil:
        """Find the best split on node samples[start:end], using sparse features

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* X_indices = self.X_indices
        cdef SIZE_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value

        cdef SIZE_t p_next
        cdef SIZE_t p_prev
        cdef bint is_samples_sorted = 0  # indicate is sorted_samples is
                                         # inititialized

        # We assume implicitly that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[f_j], features[n_drawn_constants] = features[n_drawn_constants], features[f_j]

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]
                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Sort the positive and negative parts of `Xf`
                sort(Xf + start, samples + start, end_negative - start)
                sort(Xf + start_positive, samples + start_positive,
                     end - start_positive)

                # Update index_to_samples to take into account the sort
                for p in range(start, end_negative):
                    index_to_samples[samples[p]] = p
                for p in range(start_positive, end):
                    index_to_samples[samples[p]] = p

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0

                    if end_negative != start_positive:
                        Xf[end_negative] = 0
                        end_negative += 1

                if Xf[end - 1] <= Xf[start]:
                    features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        if p + 1 != end_negative:
                            p_next = p + 1
                        else:
                            p_next = start_positive

                        while (p_next < end and
                               Xf[p_next] <= Xf[p]):
                            p = p_next
                            if p + 1 != end_negative:
                                p_next = p + 1
                            else:
                                p_next = start_positive


                        # (p_next >= end) or (X[samples[p_next], current.feature] >
                        #                     X[samples[p], current.feature])
                        p_prev = p
                        p = p_next
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p_prev], current.feature])


                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves used to avoid infinite values
                                current.threshold = <DTYPE_t>(Xf[p - 1] / 2.0 + Xf[p] / 2.0)

                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p_prev]

                                best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


cdef class RandomSparseSplitter(BaseSparseSplitter):
    """Splitter for finding a random split, using the sparse data."""

    def __reduce__(self):
        return (RandomSparseSplitter, (self.criterion,
                                       self.max_features,
                                       self.min_samples_leaf,
                                       self.min_weight_leaf,
                                       self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil:
        """Find a random split on node samples[start:end], using sparse features

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* X_indices = self.X_indices
        cdef SIZE_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef DTYPE_t current_feature_value

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t partition_end

        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value

        cdef bint is_samples_sorted = 0  # indicate that sorted_samples is
                                         # inititialized

        # We assume implicitly that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[f_j], features[n_drawn_constants] = features[n_drawn_constants], features[f_j]

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0

                    if end_negative != start_positive:
                        Xf[end_negative] = 0
                        end_negative += 1

                # Find min, max in Xf[start:end_negative]
                min_feature_value = Xf[start]
                max_feature_value = min_feature_value

                for p in range(start, end_negative):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                # Update min, max given Xf[start_positive:end]
                for p in range(start_positive, end):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = <DTYPE_t>(rand_uniform(min_feature_value,
                                                              max_feature_value,
                                                              random_state))

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    current.pos = self._partition(current.threshold,
                                                  end_negative,
                                                  start_positive,
                                                  start_positive +
                                                  (Xf[start_positive] == 0))

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current.improvement = self.criterion.impurity_improvement(impurity)

                        self.criterion.children_impurity(&current.impurity_left,
                                                         &current.impurity_right)
                        best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                self.extract_nnz(best.feature, &end_negative, &start_positive,
                                 &is_samples_sorted)

                self._partition(best.threshold, end_negative, start_positive,
                                best.pos)

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0

# deepforest/tree/_tree.pxd
# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pxd


import numpy as np
cimport numpy as np

ctypedef np.npy_uint8 DTYPE_t               # Type of X (after binning)
ctypedef np.npy_float64 DOUBLE_t            # Type of y, sample_weight
ctypedef np.npy_int32 SIZE_t                # Type for counters, child, and feature ID
ctypedef np.npy_uint32 UINT32_t             # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the internal nodes in a Tree object (

    SIZE_t left_child                        # ID of the left child of the node
    SIZE_t right_child                       # ID of the right child of the node
    SIZE_t feature                           # Feature used for splitting the node
    DTYPE_t threshold                        # Threshold value at the node

cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions.

    # Input/Output layout
    cdef public SIZE_t n_features           # Number of features in X
    cdef SIZE_t* n_classes                  # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs            # Number of outputs in y
    cdef public SIZE_t max_n_classes        # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth            # Max depth of the tree
    cdef public SIZE_t internal_node_count  # Counter for internal node IDs
    cdef public SIZE_t leaf_node_count      # Counter for leaf node IDS
    cdef public SIZE_t internal_capacity    # Capacity of internal nodes
    cdef public SIZE_t leaf_capacity        # Capacity of leaf nodes
    cdef Node* nodes                        # Array of internal nodes
    cdef double* value                      # Array of leaf nodes
    cdef SIZE_t value_stride                # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _upd_parent(self, SIZE_t parent, bint is_left) except -1 nogil
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, DTYPE_t threshold) except -1 nogil
    cdef int _resize(self, SIZE_t internal_capacity,
                     SIZE_t leaf_capacity) except -1 nogil
    cdef int _resize_node_c(self, SIZE_t internal_capacity=*) except -1 nogil
    cdef int _resize_value_c(self, SIZE_t leaf_capacity=*) except -1 nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, object X)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply_dense(self, object X)
    cdef np.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.

    cdef Splitter splitter              # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf node
    cdef double min_weight_leaf         # Minimum weight in a leaf node
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_split
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray X_idx_sorted=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)

# deepforest/tree/_tree.pyx
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This class is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx


from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy


from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import uint8 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold'],
    'formats': [np.int32, np.int32, np.int32, np.uint8],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold
    ]
})

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_internal_capacity
        cdef int init_leaf_capacity

        if tree.max_depth <= 10:
            init_internal_capacity = (2 ** (tree.max_depth + 1)) - 1
            init_leaf_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_internal_capacity = 2047
            init_leaf_capacity = 2047

        tree._resize(init_internal_capacity, init_leaf_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            # {start, end, depth, parent, is_left, impurity, n_constant_features}
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1: out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                if not is_leaf:
                    # Add internal nodes
                    node_id = tree._add_node(parent, is_left, is_leaf,
                                             split.feature, split.threshold)

                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break
                else:
                    # Update the parent nodes of leaf nodes
                    node_id = tree._upd_parent(parent, is_left)

                    # Set values for leaf nodes
                    splitter.node_value(tree.value + node_id * tree.value_stride)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0 and tree.internal_node_count > 0:
                rc = tree._resize_node_c(tree.internal_node_count)

            if rc >= 0:
                rc = tree._resize_value_c(tree.leaf_node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    internal_node_count : int
        The number of internal nodes in the tree.

    internal_capacity : int
        The current capacity (i.e., size) of the array that stores internal
        nodes, which is at least as great as `internal_node_count`.

    leaf_node_count : int
        The number of leaf nodes in the tree.

    leaf_capacity : int
        The current capacity (i.e., size) of the array that stores leaf
        nodes, which is at least as great as `leaf_capacity`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [internal_node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [internal_node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [internal_node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [internal_node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [leaf_node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each leaf node.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.internal_node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.internal_node_count]

    property n_internals:
        def __get__(self):
            return self.internal_node_count

    property n_leaves:
        def __get__(self):
            return self.leaf_node_count

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.internal_node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.internal_node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.leaf_node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.internal_node_count = 0
        self.leaf_node_count = 0
        self.internal_capacity = 0
        self.leaf_capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["max_depth"] = self.max_depth
        d["internal_node_count"] = self.internal_node_count
        d["leaf_node_count"] = self.leaf_node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.internal_node_count = d["internal_node_count"]
        self.leaf_node_count = d["leaf_node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (value_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        if self._resize_value_c(self.leaf_node_count) != 0:
            raise MemoryError("Failure on resizing leaf nodes to %d" %
                              self.leaf_node_count)

        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.leaf_node_count * self.value_stride * sizeof(double))

        if self._resize_node_c(self.internal_node_count) != 0:
            raise MemoryError("Failure on resizing internal nodes to %d" %
                              self.internal_node_count)

        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.internal_node_count * sizeof(Node))

    cdef int _resize(self, SIZE_t internal_capacity,
                     SIZE_t leaf_capacity) except -1 nogil:
        """Resize `self.nodes` to `internal_capacity`, and resize `self.value`
        to `leaf_capacity`.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        if self._resize_node_c(internal_capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError("Failure on resizing internal nodes to %d" %
                                  internal_capacity)

        if self._resize_value_c(leaf_capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError("Failure on resizing leaf nodes to %d" %
                                  leaf_capacity)

    cdef int _resize_node_c(self,
                            SIZE_t internal_capacity=SIZE_MAX) except -1 nogil:
        """Resize `self.nodes` to `internal_capacity`.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """

        if internal_capacity == self.internal_capacity and self.nodes != NULL:
            return 0

        if internal_capacity == SIZE_MAX:
            if self.internal_capacity == 0:
                internal_capacity = 3  # default initial value
            else:
                internal_capacity = 2 * self.internal_capacity

        safe_realloc(&self.nodes, internal_capacity)

        if internal_capacity < self.internal_node_count:
            self.internal_node_count = internal_capacity

        self.internal_capacity = internal_capacity
        return 0

    cdef int _resize_value_c(self,
                             SIZE_t leaf_capacity=SIZE_MAX) except -1 nogil:
        """Resize `self.value` to `leaf_capacity`.
        
        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """

        if leaf_capacity == self.leaf_capacity and self.value != NULL:
            return 0

        if leaf_capacity == SIZE_MAX:
            if self.leaf_capacity == 0:
                leaf_capacity = 3  # default initial value
            else:
                leaf_capacity = 2 * self.leaf_capacity

        safe_realloc(&self.value, leaf_capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if leaf_capacity > self.leaf_capacity:
            memset(<void*>(self.value + self.leaf_capacity * self.value_stride),
                   0, (leaf_capacity - self.leaf_capacity) *
                   self.value_stride * sizeof(double))

        if leaf_capacity < self.leaf_node_count:
            self.leaf_node_count = leaf_capacity

        self.leaf_capacity = leaf_capacity
        return 0

    cdef SIZE_t _upd_parent(self, SIZE_t parent, bint is_left) except -1 nogil:
        """Add a leaf node to the tree and connect it with its parent. Notice
        that `self.nodes` does not store any information on leaf nodes except
        the id of leaf nodes. In addition, the id of leaf nodes are multiplied
        by `_TREE_LEAF` to distinguish them from the id of internal nodes.
        
        The generated node id will be used to set `self.value` later.
        
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.leaf_node_count

        if node_id >= self.leaf_capacity:
            if self._resize_value_c() != 0:
                return SIZE_MAX

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = _TREE_LEAF * node_id
            else:
                self.nodes[parent].right_child = _TREE_LEAF * node_id

        self.leaf_node_count += 1

        return node_id

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, DTYPE_t threshold) except -1 nogil:
        """Add an internal node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.internal_node_count

        if node_id >= self.internal_capacity:
            if self._resize_node_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        # left_child and right_child will be set later
        node.feature = feature
        node.threshold = threshold

        self.internal_node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0, mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.uint8, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.int32)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t node_id = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                node_id = 0

                # While one of the two children of the current node is not a
                # leaf node
                while node.left_child > 0 or node.right_child > 0:

                    # If the left child is a leaf node
                    if node.left_child <= 0:

                        # If X[i] should be assigned to the left child
                        if X_ndarray[i, node.feature] <= node.threshold:
                            out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.left_child)
                            break
                        else:
                            node_id = node.right_child
                            node = &self.nodes[node.right_child]
                            continue

                    # If the right child is a leaf node
                    if node.right_child <= 0:

                        # If X[i] should be assigned to the right child
                        if X_ndarray[i, node.feature] > node.threshold:
                            out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.right_child)
                            break
                        else:
                            node_id = node.left_child
                            node = &self.nodes[node.left_child]
                            continue

                    # If the left and right child are both internal nodes
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node_id = node.left_child
                        node = &self.nodes[node.left_child]
                    else:
                        node_id = node.right_child
                        node = &self.nodes[node.right_child]

                # If the left and child child are both leaf nodes
                if node.left_child <= 0 and node.right_child <= 0:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.left_child)
                    else:
                        out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.right_child)

        return out

    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef SIZE_t* X_indices = <SIZE_t*>X_indices_ndarray.data
        cdef SIZE_t* X_indptr = <SIZE_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = <DTYPE_t>(X_sample[node.feature])

                    else:
                        feature_value = 0

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef SIZE_t* X_indices = <SIZE_t*>X_indices_ndarray.data
        cdef SIZE_t* X_indptr = <SIZE_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = <DTYPE_t>(X_sample[node.feature])

                    else:
                        feature_value = 0

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.leaf_node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes

        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.internal_node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

# deepforest/tree/_utils.pxd
# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_utils.pxd


import numpy as np
cimport numpy as np
from ._tree cimport Node

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_int32 SIZE_t             # Type for counters, child, and feature ID
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Node**)
    (StackRecord*)
    (PriorityHeapRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except * nogil


cdef np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)


cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil


cdef double rand_uniform(double low, double high,
                         UINT32_t* random_state) nogil


cdef double log(double x) nogil

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features

cdef class Stack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) except -1 nogil
    cdef int pop(self, StackRecord* res) nogil


# =============================================================================
# PriorityHeap data structure
# =============================================================================

# A record on the frontier for best-first tree growing
cdef struct PriorityHeapRecord:
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement

cdef class PriorityHeap:
    cdef SIZE_t capacity
    cdef SIZE_t heap_ptr
    cdef PriorityHeapRecord* heap_

    cdef bint is_empty(self) nogil
    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t pos) nogil
    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t pos, SIZE_t heap_length) nogil
    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bint is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right) except -1 nogil
    cdef int pop(self, PriorityHeapRecord* res) nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    DOUBLE_t data
    DOUBLE_t weight

cdef class WeightedPQueue:
    cdef SIZE_t capacity
    cdef SIZE_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) nogil
    cdef int reset(self) except -1 nogil
    cdef SIZE_t size(self) nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef int peek(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef DOUBLE_t get_weight_from_index(self, SIZE_t index) nogil
    cdef DOUBLE_t get_value_from_index(self, SIZE_t index) nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef SIZE_t initial_capacity
    cdef WeightedPQueue samples
    cdef DOUBLE_t total_weight
    cdef SIZE_t k
    cdef DOUBLE_t sum_w_0_k            # represents sum(weights[0:k])
                                       # = w[0] + w[1] + ... + w[k-1]

    cdef SIZE_t size(self) nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int update_median_parameters_post_push(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) nogil
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef int update_median_parameters_post_remove(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) nogil
    cdef DOUBLE_t get_median(self) nogil

# deepforest/tree/_utils.pyx

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_utils.pyx


from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln

import numpy as np
cimport numpy as np
np.import_array()

# =============================================================================
# Helper functions
# =============================================================================

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except * nogil:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


def _realloc_test():
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


# rand_r replacement using a 32bit XorShift generator
# From https://github.com/Yu-Group/iterative-Random-Forest
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Return copied data as 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data).copy()


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """Generate a random double in [low; high)."""
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


# =============================================================================
# Stack data structure
# =============================================================================

cdef class Stack:
    """A LIFO data structure.

    Attributes
    ----------
    capacity : SIZE_t
        The elements the stack can hold; if more added then ``self.stack_``
        needs to be resized.

    top : SIZE_t
        The number of elements currently on the stack.

    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) except -1 nogil:
        """Push a new element onto the stack.

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.stack_, self.capacity)

        stack = self.stack_
        stack[top].start = start
        stack[top].end = end
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].impurity = impurity
        stack[top].n_constant_features = n_constant_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """Remove the top element from the stack and copy to ``res``.

        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0


# =============================================================================
# PriorityHeap data structure
# =============================================================================

cdef class PriorityHeap:
    """A priority queue implemented as a binary heap.

    The heap invariant is that the impurity improvement of the parent record
    is larger then the impurity improvement of the children.

    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the heap

    heap_ptr : SIZE_t
        The water mark of the heap; the heap grows from left to right in the
        array ``heap_``. The following invariant holds ``heap_ptr < capacity``.

    heap_ : PriorityHeapRecord*
        The array of heap records. The maximum element is on the left;
        the heap grows from left to right
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.heap_ptr = 0
        safe_realloc(&self.heap_, capacity)

    def __dealloc__(self):
        free(self.heap_)

    cdef bint is_empty(self) nogil:
        return self.heap_ptr <= 0

    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t pos) nogil:
        """Restore heap invariant parent.improvement > child.improvement from
           ``pos`` upwards. """
        if pos == 0:
            return

        cdef SIZE_t parent_pos = (pos - 1) / 2

        if heap[parent_pos].improvement < heap[pos].improvement:
            heap[parent_pos], heap[pos] = heap[pos], heap[parent_pos]
            self.heapify_up(heap, parent_pos)

    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t pos,
                           SIZE_t heap_length) nogil:
        """Restore heap invariant parent.improvement > children.improvement from
           ``pos`` downwards. """
        cdef SIZE_t left_pos = 2 * (pos + 1) - 1
        cdef SIZE_t right_pos = 2 * (pos + 1)
        cdef SIZE_t largest = pos

        if (left_pos < heap_length and
                heap[left_pos].improvement > heap[largest].improvement):
            largest = left_pos

        if (right_pos < heap_length and
                heap[right_pos].improvement > heap[largest].improvement):
            largest = right_pos

        if largest != pos:
            heap[pos], heap[largest] = heap[largest], heap[pos]
            self.heapify_down(heap, largest, heap_length)

    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
                  SIZE_t depth, bint is_leaf, double improvement,
                  double impurity, double impurity_left,
                  double impurity_right) except -1 nogil:
        """Push record on the priority heap.

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = NULL

        # Resize if capacity not sufficient
        if heap_ptr >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap_, self.capacity)

        # Put element as last element of heap
        heap = self.heap_
        heap[heap_ptr].node_id = node_id
        heap[heap_ptr].start = start
        heap[heap_ptr].end = end
        heap[heap_ptr].pos = pos
        heap[heap_ptr].depth = depth
        heap[heap_ptr].is_leaf = is_leaf
        heap[heap_ptr].impurity = impurity
        heap[heap_ptr].impurity_left = impurity_left
        heap[heap_ptr].impurity_right = impurity_right
        heap[heap_ptr].improvement = improvement

        # Heapify up
        self.heapify_up(heap, heap_ptr)

        # Increase element count
        self.heap_ptr = heap_ptr + 1
        return 0

    cdef int pop(self, PriorityHeapRecord* res) nogil:
        """Remove max element from the heap. """
        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = self.heap_

        if heap_ptr <= 0:
            return -1

        # Take first element
        res[0] = heap[0]

        # Put last element to the front
        heap[0], heap[heap_ptr - 1] = heap[heap_ptr - 1], heap[0]

        # Restore heap invariant
        if heap_ptr > 1:
            self.heapify_down(heap, 0, heap_ptr - 1)

        self.heap_ptr = heap_ptr - 1

        return 0

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

cdef class WeightedPQueue:
    """A priority queue class, always sorted in increasing order.

    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the priority queue.

    array_ptr : SIZE_t
        The water mark of the priority queue; the priority queue grows from
        left to right in the array ``array_``. ``array_ptr`` is always
        less than ``capacity``.

    array_ : WeightedPQueueRecord*
        The array of priority queue records. The minimum element is on the
        left at index 0, and the maximum element is on the right at index
        ``array_ptr-1``.
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.array_ptr = 0
        safe_realloc(&self.array_, capacity)

    def __dealloc__(self):
        free(self.array_)

    cdef int reset(self) except -1 nogil:
        """Reset the WeightedPQueue to its state at construction

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.array_ptr = 0
        # Since safe_realloc can raise MemoryError, use `except *`
        safe_realloc(&self.array_, self.capacity)
        return 0

    cdef bint is_empty(self) nogil:
        return self.array_ptr <= 0

    cdef SIZE_t size(self) nogil:
        return self.array_ptr

    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil:
        """Push record on the array.

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = NULL
        cdef SIZE_t i

        # Resize if capacity not sufficient
        if array_ptr >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.array_, self.capacity)

        # Put element as last element of array
        array = self.array_
        array[array_ptr].data = data
        array[array_ptr].weight = weight

        # bubble last element up according until it is sorted
        # in ascending order
        i = array_ptr
        while(i != 0 and array[i].data < array[i-1].data):
            array[i], array[i-1] = array[i-1], array[i]
            i -= 1

        # Increase element count
        self.array_ptr = array_ptr + 1
        return 0

    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil:
        """Remove a specific value/weight record from the array.
        Returns 0 if successful, -1 if record not found."""
        cdef SIZE_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = self.array_
        cdef SIZE_t idx_to_remove = -1
        cdef SIZE_t i

        if array_ptr <= 0:
            return -1

        # find element to remove
        for i in range(array_ptr):
            if array[i].data == data and array[i].weight == weight:
                idx_to_remove = i
                break

        if idx_to_remove == -1:
            return -1

        # shift the elements after the removed element
        # to the left.
        for i in range(idx_to_remove, array_ptr-1):
            array[i] = array[i+1]

        self.array_ptr = array_ptr - 1
        return 0

    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil:
        """Remove the top (minimum) element from array.
        Returns 0 if successful, -1 if nothing to remove."""
        cdef SIZE_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = self.array_
        cdef SIZE_t i

        if array_ptr <= 0:
            return -1

        data[0] = array[0].data
        weight[0] = array[0].weight

        # shift the elements after the removed element
        # to the left.
        for i in range(0, array_ptr-1):
            array[i] = array[i+1]

        self.array_ptr = array_ptr - 1
        return 0

    cdef int peek(self, DOUBLE_t* data, DOUBLE_t* weight) nogil:
        """Write the top element from array to a pointer.
        Returns 0 if successful, -1 if nothing to write."""
        cdef WeightedPQueueRecord* array = self.array_
        if self.array_ptr <= 0:
            return -1
        # Take first value
        data[0] = array[0].data
        weight[0] = array[0].weight
        return 0

    cdef DOUBLE_t get_weight_from_index(self, SIZE_t index) nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested weight"""
        cdef WeightedPQueueRecord* array = self.array_

        # get weight at index
        return array[index].weight

    cdef DOUBLE_t get_value_from_index(self, SIZE_t index) nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested value"""
        cdef WeightedPQueueRecord* array = self.array_

        # get value at index
        return array[index].data

# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    """A class to handle calculation of the weighted median from streams of
    data. To do so, it maintains a parameter ``k`` such that the sum of the
    weights in the range [0,k) is greater than or equal to half of the total
    weight. By minimizing the value of ``k`` that fulfills this constraint,
    calculating the median is done by either taking the value of the sample
    at index ``k-1`` of ``samples`` (samples[k-1].data) or the average of
    the samples at index ``k-1`` and ``k`` of ``samples``
    ((samples[k-1] + samples[k]) / 2).

    Attributes
    ----------
    initial_capacity : SIZE_t
        The initial capacity of the WeightedMedianCalculator.

    samples : WeightedPQueue
        Holds the samples (consisting of values and their weights) used in the
        weighted median calculation.

    total_weight : DOUBLE_t
        The sum of the weights of items in ``samples``. Represents the total
        weight of all samples used in the median calculation.

    k : SIZE_t
        Index used to calculate the median.

    sum_w_0_k : DOUBLE_t
        The sum of the weights from samples[0:k]. Used in the weighted
        median calculation; minimizing the value of ``k`` such that
        ``sum_w_0_k`` >= ``total_weight / 2`` provides a mechanism for
        calculating the median in constant time.

    """

    def __cinit__(self, SIZE_t initial_capacity):
        self.initial_capacity = initial_capacity
        self.samples = WeightedPQueue(initial_capacity)
        self.total_weight = 0
        self.k = 0
        self.sum_w_0_k = 0

    cdef SIZE_t size(self) nogil:
        """Return the number of samples in the
        WeightedMedianCalculator"""
        return self.samples.size()

    cdef int reset(self) except -1 nogil:
        """Reset the WeightedMedianCalculator to its state at construction

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # samples.reset (WeightedPQueue.reset) uses safe_realloc, hence
        # except -1
        self.samples.reset()
        self.total_weight = 0
        self.k = 0
        self.sum_w_0_k = 0
        return 0

    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil:
        """Push a value and its associated weight to the WeightedMedianCalculator

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef int return_value
        cdef DOUBLE_t original_median = 0.0

        if self.size() != 0:
            original_median = self.get_median()
        # samples.push (WeightedPQueue.push) uses safe_realloc, hence except -1
        return_value = self.samples.push(data, weight)
        self.update_median_parameters_post_push(data, weight,
                                                original_median)
        return return_value

    cdef int update_median_parameters_post_push(
            self, DOUBLE_t data, DOUBLE_t weight,
            DOUBLE_t original_median) nogil:
        """Update the parameters used in the median calculation,
        namely `k` and `sum_w_0_k` after an insertion"""

        # trivial case of one element.
        if self.size() == 1:
            self.k = 1
            self.total_weight = weight
            self.sum_w_0_k = self.total_weight
            return 0

        # get the original weighted median
        self.total_weight += weight

        if data < original_median:
            # inserting below the median, so increment k and
            # then update self.sum_w_0_k accordingly by adding
            # the weight that was added.
            self.k += 1
            # update sum_w_0_k by adding the weight added
            self.sum_w_0_k += weight

            # minimize k such that sum(W[0:k]) >= total_weight / 2
            # minimum value of k is 1
            while(self.k > 1 and ((self.sum_w_0_k -
                                   self.samples.get_weight_from_index(self.k-1))
                                  >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)
            return 0

        if data >= original_median:
            # inserting above or at the median
            # minimize k such that sum(W[0:k]) >= total_weight / 2
            while(self.k < self.samples.size() and
                  (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)
            return 0

    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil:
        """Remove a value from the MedianHeap, removing it
        from consideration in the median calculation
        """
        cdef int return_value
        cdef DOUBLE_t original_median = 0.0

        if self.size() != 0:
            original_median = self.get_median()

        return_value = self.samples.remove(data, weight)
        self.update_median_parameters_post_remove(data, weight,
                                                  original_median)
        return return_value

    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil:
        """Pop a value from the MedianHeap, starting from the
        left and moving to the right.
        """
        cdef int return_value
        cdef double original_median = 0.0

        if self.size() != 0:
            original_median = self.get_median()

        # no elements to pop
        if self.samples.size() == 0:
            return -1

        return_value = self.samples.pop(data, weight)
        self.update_median_parameters_post_remove(data[0],
                                                  weight[0],
                                                  original_median)
        return return_value

    cdef int update_median_parameters_post_remove(
            self, DOUBLE_t data, DOUBLE_t weight,
            double original_median) nogil:
        """Update the parameters used in the median calculation,
        namely `k` and `sum_w_0_k` after a removal"""
        # reset parameters because it there are no elements
        if self.samples.size() == 0:
            self.k = 0
            self.total_weight = 0
            self.sum_w_0_k = 0
            return 0

        # trivial case of one element.
        if self.samples.size() == 1:
            self.k = 1
            self.total_weight -= weight
            self.sum_w_0_k = self.total_weight
            return 0

        # get the current weighted median
        self.total_weight -= weight

        if data < original_median:
            # removing below the median, so decrement k and
            # then update self.sum_w_0_k accordingly by subtracting
            # the removed weight

            self.k -= 1
            # update sum_w_0_k by removing the weight at index k
            self.sum_w_0_k -= weight

            # minimize k such that sum(W[0:k]) >= total_weight / 2
            # by incrementing k and updating sum_w_0_k accordingly
            # until the condition is met.
            while(self.k < self.samples.size() and
                  (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)
            return 0

        if data >= original_median:
            # removing above the median
            # minimize k such that sum(W[0:k]) >= total_weight / 2
            while(self.k > 1 and ((self.sum_w_0_k -
                                   self.samples.get_weight_from_index(self.k-1))
                                  >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)
            return 0

    cdef DOUBLE_t get_median(self) nogil:
        """Write the median to a pointer, taking into account
        sample weights."""
        if self.sum_w_0_k == (self.total_weight / 2.0):
            # split median
            return (self.samples.get_value_from_index(self.k) +
                    self.samples.get_value_from_index(self.k-1)) / 2.0
        if self.sum_w_0_k > (self.total_weight / 2.0):
            # whole median
            return self.samples.get_value_from_index(self.k-1)

# deepforest/tree/setup.py
import os
import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension(
        "_tree",
        sources=["_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )
    config.add_extension(
        "_splitter",
        sources=["_splitter.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )
    config.add_extension(
        "_criterion",
        sources=["_criterion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )
    config.add_extension(
        "_utils",
        sources=["_utils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())


# deepforest/tree/tree.py
"""
Implementation of the decision tree in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py
"""


__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]

import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import DepthFirstTreeBuilder
from ._tree import Tree
from . import _tree, _splitter, _criterion


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse": _criterion.MSE, "mae": _criterion.MAE}

DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.RandomSplitter,
}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        random_state,
        min_impurity_decrease,
        min_impurity_split,
        class_weight=None,
        presort="deprecated",
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    @property
    def n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    @property
    def n_internals(self):
        """Return the number of internal nodes of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of internal nodes.
        """
        check_is_fitted(self)
        return self.tree_.n_internals

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None
    ):

        random_state = check_random_state(self.random_state)

        if X.dtype != np.uint8:
            msg = "The dtype of `X` should be `np.uint8`, but got {} instead."
            raise RuntimeError(msg.format(X.dtype))

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        # `classes_` and `n_classes_` were set by the forest.
        if not hasattr(self, "classes_") and is_classifier(self):
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=np.int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(
                    y[:, k], return_inverse=True
                )
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.int32)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (
            np.iinfo(np.int32).max
            if self.max_depth is None
            else self.max_depth
        )

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features in ["auto", "sqrt"]:
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_)
                )
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match "
                "number of samples=%d" % (len(y), n_samples)
            )
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(
                sample_weight
            )

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. "
                "Its default value has changed from 1e-7 to 0 in "
                "version 0.23, and it will be removed in 0.25. "
                "Use the min_impurity_decrease parameter instead.",
                FutureWarning,
            )

            if min_impurity_split < 0.0:
                raise ValueError(
                    "min_impurity_split must be greater than " "or equal to 0"
                )
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be greater than " "or equal to 0"
            )

        if self.presort != "deprecated":
            warnings.warn(
                "The parameter 'presort' is deprecated and has no "
                "effect. It will be removed in v0.24. You can "
                "suppress this warning by not passing any value "
                "to the 'presort' parameter.",
                FutureWarning,
            )

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classifier(self):
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](
                    self.n_outputs_, n_samples
                )

        SPLITTERS = DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )

        if is_classifier(self):
            self.tree_ = Tree(
                self.n_features_, self.n_classes_, self.n_outputs_
            )
        else:
            self.tree_ = Tree(
                self.n_features_,
                # TODO: tree should't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.int32),
                self.n_outputs_,
            )

        builder = DepthFirstTreeBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease,
            min_impurity_split,
        )

        builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # Only return the essential data for using a tree for prediction
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        children = np.vstack(
            (self.tree_.children_left, self.tree_.children_right)
        ).T
        value = self.tree_.value

        return feature, threshold, children, value

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_features_, n_features)
            )

        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)

        # Classification
        if is_classifier(self):
            return self.classes_.take(np.argmax(pred, axis=1), axis=0)
        # Regression
        else:
            return np.squeeze(pred)


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort="deprecated",
    ):

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort,
        )

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None
    ):

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
        )

    def predict_proba(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)

        proba = self.tree_.predict(X)
        proba = proba[:, : self.n_classes_]
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba


class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        presort="deprecated",
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state,
        )

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None
    ):

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
        )


class ExtraTreeClassifier(DecisionTreeClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
    ):

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state,
        )


class ExtraTreeRegressor(DecisionTreeRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion="mse",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
    ):

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state,
        )


# deepforest/utils/__init__.py

# deepforest/utils/kfoldwrapper.py

"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold

from .. import _utils


class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    """

    def __init__(
        self,
        estimator,
        n_splits,
        n_outputs,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):

        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = []

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

    def fit_transform(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.oob_decision_function_ = np.zeros((n_samples, self.n_outputs))

        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            estimator = copy.deepcopy(self.dummy_estimator_)

            if self.verbose > 1:
                msg = "{} - - Fitting the base estimator with fold = {}"
                print(msg.format(_utils.ctime(), k))

            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_idx], y[train_idx])
            else:
                estimator.fit(
                    X[train_idx], y[train_idx], sample_weight[train_idx]
                )

            # Predict on hold-out samples
            if self.is_classifier:
                self.oob_decision_function_[
                    val_idx
                ] += estimator.predict_proba(X[val_idx])
            else:
                val_pred = estimator.predict(X[val_idx])

                # Reshape for univariate regression
                if self.n_outputs == 1 and len(val_pred.shape) == 1:
                    val_pred = np.expand_dims(val_pred, 1)
                self.oob_decision_function_[val_idx] += val_pred

            # Store the estimator
            self.estimators_.append(estimator)

        return self.oob_decision_function_

    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, self.n_outputs))  # pre-allocate results
        for estimator in self.estimators_:
            if self.is_classifier:
                out += estimator.predict_proba(X)  # classification
            else:
                if self.n_outputs > 1:
                    out += estimator.predict(X)  # multi-variate regression
                else:
                    out += estimator.predict(X).reshape(
                        n_samples, -1
                    )  # univariate regression

        return out / self.n_splits  # return the average prediction