import numpy as np
import copy
import itertools
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class SmartForest:
    """
    SmartForest: A deep, intelligent decision forest model for complex sequential and tabular data.

    SmartForest blends ideas from deep forests (cascade forest structures), LSTM-style forget gates,
    and ensemble learning using XGBoost. It is especially suited for time series or structured tabular data 
    where layer-wise feature expansion and memory-inspired filtering can enhance performance.

    Key Features:
    -------------
    - Deep cascade of XGBoost regressors
    - Optional Multi-Grained Scanning (MGS) for local feature extraction
    - Forget-gate-inspired mechanism to regulate information flow across layers
    - Early stopping to prevent overfitting
    - Full retention of best-performing model (lowest validation RMSE)

    Parameters:
    -----------
    n_estimators_per_layer : int
        Number of XGBoost regressors per layer.
    
    max_layers : int
        Maximum number of layers (depth) in the model.

    early_stopping_rounds : int
        Number of layers with no improvement before early stopping is triggered.

    param_grid : dict
        Grid of XGBoost hyperparameters to search over.

    use_gpu : bool
        If True, use GPU-accelerated training (CUDA required).

    gpu_id : int
        ID of GPU to use (if use_gpu=True).

    window_sizes : list of int
        Enables Multi-Grained Scanning if non-empty, with specified sliding window sizes.

    forget_factor : float in [0, 1]
        Simulates LSTM-style forget gate; higher values forget more past information.

    verbose : int
        Verbosity level (0 = silent, 1 = progress updates).

    Methods:
    --------
    fit(X, y, X_val=None, y_val=None):
        Train the SmartForest model layer by layer, using optional validation for early stopping.

    predict(X):
        Make predictions on new data using the trained cascade structure.

    get_best_model():
        Returns a copy of the best model and the corresponding RMSE from validation.

    Example:
    --------
    >>> model = SmartForest(n_estimators_per_layer=5, max_layers=10, window_sizes=[2, 3], forget_factor=0.2)
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> y_pred = model.predict(X_val)
    >>> best_model, best_rmse = model.get_best_model()
    """
    def __init__(self, n_estimators_per_layer = 5, max_layers = 10, early_stopping_rounds = 3, param_grid = None,
                 use_gpu = False, gpu_id = 0, window_sizes = [], forget_factor = 0, verbose = 1):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.param_grid = param_grid or {
            "objective": ["reg:squarederror"],
            "random_state": [42],
            'seed': [0],
            'n_estimators': [100],
            'max_depth': [6],
            'min_child_weight': [4],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0],
            'reg_alpha': [0],
            'reg_lambda': [1],
            'learning_rate': [0.05],
        }
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.window_sizes = window_sizes
        self.forget_factor = forget_factor
        self.layers = []
        self.best_model = None
        self.best_rmse = float("inf")
        self.verbose = verbose

    def _get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _multi_grained_scanning(self, X, y):
        new_features = []
        for window_size in self.window_sizes:
            if X.shape[1] < window_size:
                continue
            for start in range(X.shape[1] - window_size + 1):
                window = X[:, start:start + window_size]
                if y is None:
                    new_features.append(window)
                    continue

                param_combos = self._get_param_combinations()
                for params in param_combos:
                    if self.use_gpu:
                        params['tree_method'] = 'hist'
                        params['device'] = 'cuda'
                    model = XGBRegressor(**params)
                    model.fit(window, y)
                    preds = model.predict(window).reshape(-1, 1)
                    new_features.append(preds)
        return np.hstack(new_features) if new_features else X

    def _apply_forget_gate(self, X, layer_index):
        forget_weights = np.random.rand(X.shape[1]) * self.forget_factor
        return X * (1 - forget_weights)

    def _fit_layer(self, X, y, X_val=None, y_val=None, layer_index=0):
        layer = []
        layer_outputs = []
        param_combos = self._get_param_combinations()
        X = self._apply_forget_gate(X, layer_index)

        for i in range(self.n_estimators_per_layer):
            best_rmse = float('inf')
            best_model = None

            for params in param_combos:
                if self.use_gpu:
                    params['tree_method'] = 'hist'
                    params['device'] = 'cuda'

                params = params.copy()  # Prevent modification from affecting the next loop iteration
                params['random_state'] = i  # Use a different random seed for each model to enhance diversity

                model = XGBRegressor(**params)
                model.fit(X, y)

                if X_val is not None:
                    preds_val = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                else:
                    best_model = model

            preds = best_model.predict(X).reshape(-1, 1)
            layer.append(best_model)
            layer_outputs.append(preds)

        output = np.hstack(layer_outputs)
        return layer, output

    def fit(self, X, y, X_val=None, y_val=None):
        X_current = self._multi_grained_scanning(X, y)
        X_val_current = self._multi_grained_scanning(X_val, y_val) if X_val is not None else None
        no_improve_rounds = 0

        for layer_index in range(self.max_layers):
            if self.verbose: print(f"Training Layer {layer_index + 1}")
            layer, output = self._fit_layer(X_current, y, X_val_current, y_val, layer_index)
            self.layers.append(layer)
            X_current = np.hstack([X_current, output])

            if X_val is not None:
                val_outputs = []
                for reg in layer:
                    n_features = reg.n_features_in_
                    preds = reg.predict(X_val_current[:, :n_features]).reshape(-1, 1)
                    val_outputs.append(preds)
                val_output = np.hstack(val_outputs)
                X_val_current = np.hstack([X_val_current, val_output])

                y_pred = self.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                if self.verbose: print(f"Validation RMSE: {rmse:.4f}")

                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_model = copy.deepcopy(self.layers)
                    no_improve_rounds = 0
                    if self.verbose: print(f"✅ New best RMSE: {self.best_rmse:.4f}")
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        if self.verbose: print("Early stopping triggered.")
                        break

    def predict(self, X):
        X_current = self._multi_grained_scanning(X, None)
        X_current = self._apply_forget_gate(X_current, layer_index=-1)

        for layer in self.layers:
            layer_outputs = []
            for reg in layer:
                n_features = reg.n_features_in_
                preds = reg.predict(X_current[:, :n_features]).reshape(-1, 1)
                layer_outputs.append(preds)
            output = np.hstack(layer_outputs)
            X_current = np.hstack([X_current, output])

        final_outputs = []
        for reg in self.layers[-1]:
            n_features = reg.n_features_in_
            final_outputs.append(reg.predict(X_current[:, :n_features]).reshape(-1, 1))
        return np.mean(np.hstack(final_outputs), axis=1)

    def get_best_model(self):
        return self.best_model, self.best_rmse

"""
# ============================== Test Example ==============================
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# X, y = load_diabetes(return_X_y=True) # Using diabetes dataset
X, y = fetch_california_housing(return_X_y=True) # Using house price dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter grid
param_grid = {
    "objective": ["reg:squarederror"],
    "random_state": [42],
    'seed': [0],
    'n_estimators': [100],
    'max_depth': [6],
    'min_child_weight': [4],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0],
    'reg_alpha': [0],
    'reg_lambda': [1],
    'learning_rate': [0.05],
}

# Create the model with Multi-Grained Scanning enabled (with window sizes 2 and 3)
regr = SmartForest(
    n_estimators_per_layer = 5,
    max_layers = 10,
    early_stopping_rounds = 5,
    param_grid = param_grid,
    use_gpu = False,
    gpu_id = 0,
    window_sizes = [],  # Enables MGS if e.g., [2, 3], else empty disables MGS.
    forget_factor = 0.,  # Set forget factor to simulate forget gate behavior
    verbose = 1
)

regr.fit(X_train, y_train, X_val, y_val)

# Predict on validation set and evaluate
y_pred = regr.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("\nFinal RMSE:", rmse)

# Output best model and RMSE
best_model, best_rmse = regr.get_best_model()
print("\nBest validation RMSE:", best_rmse)
"""

# ============================================================================================================================================================

import numpy as np
import copy
import itertools
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class SmartForest4D:
    """
    SmartForest4D is an ensemble learning model designed to handle complex 4D input data 
    (samples, time, spatial, features). It integrates ideas from gradient-boosted decision trees 
    (XGBoost) with LSTM-style forget gates and spatial max pooling.

    The model builds layers of regressors, each layer taking the previous output as part of its 
    input (deep forest style). A forget gate mechanism is applied along the time dimension to 
    emphasize recent temporal information. Spatial max pooling is used to reduce dimensionality 
    across spatial units before flattening and feeding into the regressors.

    Parameters:
    -----------
    n_estimators_per_layer : int
        Number of XGBoost regressors per layer.

    max_layers : int
        Maximum number of layers in the deep forest.

    early_stopping_rounds : int
        Number of rounds without improvement on the validation set before early stopping.

    param_grid : dict
        Dictionary of hyperparameter lists to search over for XGBoost.

    use_gpu : bool
        Whether to use GPU for training XGBoost models.

    gpu_id : int
        GPU device ID to use if use_gpu is True.

    kernel: np.ndarray
        Convolutional kernel for spatial processing.
        # ===============================
        # 0. Do nothing
        # ===============================

        identity_kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        # ===============================
        # 1. Sobel Edge Detection Kernels
        # ===============================

        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        # ===============================
        # 2. Gaussian Blur Kernel (3x3)
        # ===============================
        gaussian_kernel = (1/16) * np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ])

        # ===============================
        # 3. Morphological Structuring Element (3x3 cross)
        # Used in binary dilation/erosion
        # ===============================
        morph_kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        # ===============================
        # 4. Sharpening Kernel
        # Enhances edges and contrast
        # ===============================
        sharpen_kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])

        # ===============================
        # 5. Embossing Kernel
        # Creates a 3D-like shadowed effect
        # ===============================
        emboss_kernel = np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ])

    spatial_h : int
        The height of the 2D grid for the flattened spatial dimension.

    spatial_w : int
        The width of the 2D grid for the flattened spatial dimension.
    
    forget_factor : float
        Exponential decay rate applied along the time axis. Higher values mean stronger forgetting.

    verbose : int
        Verbosity level for training output.

    Attributes:
    -----------
    layers : list
        List of trained layers, each containing a list of regressors.

    best_model : list
        The set of layers corresponding to the best validation RMSE seen during training.

    best_rmse : float
        The lowest RMSE achieved on the validation set.

    Methods:
    --------
    fit(X, y, X_val=None, y_val=None):
        Train the SmartForest4D model on the given 4D input data.

    predict(X):
        Predict targets for new 4D input data using the trained model.

    get_best_model():
        Return the best set of layers and corresponding RMSE.

    Notes:
    ------
    - The product of spatial_h and spatial_w must equal spatial_size (spatial_h * spatial_w = spatial_size).

    Example:
    --------
    >>> model = SmartForest4D(n_estimators_per_layer=5, max_layers=10, early_stopping_rounds=3, forget_factor=0.3, verbose=1)
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> y_pred = model.predict(X_val)
    >>> best_model, best_rmse = model.get_best_model()
    """
    def __init__(self, n_estimators_per_layer=5, max_layers=10, early_stopping_rounds=3, param_grid=None,
                 use_gpu=False, gpu_id=0, kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), spatial_h=None, spatial_w=None,
                 forget_factor=0.0, verbose=1):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.param_grid = param_grid or {
            "objective": ["reg:squarederror"],
            "random_state": [42],
            'n_estimators': [100],
            'max_depth': [6],
            'min_child_weight': [4],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0],
            'reg_alpha': [0],
            'reg_lambda': [1],
            'learning_rate': [0.05],
        }
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.kernel = kernel
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.forget_factor = forget_factor
        self.layers = []
        self.best_model = None
        self.best_rmse = float("inf")
        self.verbose = verbose
        if (self.spatial_h is None) or (self.spatial_w is None): 
            raise ValueError("Please specify spatial_h and spatial_w")

    def _get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _prepare_input(self, X, y=None, apply_forget=False, layer_index=0):
        # Ensure 4D: (samples, time, spatial, features)
        if X.ndim == 2:
            X = X[:, np.newaxis, np.newaxis, :]
        elif X.ndim == 3:
            X = X[:, :, np.newaxis, :]
        elif X.ndim == 4:
            pass
        else:
            raise ValueError("Input must be 2D, 3D, or 4D.")

        n_samples, n_time, n_spatial, n_features = X.shape

        if apply_forget and self.forget_factor > 0:
            decay = np.exp(-self.forget_factor * np.arange(n_time))[::-1]
            decay = decay / decay.sum()
            decay = decay.reshape(1, n_time, 1, 1)
            X = X * decay

        # Apply convolutional kernels:
        if n_spatial != 1:
            if self.spatial_h * self.spatial_w != n_spatial: raise ValueError("spatial_h * spatial_w != n_spatial")
            X_out = np.zeros_like(X)
            for sample in range(X.shape[0]):
                for t in range(X.shape[1]):
                    for f in range(X.shape[3]):
                        spatial_2d = X[sample, t, :, f].reshape(self.spatial_h, self.spatial_w)
                        # Apply 2D convolution
                        filtered = ndimage.convolve(spatial_2d, self.kernel, mode='constant', cval=0.0)
                        # Flatten back to (20,) and store
                        X_out[sample, t, :, f] = filtered.reshape(n_spatial)
            X = X_out; del(X_out)
        # Max pooling over spatial dim
        X_pooled = X.max(axis=2)  # (samples, time, features)
        X_flattened = X_pooled.reshape(n_samples, -1)  # (samples, time * features)
        return X_flattened

    def _fit_layer(self, X, y, X_val=None, y_val=None, layer_index=0):
        layer = []
        layer_outputs = []
        param_combos = self._get_param_combinations()

        for i in range(self.n_estimators_per_layer):
            best_rmse = float('inf')
            best_model = None

            for params in param_combos:
                if self.use_gpu:
                    params['tree_method'] = 'hist'
                    params['device'] = 'cuda'

                params = params.copy()
                params['random_state'] = i

                model = XGBRegressor(**params)
                model.fit(X, y)

                if X_val is not None:
                    preds_val = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                else:
                    best_model = model

            preds = best_model.predict(X).reshape(-1, 1)
            layer.append(best_model)
            layer_outputs.append(preds)

        output = np.hstack(layer_outputs)
        return layer, output

    def fit(self, X, y, X_val=None, y_val=None):
        y = y.ravel()
        X_current = self._prepare_input(X, apply_forget=True)
        X_val_current = self._prepare_input(X_val, apply_forget=True) if X_val is not None else None

        no_improve_rounds = 0

        for layer_index in range(self.max_layers):
            if self.verbose:
                print(f"Training Layer {layer_index + 1}")

            layer, output = self._fit_layer(X_current, y, X_val_current, y_val, layer_index)
            self.layers.append(layer)
            X_current = np.hstack([X_current, output])

            if X_val is not None:
                val_outputs = []
                for reg in layer:
                    n_features = reg.n_features_in_
                    preds = reg.predict(X_val_current[:, :n_features]).reshape(-1, 1)
                    val_outputs.append(preds)
                val_output = np.hstack(val_outputs)
                X_val_current = np.hstack([X_val_current, val_output])

                y_pred = self.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                if self.verbose:
                    print(f"Validation RMSE: {rmse:.4f}")

                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_model = copy.deepcopy(self.layers)
                    no_improve_rounds = 0
                    if self.verbose:
                        print(f"✅ New best RMSE: {self.best_rmse:.4f}")
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

    def predict(self, X):
        X_current = self._prepare_input(X, apply_forget=True)

        for layer in self.layers:
            layer_outputs = []
            for reg in layer:
                n_features = reg.n_features_in_
                preds = reg.predict(X_current[:, :n_features]).reshape(-1, 1)
                layer_outputs.append(preds)
            output = np.hstack(layer_outputs)
            X_current = np.hstack([X_current, output])

        final_outputs = []
        for reg in self.layers[-1]:
            n_features = reg.n_features_in_
            final_outputs.append(reg.predict(X_current[:, :n_features]).reshape(-1, 1))
        return np.mean(np.hstack(final_outputs), axis=1)

    def get_best_model(self):
        return self.best_model, self.best_rmse

"""
# ============================== Test Example ==============================
import numpy as np
import copy
import itertools
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic 4D data: (samples, time, spatial, features)
# time order is like [t (today), t - 1 (yesterday), t -2, ...]
n_samples = 200
n_time = 5
n_spatial = 4
n_features = 5

np.random.seed(42)
X = np.random.rand(n_samples, n_time, n_spatial, n_features)
y = X[:, :3, :2, :4].mean(axis=(1, 2, 3)) + 0.1 * np.random.randn(n_samples)
y = y.ravel()

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SmartForest4D(
    n_estimators_per_layer=5,
    max_layers=20,
    early_stopping_rounds=5,
    spatial_h = 2, 
    spatial_w = 2,
    forget_factor=0.1,
    verbose=1
)
model.fit(X_train, y_train, X_val, y_val)

# Predict
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("\n✅ Final RMSE on validation set:", rmse)


# Output best model and RMSE
best_model, best_rmse = model.get_best_model()
print("\nBest validation RMSE:", best_rmse)
"""