import numpy as np
import copy
import itertools
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    eval_metric : str
        Statistical metric for evaluating model performance.

    Attributes:
    -----------
    layers : list
        List of trained layers, each containing a list of regressors.

    best_model : list
        The set of layers corresponding to the best validation RMSE seen during training.

    best_score : float
        The best metric e.g., lowest RMSE achieved on the validation set.

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
                 forget_factor=0.0, verbose=1, eval_metric='rmse'):
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
        self.verbose = verbose
        self.eval_metric = eval_metric.lower()
        self.best_score = float("inf") if self.eval_metric != 'r2' else float("-inf")
        if (self.spatial_h is None) or (self.spatial_w is None): 
            raise ValueError("Please specify spatial_h and spatial_w")

    def _evaluate(self, y_true, y_pred):
        if self.eval_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.eval_metric == 'nrmse':
            return np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true))
        elif self.eval_metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.eval_metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        elif self.eval_metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown evaluation metric: {self.eval_metric}")

    def _get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _prepare_input(self, X, y=None, apply_forget=False, layer_index=0):
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

        if n_spatial != 1:
            if self.spatial_h * self.spatial_w != n_spatial: raise ValueError("spatial_h * spatial_w != n_spatial")
            X_out = np.zeros_like(X)
            for sample in range(X.shape[0]):
                for t in range(X.shape[1]):
                    for f in range(X.shape[3]):
                        spatial_2d = X[sample, t, :, f].reshape(self.spatial_h, self.spatial_w)
                        filtered = ndimage.convolve(spatial_2d, self.kernel, mode='constant', cval=0.0)
                        X_out[sample, t, :, f] = filtered.reshape(n_spatial)
            X = X_out; del(X_out)
        X_pooled = X.max(axis=2)
        X_flattened = X_pooled.reshape(n_samples, -1)
        return X_flattened

    def _fit_layer(self, X, y, X_val=None, y_val=None, layer_index=0):
        layer = []
        layer_outputs = []
        param_combos = self._get_param_combinations()

        for i in range(self.n_estimators_per_layer):
            best_metric = float('inf')
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
                    metric = self._evaluate(y_val, preds_val)
                    if metric < best_metric:
                        best_metric = metric
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
                score = self._evaluate(y_val, y_pred)
                if self.verbose:
                    print(f"Validation {self.eval_metric.upper()}: {score:.4f}")

                improvement = (score < self.best_score) if self.eval_metric != 'r2' else (score > self.best_score)
                if improvement:
                    self.best_score = score
                    self.best_model = copy.deepcopy(self.layers)
                    no_improve_rounds = 0
                    if self.verbose:
                        print(f"\u2705 New best {self.eval_metric.upper()}: {self.best_score:.4f}")
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
        return self.best_model, self.best_score

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

# ============================================================================================================================================================
# Function mode

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

def srcnn(learning_rate=0.001):
    """
    Builds and compiles a Super-Resolution Convolutional Neural Network (SRCNN) model 
    that fuses features from both low-resolution and high-resolution images.

    This model uses two parallel input streams:
    - A low-resolution input which undergoes upscaling through convolutional layers.
    - A high-resolution input from which texture features are extracted and fused with the low-resolution stream.

    Args:
        save_path (str, optional): Path to save the compiled model. If None, the model is not saved.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    # Input layers
    lowres_input = layers.Input(shape=(None, None, 1))  # Low-resolution input
    highres_input = layers.Input(shape=(None, None, 1))  # High-resolution image

    # Feature extraction from high-resolution image
    highres_features = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(highres_input)
    highres_features = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(highres_features)

    # Processing low-resoltuion input
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(lowres_input)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Fusion of high-resolution image textures
    fusion = layers.Concatenate()([x, highres_features])
    fusion = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(fusion)
    fusion = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(fusion)

    # Output
    output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(fusion)

    model = keras.Model(inputs=[lowres_input, highres_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    return model

def print_model(model):
    return model.summary()

def train(lowres_data, highres_data, epochs=100, batch_size=1, verbose=1, save_path=None):
    model = srcnn()
    # Train SRCNN
    model.fit([lowres_data, highres_data], highres_data, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # Save the complete model
    # Recommended in newer versions of Keras (TensorFlow 2.11+): e.g., 'texture_fusion_model.keras'
    if save_path: model.save(save_path)

def apply(model, lowres_data_app, highres_data):
    super_resolved = model.predict([lowres_data_app, highres_data]).squeeze() 
    super_resolved = xr.DataArray(
        super_resolved, 
        dims = ("latitude", "longitude"), 
        coords={"latitude": highres_data.latitude, "longitude": highres_data.longitude}, 
        name="super_res"
    )
    return super_resolved

def load_model(save_path):
    model = load_model('texture_fusion_model.keras') 

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Class mode

import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

class TextureFusionSRCNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Input layers
        lowres_input = layers.Input(shape=(None, None, 1))  # Low-resolution input
        highres_input = layers.Input(shape=(None, None, 1))  # High-resolution image

        # Feature extraction from high-resolution image
        highres_features = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(highres_input)
        highres_features = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(highres_features)

        # Processing low-resolution input
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(lowres_input)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        # Fusion of high-resolution image textures
        fusion = layers.Concatenate()([x, highres_features])
        fusion = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(fusion)
        fusion = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(fusion)

        # Output
        output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(fusion)

        model = keras.Model(inputs=[lowres_input, highres_input], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")

        return model

    def summary(self):
        return self.model.summary()

    def train(self, lowres_data, highres_data, epochs=100, batch_size=1, verbose=1, save_path=None):
        early_stop = EarlyStopping(
            monitor='loss',       # You can change to 'val_loss' if you add validation
            patience=10,          # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True
        )

        self.model.fit(
            [lowres_data, highres_data], highres_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop]
        )

        if save_path:
            self.model.save(save_path)

    def apply(self, lowres_data_app, highres_data):
        super_resolved = self.model.predict([lowres_data_app, highres_data]).squeeze()
        return super_resolved

    @staticmethod
    def load(save_path):
        model = keras.models.load_model(save_path)
        instance = TextureFusionSRCNN()
        instance.model = model
        return instance

