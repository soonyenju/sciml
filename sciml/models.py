import numpy as np
import copy
import itertools
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

warnings.simplefilter('ignore')

class XGBoostDeepForestRegressorWithForgetGate:
    def __init__(self, n_estimators_per_layer = 2, max_layers = 5, early_stopping_rounds = 2, param_grid = None,
                 use_gpu = False, gpu_id=0, window_sizes=[2, 3], forget_factor=0.5):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.param_grid = param_grid or {
            'max_depth': [3],
            'learning_rate': [0.1],
            'n_estimators': [100]
        }
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.window_sizes = window_sizes
        self.forget_factor = forget_factor
        self.layers = []
        self.best_model = None
        self.best_rmse = float("inf")

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
                print(f"Validation RMSE: {rmse:.4f}")

                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_model = copy.deepcopy(self.layers)
                    no_improve_rounds = 0
                    print(f"✅ New best RMSE: {self.best_rmse:.4f}")
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        print("Early stopping triggered.")
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


# ========== Test Example ==========
X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter grid
param_grid = {
    "objective": ["reg:squarederror"],
    "random_state": [0, 42],
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
df_reg = XGBoostDeepForestRegressorWithForgetGate(
    n_estimators_per_layer=5,
    max_layers=10,
    early_stopping_rounds=5,
    param_grid=param_grid,
    use_gpu=True,
    gpu_id=0,
    window_sizes=[],  # Enables MGS if e.g., [2, 3], else empty disables MGS.
    forget_factor=0.  # Set forget factor to simulate forget gate behavior
)

df_reg.fit(X_train, y_train, X_val, y_val)

# Predict on validation set and evaluate
y_pred = df_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("\nFinal RMSE:", rmse)

# Output best model and RMSE
best_model, best_rmse = df_reg.get_best_model()
print("\nBest validation RMSE:", best_rmse)
