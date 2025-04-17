import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def get_metrics(df, truth = 'truth', pred = 'pred', return_dict = False):
    '''
    Calculate statistical measures between validation and prediction sequences
    '''
    df = df[[truth, pred]].copy().dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()[truth], df.dropna()[pred])
    r2 = r_value**2
    mse = mean_squared_error(df.dropna()[truth], df.dropna()[pred])
    rmse = np.sqrt(mse)
    mbe = np.mean(df.dropna()[pred] - df.dropna()[truth])
    mae = (df.dropna()[pred] - df.dropna()[truth]).abs().mean()
    if return_dict:
        return pd.DataFrame.from_dict([{
            'r2': r2, 
            'Slope': slope, 
            'RMSE': rmse, 
            'MBE': mbe, 
            'MAE': mae, 
            'Intercept': intercept, 
            'p-value': p_value, 
            'std_err': std_err
        }])
    else:
        return r2, slope, rmse, mbe, mae, intercept, p_value, std_err

# ===============================================================================================================================
# Machine learning algorithms
def train_ml(
    X_train, y_train, model_name = 'XGB', 
    xgb_params_user = None, rfr_params_user = None, 
    mlp_params_user = None, svr_params_user = None,
    df21_params_user = None,
    gpu = False, partial_mode = False
    ):
    # -------------------------------------------------------------------------
    # Setup parameters:
    if xgb_params_user:
        xgb_params = xgb_params_user
    else:
        xgb_params = {
            "objective": "reg:squarederror",
            "random_state": 0,
            'seed': 0,
            'n_estimators': 100,
            'max_depth': 6, 
            'min_child_weight': 4, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'gamma': 0, 
            'reg_alpha': 0, 
            'reg_lambda': 1,
            'learning_rate': 0.05, 
        }

        xgb_gpu_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            # "n_gpus": 2, 
        }

        if gpu: xgb_params.update(xgb_gpu_params)

    if rfr_params_user:
        rfr_params = rfr_params_user
    else:
        rfr_params = {
            'max_depth': 20,
            'min_samples_leaf': 3,
            'min_samples_split': 12,
            'n_estimators': 100,
            'n_jobs': -1
        }

    if df21_params_user:
        df21_params = df21_params_user
    else:
        df21_params = {
            'random_state': 1, 
            'verbose' : 0, 
            'predictor': "xgboost", 
            'n_jobs' : -1, 
            'predictor_kwargs' : xgb_params, 
            'partial_mode' : partial_mode
        }
    # -------------------------------------------------------------------------
    # Run:
    if model_name == "XGB":
        from xgboost import XGBRegressor
        regr = XGBRegressor(**xgb_params)
    elif model_name == "MLP":
        from sklearn.neural_network import MLPRegressor
        regr = MLPRegressor(**mlp_params_user)
    elif model_name == "RFR":
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(**rfr_params)
    elif model_name == "SVR":
        from sklearn.svm import SVR
        regr = SVR(**svr_params_user)
    elif model_name == "DF21":
        from deepforest import CascadeForestRegressor
        # https://deep-forest.readthedocs.io/en/latest/api_reference.html?highlight=CascadeForestRegressor#cascadeforestregressor
        # predictor: {"forest", "xgboost", "lightgbm"}
        # regr = CascadeForestRegressor(random_state = 1, verbose = 0, predictor = "xgboost", n_jobs = -1, predictor_kwargs  = xgb_params, partial_mode = partial_mode)
        regr = CascadeForestRegressor(**df21_params)
    regr.fit(X_train, y_train)
    return regr

def test_ml(X_test, y_test, regr):
    res = y_test.copy() # y_test is 2D pandas dataframe.
    res.columns = ['truth']
    res['pred'] = regr.predict(X_test)
    return res

def run_ensemble(X_train, y_train, n_models = 10, frac_sample = 0.8):
    base_params_xgb = {
        "objective": "reg:squarederror",
        'seed': 0,
        "random_state": 0,
    }
    params_xgb = deepcopy(base_params_xgb)
    # dropout-like regularization
    params_xgb.update({
        "subsample": 0.8,  # Use 80% of the data for each tree
        "colsample_bytree": 0.8,  # Use 80% of the features for each tree
    })

    models = []
    for i in tqdm(range(n_models)):
        # Create a bootstrapped dataset
        y_resampled = y_train.copy().sample(frac = frac_sample, random_state = i)
        X_resampled = X_train.copy().loc[y_resampled.index]
        # print(y_resampled.sort_index().index[0], y_resampled.sort_index().index[-1])

        # Train the XGBoost model
        params_xgb.update({'random_state': i})
        model = XGBRegressor(**params_xgb)
        model.fit(X_resampled, y_resampled)
        models.append(model)
    return models

# ===============================================================================================================================
# Deep learning neural networks

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import models
    # from keras.layers import Dropout
    from keras.callbacks import EarlyStopping
    from scitbx.stutils import *
except Exception as e:
    print(e)

def train_lstm(X_train, y_train, nfeature, ntime, verbose = 2, epochs = 200, batch_size = 64):
    # create and fit the LSTM network
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(nfeature, ntime)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    # model.fit(X_train.reshape(-1, nsites, nfeats), y_train, epochs=100, batch_size=256, verbose=2, callbacks=[es])
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose=verbose)
    return model


'''
# ========================================================================================================
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class XGBoostDeepForestRegressor:
    def __init__(self, n_estimators_per_layer=2, max_layers=20, early_stopping_rounds=2):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.layers = []

    def _fit_layer(self, X, y):
        layer = []
        layer_outputs = []
        for _ in range(self.n_estimators_per_layer):
            reg = XGBRegressor()
            reg.fit(X, y)
            preds = reg.predict(X).reshape(-1, 1)
            layer.append(reg)
            layer_outputs.append(preds)
        output = np.hstack(layer_outputs)
        return layer, output

    def fit(self, X, y, X_val=None, y_val=None):
        X_current = X.copy()
        best_rmse = float("inf")
        no_improve_rounds = 0

        for layer_index in range(self.max_layers):
            print(f"Training Layer {layer_index + 1}")
            layer, output = self._fit_layer(X_current, y)
            self.layers.append(layer)
            X_current = np.hstack([X_current, output])

            if X_val is not None:
                y_pred = self.predict(X_val)
                # rmse = mean_squared_error(y_val, y_pred, squared=False)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                print(f"Validation RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        print("Early stopping triggered.")
                        break

    def predict(self, X):
        X_current = X.copy()
        for layer in self.layers:
            layer_outputs = []
            for reg in layer:
                n_features = reg.n_features_in_
                preds = reg.predict(X_current[:, :n_features]).reshape(-1, 1)
                layer_outputs.append(preds)
            output = np.hstack(layer_outputs)
            X_current = np.hstack([X_current, output])

        # Final prediction = average of last layer regressors
        final_outputs = []
        for reg in self.layers[-1]:
            n_features = reg.n_features_in_
            final_outputs.append(reg.predict(X_current[:, :n_features]).reshape(-1, 1))
        return np.mean(np.hstack(final_outputs), axis=1)


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

df_reg = XGBoostDeepForestRegressor(n_estimators_per_layer=2, max_layers=5)
df_reg.fit(X_train, y_train, X_val, y_val)

y_pred = df_reg.predict(X_val)
# rmse = mean_squared_error(y_val, y_pred, squared=False)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Final RMSE:", rmse)

# ----------------------------------------------------------------------------------------------------

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import itertools

class XGBoostDeepForestRegressor:
    def __init__(self, n_estimators_per_layer=2, max_layers=20, early_stopping_rounds=2, param_grid=None, use_gpu=True, gpu_id=0):
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
        self.layers = []

    def _get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _fit_layer(self, X, y, X_val=None, y_val=None):
        layer = []
        layer_outputs = []
        param_combos = self._get_param_combinations()

        for i in range(self.n_estimators_per_layer):
            best_rmse = float('inf')
            best_model = None

            for params in param_combos:
                # Set GPU support parameters in XGBRegressor
                if self.use_gpu:
                    params['tree_method'] = 'hist'  # Use hist method
                    params['device'] = 'cuda'  # Enable CUDA for GPU

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

            final_model = best_model
            preds = final_model.predict(X).reshape(-1, 1)
            layer.append(final_model)
            layer_outputs.append(preds)

        output = np.hstack(layer_outputs)
        return layer, output

    def fit(self, X, y, X_val=None, y_val=None):
        X_current = X.copy()
        X_val_current = X_val.copy() if X_val is not None else None

        best_rmse = float("inf")
        no_improve_rounds = 0

        for layer_index in range(self.max_layers):
            print(f"Training Layer {layer_index + 1}")
            layer, output = self._fit_layer(X_current, y, X_val_current, y_val)
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

                if rmse < best_rmse:
                    best_rmse = rmse
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        print("Early stopping triggered.")
                        break

    def predict(self, X):
        X_current = X.copy()
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


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter grid
param_grid = {
    'max_depth': [3, 4],
    'learning_rate': [0.1, 0.05],
    'n_estimators': [50, 100]
}

# Create and fit the model with GPU enabled
df_reg = XGBoostDeepForestRegressor(
    n_estimators_per_layer=2,
    max_layers=5,
    early_stopping_rounds=2,
    param_grid=param_grid,
    use_gpu=True,  # Enable GPU acceleration
    gpu_id=0  # Default to the first GPU
)

df_reg.fit(X_train, y_train, X_val, y_val)

# Final evaluation
y_pred = df_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Final RMSE:", rmse)

# ----------------------------------------------------------------------------------------------------

xgb_params = {
    "objective": "reg:squarederror",
    "random_state": 0,
    'seed': 0,
    'n_estimators': 100,
    'max_depth': 6, 
    'min_child_weight': 4, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'gamma': 0, 
    'reg_alpha': 0, 
    'reg_lambda': 1,
    'learning_rate': 0.05, 
}

from xgboost import XGBRegressor
regr = XGBRegressor(**xgb_params)

regr.fit(X_train, y_train)
y_pred = regr.predict(X_val)


from scipy import stats

stats.linregress(y_val, y_pred)

'''