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
    from scitbx.utils import *
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

# ===============================================================================================================================
# Training utils
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

# randomly select sites
def random_select(ds, count, num, random_state = 0):
    np.random.seed(random_state)
    idxs = np.random.choice(np.delete(np.arange(len(ds)), count), num, replace = False)
    return np.sort(idxs)

def split(Xs, ys, return_index = False, test_size = 0.33, random_state = 42):
    if return_index:
        sss = ShuffleSplit(n_splits=1, test_size = test_size, random_state = random_state)
        sss.get_n_splits(Xs, ys)
        train_index, test_index = next(sss.split(Xs, ys)) 
        return (train_index, test_index)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, 
            test_size = test_size, 
            random_state = random_state
        )
        return (X_train, X_test, y_train, y_test)
    
def split_cut(Xs, ys, test_ratio = 0.33):
    """
    Split the timeseries into before and after halves
    """
    assert ys.ndim == 2, 'ys must be 2D!'
    assert len(Xs) == len(ys), 'Xs and ys should be equally long!'
    assert type(Xs) == type(ys), 'Xs and ys should be the same data type!'
    if not type(Xs) in [pd.core.frame.DataFrame, np.ndarray]: raise Exception('Only accept numpy ndarray or pandas dataframe')
    anchor = int(np.floor(len(ys) * (1 - test_ratio)))

    if type(Xs) == pd.core.frame.DataFrame:
        X_train = Xs.iloc[0: anchor, :]
        X_test = Xs.iloc[anchor::, :]
        y_train = ys.iloc[0: anchor, :]
        y_test = ys.iloc[anchor::, :]
    else:
        X_train = Xs[0: anchor, :]
        X_test = Xs[anchor::, :]
        y_train = ys[0: anchor, :]
        y_test = ys[anchor::, :]

    assert len(X_train) + len(X_test) == len(Xs), 'The sum of train and test lengths must equal to Xs/ys!'

    return (X_train, X_test, y_train, y_test)