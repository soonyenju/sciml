import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

def get_metrics(df, truth = 'truth', pred = 'pred'):
    df = df[[truth, pred]].copy().dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()[truth], df.dropna()[pred])
    r2 = r_value**2
    mse = mean_squared_error(df.dropna()[truth], df.dropna()[pred])
    rmse = np.sqrt(mse)
    mbe = np.mean(df.dropna()[pred] - df.dropna()[truth])
    mae = (df.dropna()[pred] - df.dropna()[truth]).abs().mean()
    return r2, slope, rmse, mbe, mae, intercept, p_value, std_err


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