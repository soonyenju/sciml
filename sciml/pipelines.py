def train_ml(X_train, y_train, model_name = 'XGB', gpu = False):
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

    rfr_params = {
        'max_depth': 20,
        'min_samples_leaf': 3,
        'min_samples_split': 12,
        'n_estimators': 100,
        'n_jobs': -1
    }
    if model_name == "XGB":
        from xgboost import XGBRegressor
        regr = XGBRegressor(**xgb_params)
    elif model_name == "MLP":
        from sklearn.neural_network import MLPRegressor
        regr = MLPRegressor()
    elif model_name == "RFR":
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(**rfr_params)
    elif model_name == "SVR":
        from sklearn.svm import SVR
        regr = SVR()
    elif model_name == "DF21":
        from deepforest import CascadeForestRegressor
        # https://deep-forest.readthedocs.io/en/latest/api_reference.html?highlight=CascadeForestRegressor#cascadeforestregressor
        # predictor: {"forest", "xgboost", "lightgbm"}
        regr = CascadeForestRegressor(random_state = 1, verbose = 0, predictor = "xgboost", n_jobs = -1, predictor_kwargs  = xgb_params)
    regr.fit(X_train, y_train)
    return regr