import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.exceptions import NotFittedError 
import warnings
import copy
from time import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# 忽略 Scikit-learn 和 XGBoost 的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class DeepForestBase(BaseEstimator):
    """DeepForest 的基础抽象类，处理参数和初始化。"""
    def __init__(self, 
                 base_estimator='xgb', 
                 n_estimators_per_type=1,
                 n_trees=50, 
                 max_layers=10, 
                 n_splits=3,
                 n_tolerant_rounds=2, 
                 random_state=None,
                 verbose=1,
                 use_gpu=False):  # <--- 新增 GPU 参数
        
        self.base_estimator = base_estimator.lower()
        self.n_estimators_per_type = n_estimators_per_type
        self.n_trees = n_trees
        self.max_layers = max_layers
        self.n_splits = n_splits
        self.n_tolerant_rounds = n_tolerant_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu # <--- 存储 GPU 选项
        
        # 内部状态
        self.layers_ = []
        self.n_outputs_ = 0
        self.n_layers_fitted_ = 0
        
        if self.base_estimator not in ['xgb', 'rf']:
            raise ValueError("base_estimator must be 'xgb' (XGBoost) or 'rf' (Random Forest).")

    def _get_prototype_model(self, is_classifier, seed_offset):
        """返回基础估计器原型（用于克隆和 K-Fold 训练）。"""
        rs = self.random_state + seed_offset if self.random_state is not None else None
        
        if is_classifier:
            default_params = {'n_estimators': self.n_trees, 'n_jobs': 1, 'random_state': rs}
            
            if self.base_estimator == 'xgb':
                max_depth = 5 
                booster = 'gbtree' if seed_offset % 2 == 0 else 'dart'
                
                # <--- GPU 逻辑实现 --->
                tree_method = 'gpu_hist' if self.use_gpu else 'hist'
                
                return xgb.XGBClassifier(
                    objective='multi:softprob', use_label_encoder=False,
                    max_depth=max_depth, booster=booster, tree_method=tree_method, 
                    **default_params
                )
            elif self.base_estimator == 'rf':
                # Random Forest 不支持 GPU，忽略 use_gpu 标志
                max_features = 'sqrt' if seed_offset % 2 == 0 else 1
                return RandomForestClassifier(max_features=max_features, **default_params)
        
        else: # 回归器
            default_params = {'n_estimators': self.n_trees, 'n_jobs': 1, 'random_state': rs}
            
            if self.base_estimator == 'xgb':
                # <--- GPU 逻辑实现 --->
                tree_method = 'gpu_hist' if self.use_gpu else 'hist'

                return xgb.XGBRegressor(
                    objective='reg:squarederror', tree_method=tree_method,
                    max_depth=5, **default_params
                )
            elif self.base_estimator == 'rf':
                return RandomForestRegressor(**default_params)
        
        raise ValueError("Invalid base estimator configuration.")


    def _fit_layer(self, X_in, y, layer_idx, is_classifier):
        """拟合单个级联层，返回下一层输入和验证性能。"""
        n_samples = X_in.shape[0]
        
        if is_classifier:
            n_cols = self.n_outputs_ 
            y_proc = y
        else:
            n_cols = self.n_outputs_
            y_proc = y

        new_features = []
        layer_estimators = [] 
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, 
                   random_state=self.random_state + layer_idx)
        
        for i in range(self.n_estimators_per_type * 2):
            prototype = self._get_prototype_model(is_classifier, i + layer_idx * 100)
            oob_pred = np.zeros((n_samples, n_cols))
            estimator_fold_clones = []

            if self.verbose > 1:
                gpu_tag = " (GPU)" if self.base_estimator == 'xgb' and self.use_gpu else ""
                est_type = self.base_estimator + ('_type1' if i%2 == 0 else '_type2') + gpu_tag
                print(f"--- Fitting Layer {layer_idx}, Estimator {i+1} ({est_type})...")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_in, y_proc)):
                X_train, X_val = X_in[train_idx], X_in[val_idx]
                y_train = y_proc[train_idx]
                
                estimator_clone = copy.deepcopy(prototype)
                
                if self.base_estimator == 'xgb' and is_classifier:
                    # XGBoost 内部可能需要这个属性，特别是处理多分类时
                    estimator_clone.n_classes_ = self.n_outputs_
                
                estimator_clone.fit(X_train, y_train)
                
                if is_classifier:
                    proba = estimator_clone.predict_proba(X_val)
                    oob_pred[val_idx] = proba
                else:
                    pred = estimator_clone.predict(X_val)
                    if self.n_outputs_ == 1 and pred.ndim == 1:
                        pred = np.expand_dims(pred, axis=1)
                    oob_pred[val_idx] = pred
                
                estimator_fold_clones.append(estimator_clone) 
            
            layer_estimators.append(estimator_fold_clones) 
            new_features.append(oob_pred)
        
        avg_output_oob = np.mean(np.array(new_features), axis=0)
        
        if is_classifier:
            y_pred_labels = np.argmax(avg_output_oob, axis=1)
            val_performance = accuracy_score(y_proc, y_pred_labels)
        else:
            val_performance = mean_squared_error(y_proc, avg_output_oob)
        
        # 组装下一层的输入特征
        X_out = np.hstack([X_in] + new_features)

        return X_out, layer_estimators, val_performance

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        is_classifier = isinstance(self, ClassifierMixin)

        if is_classifier:
            self.classes_, y_encoded = np.unique(y, return_inverse=True)
            self.n_outputs_ = len(self.classes_)
            y_target = y_encoded
        else:
            if y.ndim == 1:
                y = np.expand_dims(y, axis=1)
            self.n_outputs_ = y.shape[1]
            y_target = y

        if self.random_state is None:
            self.random_state = np.random.randint(0, 1000)

        X_original = X # 存储原始特征，用于预测阶段的拼接
        X_current = X
        self.layers_ = []
        
        best_performance = -np.inf if is_classifier else np.inf
        tolerance_counter = 0

        if self.verbose > 0:
            gpu_mode = " (GPU Enabled)" if self.base_estimator == 'xgb' and self.use_gpu else ""
            print(f"Starting Deep Forest ({'Classifier' if is_classifier else 'Regressor'}) Training{gpu_mode}...")
            print(f"Base Estimator: {self.base_estimator.upper()}")

        for layer_idx in range(self.max_layers):
            if self.verbose > 0:
                print(f"--- Layer {layer_idx}: {X_current.shape[1]} features in.")
            
            X_next, layer_estimators, val_performance = self._fit_layer(
                X_current, y_target, layer_idx, is_classifier
            )
            
            if self.verbose > 0:
                perf_str = f"Val Acc: {val_performance:.4f}" if is_classifier else f"Val MSE: {val_performance:.4f}"
                print(f"--- Layer {layer_idx} Finished. {perf_str}")
                
            improved = (val_performance > best_performance) if is_classifier else (val_performance < best_performance)

            if improved:
                best_performance = val_performance
                tolerance_counter = 0
                
                self.layers_.append(layer_estimators)
                self.n_layers_fitted_ = len(self.layers_)
                X_current = X_next
            else:
                tolerance_counter += 1
                if self.verbose > 0:
                    print(f"--- Performance did not improve. Tolerance: {tolerance_counter}/{self.n_tolerant_rounds}")
                
                if tolerance_counter >= self.n_tolerant_rounds:
                    if self.verbose > 0:
                        print("Stopping due to early stopping.")
                    break
                    
                X_current = X_next # 保持下一轮的输入，尽管性能没有提升
                
        return self

    def _predict_layers(self, X_in):
        """
        预测阶段，在所有已拟合层上进行前向传播。
        修复了维度不匹配的 Bug: 确保 X_current 在每层都包含所有历史增强特征。
        """
        X_original = X_in.copy() 
        X_current = X_original.copy() 
        final_output_layers = []
        is_classifier = isinstance(self, ClassifierMixin)
        
        # 用于累积所有层输出的特征列表
        accumulated_predictions = [] 
        
        for layer_idx in range(self.n_layers_fitted_):
            
            layer_estimators = self.layers_[layer_idx]
            base_estimator_predictions = []
            
            if self.verbose > 0:
                 print(f"--- Layer {layer_idx} predicting...")
            
            # 1. 预测并生成当前层的增强特征
            for estimator_group in layer_estimators:
                
                if is_classifier:
                    fold_preds = [e.predict_proba(X_current) for e in estimator_group]
                else:
                    fold_preds = [e.predict(X_current) for e in estimator_group]
                    
                    if self.n_outputs_ == 1 and fold_preds[0].ndim == 1:
                        fold_preds = [np.expand_dims(p, axis=1) for p in fold_preds]
                    
                avg_base_estimator_output = np.mean(np.array(fold_preds), axis=0)
                base_estimator_predictions.append(avg_base_estimator_output)
            
            # 2. 存储当前层的最终输出（用于返回）
            current_layer_output = np.mean(np.array(base_estimator_predictions), axis=0)
            final_output_layers.append(current_layer_output)
            
            # 3. 更新下一轮的输入 X_current (只有当前层不是最后一层时才需要)
            if layer_idx < self.n_layers_fitted_ - 1:
                # 将当前层的增强特征加入积累列表
                accumulated_predictions.extend(base_estimator_predictions) 
                
                # 下一层输入 = [原始特征] + [所有积累的增强特征]
                X_current = np.hstack([X_original] + accumulated_predictions)
                
        return final_output_layers


class DeepForestClassifier(DeepForestBase, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.classes_ = None 

    def predict_proba(self, X):
        X_in = np.asarray(X)
        if self.n_layers_fitted_ == 0:
            raise NotFittedError(f"This {type(self).__name__} instance is not fitted yet.")
            
        final_output_layers = self._predict_layers(X_in)
        
        return final_output_layers[-1]

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DeepForestRegressor(DeepForestBase, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        X_in = np.asarray(X)
        if self.n_layers_fitted_ == 0:
            raise NotFittedError(f"This {type(self).__name__} instance is not fitted yet.")
            
        final_output_layers = self._predict_layers(X_in)
        
        output = final_output_layers[-1]
        
        if self.n_outputs_ == 1 and output.ndim == 2:
            return output.flatten()
            
        return output


# --- 运行示例（分类器和回归器）---

# --- 1. 分类任务示例 (默认: XGBoost, 启用 GPU) ---
print("=" * 40)
print("分类任务示例 (Base Estimator: XGBoost)")
print("=" * 40)

X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=5, n_redundant=10, 
    n_classes=3, n_clusters_per_class=2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

clf_model = DeepForestClassifier(
    base_estimator='xgb',
    n_estimators_per_type=1,
    n_trees=50,
    max_layers=5,
    n_splits=3,
    verbose=1,
    use_gpu=True # 如果你有 NVIDIA GPU，请改为 True 测试加速
)

clf_start_time = time()
clf_model.fit(X_train_clf, y_train_clf)
clf_train_time = time() - clf_start_time

y_pred_clf = clf_model.predict(X_test_clf)
clf_acc = accuracy_score(y_test_clf, y_pred_clf)

print(f"\nTraining completed in {clf_train_time:.2f} seconds. Fitted layers: {clf_model.n_layers_fitted_}")
print(f"Classification Test Accuracy (XGBoost DF): {clf_acc:.4f}")

# --- 2. 回归任务示例 (修复后的 XGBoost 累积特征) ---
print("\n" + "=" * 40)
print("回归任务示例 (Base Estimator: XGBoost, 修复维度 Bug)")
print("=" * 40)

X_reg, y_reg = make_regression(n_samples=500, n_features=10, n_targets=2, random_state=42) 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg_model = DeepForestRegressor(
    base_estimator='xgb', # 使用 XGBoost
    n_estimators_per_type=1,
    n_trees=40,
    max_layers=5,
    n_splits=3,
    verbose=1,
    use_gpu=True
)

reg_start_time = time()
reg_model.fit(X_train_reg, y_train_reg)
reg_train_time = time() - reg_start_time

y_pred_reg = reg_model.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg, multioutput='uniform_average')

print(f"\nTraining completed in {reg_train_time:.2f} seconds. Fitted layers: {reg_model.n_layers_fitted_}")
print(f"Regression Test Mean Squared Error (XGBoost DF): {reg_mse:.4f}")
