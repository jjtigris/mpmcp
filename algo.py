from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
"""
The encapsulation of algorithms.

Require a parameter in __init__ as the params of the actual model.
Require a function named predict, taking input data and outputing confidence on the positive label
Recommend default param settings as static members of class definition
"""

# bascial algos
class logiAlgo(LogisticRegression):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {}
    DEFAULT_ENUM_BOOK = {
        'penalty':['l1','l2'], 
        'solver': ["liblinear", "saga"]
        }
    DEFAULT_STATIC_BOOK = {'max_iter':2000}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        
    def predictor(self, X):   
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
        

class NNAlgo(MLPClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate_init': [0.001, 0.1]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(100,), (256,), (100, 500), (500, 500), (100, 500, 100)]}  # NAS
    DEFAULT_STATIC_BOOK = {'max_iter':2000, 'early_stopping': True}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]


class svmAlgo(SVC):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5, 0.8]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'tol': [1e-3, 1e-4, 1e-5], 'kernel':['poly', 'rbf']} 
    DEFAULT_STATIC_BOOK = {'max_iter':2000, 'probability':True}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]

class rfcAlgo(RandomForestClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 150], 'max_depth': [10, 50]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {} 
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):

        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
    
class extAlgo(ExtraTreesClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 200], 'max_depth': [10, 50]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

# boost algos
class gBoostAlgo(GradientBoostingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [30, 200], 'max_depth':[3, 30]}
    DEFAULT_ENUM_BOOK = {'loss' : ['log_loss', 'deviance', 'exponential'], 'subsample':[0.5, 0.8]}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

class rfBoostAlgo(AdaBoostClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [50, 150]}
    DEFAULT_ENUM_BOOK = {}
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        self.estimator = RandomForestClassifier(n_estimators=10)

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]
    
class svmBoostAlgo(AdaBoostClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [50, 150]}
    DEFAULT_ENUM_BOOK = {}
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        self.params['algorithm'] = 'SAMME'
        self.estimator = SVC(C=0.8, kernel='linear', max_iter=2000, probability=True)

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]
    
class logiBoostAlgo(AdaBoostClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [50, 150]}
    DEFAULT_ENUM_BOOK = {}
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        self.estimator = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=500)

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

class dtBoostAlgo(AdaBoostClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [30, 100]}
    DEFAULT_ENUM_BOOK = {}
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        
        self.estimator = DecisionTreeClassifier(max_depth=5)

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

# bagging algos
class svmBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 30]}
    DEFAULT_ENUM_BOOK = {'tol': [1e-3, 1e-4, 1e-5], 'max_samples':[0.5, 0.8], 'kernel':['linear', 'rbf']}
    DEFAULT_STATIC_BOOK = {'bootstrap': True, 'C':0.5, 'max_iter':500}

    def __init__(self, params):
        svm_params = {k: v for k, v in params.items() 
                      if (k in svmAlgo.DEFAULT_STATIC_BOOK 
                          or k in svmAlgo.DEFAULT_CONTINUOUS_BOOK 
                          or k in svmAlgo.DEFAULT_ENUM_BOOK)}
        svm_params['kernel'] = 'rbf'
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}

        base_estimator = svmAlgo(svm_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]
    

class logiBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5, 1.0]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 100]}
    DEFAULT_ENUM_BOOK = {'max_samples':[0.5, 0.8]}
    DEFAULT_STATIC_BOOK = {'max_iter': 1000, 'bootstrap': True}

    def __init__(self, params):
        logi_params = {k: v for k, v in params.items() 
                       if (k in logiAlgo.DEFAULT_STATIC_BOOK 
                           or k in logiAlgo.DEFAULT_DISCRETE_BOOK 
                           or k in logiAlgo.DEFAULT_ENUM_BOOK)}
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}
        
        base_estimator = logiAlgo(logi_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]


class NNBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [30, 150]}
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(50,), (100,), (100,256)], 'max_samples':[0.5, 0.8]}
    DEFAULT_STATIC_BOOK = {'bootstrap': True, 'max_iter': 500}

    def __init__(self, params):
        NN_params = {k: v for k, v in params.items() 
                      if (k in NNAlgo.DEFAULT_STATIC_BOOK 
                          or k in NNAlgo.DEFAULT_DISCRETE_BOOK 
                          or k in NNAlgo.DEFAULT_ENUM_BOOK)}
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}

        base_estimator = NNAlgo(NN_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

# XGB and LGB Added
import xgboost as xgb
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin


class xgbAlgo(BaseEstimator, ClassifierMixin):
    DEFAULT_CONTINUOUS_BOOK = {} 
    DEFAULT_DISCRETE_BOOK = {'max_depth': [3, 20], 'n_estimators': [100, 500] }   
    DEFAULT_ENUM_BOOK = {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.01, 0.05, 0.1],
        'eval_metric': ['logloss', 'auc']
    }
    DEFAULT_STATIC_BOOK = {}  

    def __init__(self, params):
        # binary task
        params['objective'] = 'binary:logistic'
        self.params = params
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predictor(self, X):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]
    
class NNxgbAlgo(MLPClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate_init': [0.0001, 0.1]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(50,), (100,), (256,), (100,256), (100, 256, 100)]}  # NAS
    DEFAULT_STATIC_BOOK = {'max_iter':500, 'early_stopping': False}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]

    def fit(self, X, y):

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use XGBoost's MLP implementation
        xgb_params = {
            'eta': self.params['learning_rate_init'], # learning rate
            'max_depth': self.params['hidden_layer_sizes'][0], # depth of the trees
            'objective': 'binary:logistic', # binary classification problem
            'eval_metric': 'logloss' # logloss evaluation metric
        }

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = class_weights[y_train]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        num_rounds = self.params.get('max_iter', 2000)
        early_stopping = self.params.get('early_stopping', True)

        super().fit(X_train, y_train)
        if early_stopping:
            evals = [(dval, 'eval')]
            self.model = xgb.train(xgb_params, dtrain, num_rounds, evals=evals, early_stopping_rounds=10)
        else:
            self.model = xgb.train(xgb_params, dtrain, num_rounds)

class lgbmAlgo(BaseEstimator, ClassifierMixin):
    DEFAULT_CONTINUOUS_BOOK = {}  # Add default continuous hyperparameters if needed
    DEFAULT_DISCRETE_BOOK = {'num_leaves': [10, 150],
                            'n_estimators': [50, 500]}    
    DEFAULT_ENUM_BOOK = {
        'boosting_type': ['gbdt', 'dart', 'goss'],
        'learning_rate': [0.01, 0.05, 0.1],
        'metric': ['binary_logloss', 'auc']
    }
    DEFAULT_STATIC_BOOK = {'force_col_wise':True}      # Add default static hyperparameters if needed

    def __init__(self, params):
        # binary task
        params['objective'] = 'binary'
        params['verbose'] = 0
        self.params = params
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predictor(self, X):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]
