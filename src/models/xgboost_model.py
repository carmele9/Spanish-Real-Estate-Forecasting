import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


class XGBoostModel:

    def __init__(self):

        self.model = XGBRegressor(
             n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1
        )
        

        self.predictions = None

    def fit(self, X_train, y_train):

        self.model.fit(X_train, y_train, verbose=True)

    def predict(self, X_test):

        self.predictions = self.model.predict(X_test)

        return self.predictions

    def evaluate(self, y_test):

        preds_real = np.expm1(self.predictions)
        y_real = np.expm1(y_test)

        mae = mean_absolute_error(
            y_real,
            preds_real
        )

        return mae

    def save_model(self, path):

        joblib.dump(self.model, path)
    
    def load_model(self, path):
        
        self.model = joblib.load(path)