from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import pickle

class NeuralProphetModel:

    def __init__(self):

        self.model = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            learning_rate=0.01,
        )

        self.forecast = None
        self.last_date = None
        self.train_df = None

    def prepare_data(self, df):

        prophet_df = df.rename(
            columns={
                "date": "ds",
                "valor_log": "y"
            }
        )

        return prophet_df[["ds", "y"]]

    def fit(self, train_df):

        prophet_df = self.prepare_data(train_df)
        self.train_df = prophet_df
        self.last_date = prophet_df["ds"].max()

        self.model.fit(prophet_df, freq="QS")

    def predict(self, periods):

        future = self.model.make_future_dataframe(
            df=self.train_df,
            periods=periods,
            n_historic_predictions=True
        )

        self.forecast = self.model.predict(future)

        return self.forecast

    def evaluate(self, test_df):

        test_real = np.expm1(test_df["valor_log"])

        preds = self.forecast["yhat1"].tail(len(test_df))
        preds_real = np.expm1(preds.values)

        mae = mean_absolute_error(
            test_real,
            preds_real
        )

        return mae
    
    def save_model(self, path):

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):

        with open(path, "rb") as f:
            self.model = pickle.load(f)