from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


class ProphetModel:

    def __init__(self):

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        self.forecast = None

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

        self.model.fit(prophet_df)

    def predict(self, periods):

        future = self.model.make_future_dataframe(
            periods=periods,
            freq="Q"
        )

        self.forecast = self.model.predict(future)

        return self.forecast

    def evaluate(self, test_df):

        test_real = np.expm1(test_df["valor_log"])

        preds = self.forecast["yhat"].tail(len(test_df))
        preds_real = np.expm1(preds)

        mae = mean_absolute_error(
            test_real,
            preds_real
        )

        return mae