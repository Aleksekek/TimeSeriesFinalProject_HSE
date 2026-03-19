import numpy as np
import pandas as pd


class NaiveSeasonal:
    def __init__(self, seasonal_period: int):
        self.seasonal_period = seasonal_period
        self.y = None

    def fit(self, y: np.ndarray):
        self.y = y[-self.seasonal_period :]

    def predict(self, horizon: int) -> np.ndarray:
        pred = []
        batch_count = int(np.ceil(horizon / self.seasonal_period))
        for batch in range(batch_count):
            for val in self.y:
                if horizon:
                    pred.append(val)
                    horizon -= 1
        return pd.Series(pred)
