import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta


class StatsForecastAdapter:
    """
    Адаптер для моделей из statsforecast, чтобы они работали с интерфейсом fit/predict.
    """

    def __init__(self, model_class, model_params=None):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = None
        self.fitted = False

    def fit(self, y: np.ndarray):
        """
        Обучает модель на массиве y.
        """
        # Создаем DataFrame в формате, который ожидает StatsForecast
        train_df = pd.DataFrame(
            {
                "unique_id": [1] * len(y),
                "ds": pd.date_range(start="2000-01-01", periods=len(y), freq="D"),
                "y": y,
            }
        )

        # Создаем модель
        model_instance = self.model_class(**self.model_params)

        # Обучаем
        self.model = StatsForecast(models=[model_instance], freq="D", n_jobs=1)
        self.model.fit(train_df)
        self.fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Делает прогноз на horizon шагов вперед.
        """
        if not self.fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Делаем прогноз
        forecast_df = self.model.predict(h=horizon)

        # Извлекаем значения - они могут быть в колонке с именем модели
        model_name = self.model_class.__name__
        if model_name in forecast_df.columns:
            return forecast_df[model_name].values
        else:
            # Если не нашли по имени, берем первую числовую колонку
            numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return forecast_df[numeric_cols[0]].values
            else:
                raise ValueError("Не удалось извлечь прогноз из результата")
