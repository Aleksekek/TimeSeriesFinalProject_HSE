import warnings
from typing import Dict, List, Optional, Tuple

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class CatBoostAdapter:
    """
    Адаптер для CatBoost.
    """

    def __init__(
        self,
        model_params: Dict = None,
        history: int = 60,
        forecast_horizon: int = 16,
        use_scaling: bool = True,
    ):
        """
        Parameters:
        -----------
        model_params : Dict
            Параметры CatBoost
        history : int
            Размер окна истории для создания признаков
        forecast_horizon : int
            Горизонт прогнозирования
        use_scaling : bool
            Использовать ли стандартизацию
        """
        self.model_params = model_params or {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 30,
        }
        self.history = history
        self.forecast_horizon = forecast_horizon
        self.use_scaling = use_scaling
        self.model = None
        self.scaler_x = StandardScaler() if use_scaling else None
        self.scaler_y = StandardScaler() if use_scaling else None
        self.feature_names = None

    def _create_features(self, series: np.ndarray) -> pd.DataFrame:
        """
        Создает признаки из временного ряда.
        """
        df = pd.DataFrame()

        # Лаги
        for lag in [1, 2, 3, 7, 14, 30]:
            if lag <= len(series):
                df[f"lag_{lag}"] = pd.Series(series).shift(lag)

        # Скользящие средние
        for window in [7, 14, 30]:
            if window <= len(series):
                rolling = pd.Series(series).rolling(window=window, min_periods=1)
                df[f"rolling_mean_{window}"] = rolling.mean().shift(1)
                df[f"rolling_std_{window}"] = rolling.std().shift(1)
                df[f"rolling_max_{window}"] = rolling.max().shift(1)
                df[f"rolling_min_{window}"] = rolling.min().shift(1)

        # Разности
        df["diff_1"] = pd.Series(series).diff(1)
        df["diff_7"] = pd.Series(series).diff(7)

        # Трендовые признаки
        if len(series) >= 14:
            df["trend_7"] = pd.Series(series).diff(7).rolling(7, min_periods=1).mean()

        # Отношения
        if "rolling_mean_7" in df.columns:
            df["value_to_mean_7"] = pd.Series(series) / (df["rolling_mean_7"] + 1e-8)

        return df

    def _prepare_data(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения (прогноз на 1 шаг вперед).
        """
        values = series.values.astype(np.float32)

        X_list = []
        y_list = []

        for i in range(len(values) - self.history):
            X_list.append(values[i : i + self.history])
            y_list.append(values[i + self.history])

        if len(X_list) < 10:
            return None, None

        # Создаем признаки для каждого окна
        X_features = []
        for x in X_list:
            features = self._create_features(x)
            # Берем последнюю строку (самые свежие признаки)
            # и заполняем NaN нулями
            last_features = features.iloc[-1].fillna(0).values
            X_features.append(last_features)

        X = np.array(X_features)
        y = np.array(y_list)

        # Нормализуем
        if self.use_scaling:
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        return X, y

    def fit(self, train_series: pd.Series, val_series: pd.Series = None):
        """
        Обучает модель.
        """
        # Подготавливаем данные
        X_train, y_train = self._prepare_data(train_series)

        if X_train is None or len(X_train) < 10:
            print(f"    ⚠️  Недостаточно данных для обучения")
            return self

        # Подготавливаем валидационные данные
        eval_set = None
        if val_series is not None and len(val_series) > self.history:
            X_val, y_val = self._prepare_data(val_series)
            if X_val is not None and len(X_val) > 0:
                if self.use_scaling:
                    # Используем тот же scaler, что и для train
                    X_val = self.scaler_x.transform(X_val)
                    y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
                eval_set = (X_val, y_val)

        # Обучаем модель
        self.model = cb.CatBoostRegressor(**self.model_params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
            verbose_eval=False,
            plot=False,
        )

        # Сохраняем названия признаков
        feature_df = self._create_features(np.zeros(self.history))
        self.feature_names = feature_df.columns.tolist()

        return self

    def predict(self, last_series: pd.Series, horizon: int = None) -> np.ndarray:
        """
        Делает рекурсивный прогноз на horizon шагов.
        """
        if self.model is None:
            return np.full(horizon or self.forecast_horizon, last_series.iloc[-1])

        h = horizon or self.forecast_horizon
        predictions = []

        # Текущие значения для рекурсивного прогноза
        current_values = last_series.values[-self.history :].copy()

        for step in range(h):
            # Создаем признаки из текущего окна
            features = self._create_features(current_values)
            X = features.iloc[-1:].fillna(0).values

            # Нормализуем
            if self.use_scaling:
                X = self.scaler_x.transform(X)

            # Прогнозируем следующий шаг
            pred_scaled = self.model.predict(X)[0]

            # Возвращаем в исходный масштаб
            if self.use_scaling:
                pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            else:
                pred = pred_scaled

            pred = max(0, pred)  # неотрицательные продажи
            predictions.append(pred)

            # Обновляем окно для следующего шага
            current_values = np.append(current_values[1:], pred)

        return np.array(predictions)


# Фабрика для создания адаптера
def create_catboost_adapter(history=60, forecast_horizon=16):
    """Создает прямой адаптер CatBoost"""
    return CatBoostAdapter(
        model_params={
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 30,
            "task_type": "GPU",
        },
        history=history,
        forecast_horizon=forecast_horizon,
        use_scaling=True,
    )
