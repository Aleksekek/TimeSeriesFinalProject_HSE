import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import warnings
warnings.filterwarnings('ignore')


class AutoETSAdapter:
    """
    Прямой адаптер для AutoETS.
    Обучается на каждой паре магазин-товар отдельно.
    """
    
    def __init__(self,
                 forecast_horizon: int = 16,
                 season_length: int = 7,
                 model_params: Dict = None):
        """
        Parameters:
        -----------
        forecast_horizon : int
            Горизонт прогнозирования
        season_length : int
            Длина сезонного периода
        model_params : Dict
            Дополнительные параметры для AutoETS
        """
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.model_params = model_params or {}
        self.model = None
        self.last_values = None
        
    def fit(self, 
            train_series: pd.Series,
            val_series: pd.Series = None):
        """
        Обучает модель AutoETS.
        
        Parameters:
        -----------
        train_series : pd.Series
            Обучающий ряд
        val_series : pd.Series, optional
            Валидационный ряд (не используется AutoETS, но оставляем для совместимости)
        """
        if len(train_series) < 2 * self.season_length:
            print(f"    ⚠️  Недостаточно данных для AutoETS: {len(train_series)} < {2 * self.season_length}")
            # Сохраняем последние значения для fallback
            self.last_values = train_series.values[-min(self.season_length, len(train_series)):]
            return self
        
        try:
            # Создаем DataFrame в формате StatsForecast
            train_df = pd.DataFrame({
                'unique_id': ['series'] * len(train_series),
                'ds': pd.date_range(start='2000-01-01', periods=len(train_series), freq='D'),
                'y': train_series.values
            })
            
            # Создаем модель AutoETS
            ets_model = AutoETS(season_length=self.season_length, **self.model_params)
            
            # Обучаем
            self.model = StatsForecast(
                models=[ets_model],
                freq='D',
                n_jobs=1,
                verbose=False
            )
            self.model.fit(train_df)
            
            # Сохраняем последние значения для возможных fallback
            self.last_values = train_series.values[-self.season_length:]
            
        except Exception as e:
            print(f"    ⚠️  Ошибка обучения AutoETS: {e}")
            self.last_values = train_series.values[-min(self.season_length, len(train_series)):]
        
        return self
    
    def predict(self, last_series: pd.Series, horizon: int = None) -> np.ndarray:
        """
        Делает прогноз.
        
        Parameters:
        -----------
        last_series : pd.Series
            Последние значения ряда (используются только для fallback)
        horizon : int, optional
            Горизонт прогноза
            
        Returns:
        --------
        np.ndarray
            Прогноз
        """
        h = horizon or self.forecast_horizon
        
        if self.model is None:
            # Если модель не обучена, используем fallback
            return self._fallback_predict(last_series, h)
        
        try:
            # Делаем прогноз
            forecast_df = self.model.predict(h=h)
            
            # Извлекаем прогноз (колонка с именем модели)
            pred_column = [col for col in forecast_df.columns if col not in ['unique_id', 'ds']][0]
            predictions = forecast_df[pred_column].values
            
            # Обрезаем отрицательные значения
            predictions = np.maximum(0, predictions)
            
            return predictions[:h]
            
        except Exception as e:
            print(f"    ⚠️  Ошибка прогноза AutoETS: {e}")
            return self._fallback_predict(last_series, h)
    
    def _fallback_predict(self, last_series: pd.Series, horizon: int) -> np.ndarray:
        """
        Fallback прогноз на случай ошибок.
        """
        if self.last_values is not None and len(self.last_values) > 0:
            # Повторяем последний известный сезон
            n_repeats = (horizon // len(self.last_values)) + 1
            predictions = np.tile(self.last_values, n_repeats)[:horizon]
            return predictions
        elif len(last_series) > 0:
            # Возвращаем последнее значение
            return np.full(horizon, last_series.iloc[-1])
        else:
            return np.zeros(horizon)


# Фабрика для создания адаптера
def create_autoets_adapter(forecast_horizon=16, season_length=7):
    """Создает прямой адаптер AutoETS"""
    return AutoETSAdapter(
        forecast_horizon=forecast_horizon,
        season_length=season_length,
    )