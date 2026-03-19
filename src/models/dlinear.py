import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear Model
    Разлагает ряд на тренд и сезонность, затем применяет линейные слои
    """

    def __init__(self, seq_len, pred_len, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual

        # Разложение на тренд и сезонность через скользящее среднее
        kernel_size = 25
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        if individual:
            # Для каждого признака свой линейный слой
            self.Linear_Season = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(1):  # у нас 1 признак
                self.Linear_Season.append(nn.Linear(seq_len, pred_len))
                self.Linear_Trend.append(nn.Linear(seq_len, pred_len))
        else:
            self.Linear_Season = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = x.squeeze(-1)  # [batch, seq_len]

        # Разложение на тренд (скользящее среднее)
        trend = self.avg_pool(x.unsqueeze(1)).squeeze(1)  # [batch, seq_len]

        # Сезонность = исходный - тренд
        seasonal = x - trend

        if self.individual:
            seasonal_output = torch.zeros([x.size(0), self.pred_len]).to(x.device)
            trend_output = torch.zeros([x.size(0), self.pred_len]).to(x.device)
            seasonal_output = self.Linear_Season[0](seasonal)
            trend_output = self.Linear_Trend[0](trend)
        else:
            seasonal_output = self.Linear_Season(seasonal)
            trend_output = self.Linear_Trend(trend)

        # Суммируем
        output = seasonal_output + trend_output
        return output.unsqueeze(-1)  # [batch, pred_len, 1]


class DLinearAdapter:
    """
    Адаптер для DLinear модели
    """

    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 16,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = None,
    ):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.scaler = StandardScaler()
        self.level_name = None

        print(
            f"DLinear инициализирован (seq_len={seq_len}, pred_len={pred_len}) на {self.device}"
        )

    def set_level_name(self, level_name: str):
        self.level_name = level_name
        return self

    def _prepare_data(self, series_data: pd.DataFrame, fit_scaler: bool = False):
        """
        Подготавливает данные для обучения
        """
        values = series_data["value"].values.astype(np.float32).reshape(-1, 1)

        if fit_scaler:
            # Нормализуем
            values = self.scaler.fit_transform(values).flatten()
        else:
            values = self.scaler.transform(values).flatten()

        # Создаем последовательности
        X, y = [], []
        for i in range(len(values) - self.seq_len - self.pred_len + 1):
            X.append(values[i : i + self.seq_len])
            y.append(values[i + self.seq_len : i + self.seq_len + self.pred_len])

        if len(X) == 0:
            return None, None

        X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        y = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

        return X, y

    def fit(
        self,
        train_data: pd.DataFrame = None,
        val_data: pd.DataFrame = None,
        external_data: Dict = None,
    ):
        """
        Обучает DLinear модель
        """
        if self.level_name:
            print(f"    Обучение DLinear для {self.level_name}")

        if train_data is None or len(train_data) < self.seq_len + self.pred_len:
            if self.level_name:
                print(f"    ⚠️  Недостаточно данных, пропускаем")
            return self

        # Подготавливаем данные
        X, y = self._prepare_data(train_data, fit_scaler=True)

        if X is None or len(X) < self.batch_size:
            if self.level_name:
                print(f"    ⚠️  Недостаточно последовательностей, пропускаем")
            return self

        # Создаем DataLoader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Создаем модель
        self.model = DLinear(
            seq_len=self.seq_len, pred_len=self.pred_len, individual=False
        ).to(self.device)

        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.level_name and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"      Эпоха {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        if self.level_name:
            print(f"    ✅ Обучение завершено")

        return self

    def predict(
        self, horizon: int, last_data: pd.DataFrame, external_data: Dict = None
    ) -> np.ndarray:
        """
        Делает прогноз
        """
        if self.model is None:
            # Если модель не обучена, возвращаем последнее значение
            return np.full(horizon, last_data["value"].iloc[-1])

        # Берем последние seq_len значений
        last_values = last_data["value"].values[-self.seq_len :].reshape(-1, 1)

        if len(last_values) < self.seq_len:
            return np.full(horizon, last_values[-1][0])

        # Нормализуем
        last_values_norm = self.scaler.transform(last_values).flatten()

        # Преобразуем в тензор
        X = (
            torch.FloatTensor(last_values_norm)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )

        # Прогнозируем
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X).cpu().numpy().squeeze()

        # Возвращаем в исходный масштаб
        pred = self.scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()

        # Если прогноз короче, чем нужно, дополняем последним значением
        if len(pred) < horizon:
            pred = np.append(pred, [pred[-1]] * (horizon - len(pred)))

        return np.maximum(0, pred)


# Фабрика для создания адаптера
def create_dlinear_adapter(seq_len=60, pred_len=16, epochs=30):
    """Создает адаптер DLinear"""
    return DLinearAdapter(
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
