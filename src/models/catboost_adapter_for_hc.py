from typing import Dict, List, Optional, Tuple

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


class CatBoostHierarchicalAdapter:
    """
    Адаптер для CatBoost, который работает с агрегированными рядами в иерархическом пайплайне.
    """

    def __init__(
        self,
        model_params: Dict = None,
        horizon: int = 16,
        history: int = 60,
        use_external_features: bool = True,
    ):
        """
        Parameters:
        -----------
        model_params : Dict
            Параметры для CatBoost
        horizon : int
            Горизонт прогнозирования
        history : int
            Размер окна истории для создания лагов
        use_external_features : bool
            Использовать ли внешние признаки (праздники, нефть и т.д.)
        """
        self.model_params = model_params or {
            "iterations": 800,
            "learning_rate": 0.03,
            "depth": 5,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 50,
        }
        self.horizon = horizon
        self.history = history
        self.use_external_features = use_external_features
        self.model = None
        self.feature_names = None
        self.level_name = None

    def set_level_name(self, level_name: str):
        """Устанавливает имя уровня для логирования"""
        self.level_name = level_name
        return self

    def _create_features(
        self, series_data: pd.DataFrame, external_data: Dict = None
    ) -> pd.DataFrame:
        """
        Создает признаки для временного ряда.
        """
        df = series_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Базовые временные признаки
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        # Лаги
        for lag in [1, 2, 3, 7, 14]:
            if lag <= self.history:
                df[f"lag_{lag}"] = df["value"].shift(lag)

        # Скользящие статистики
        for window in [7, 14]:
            if window <= self.history:
                df[f"rolling_mean_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).mean()
                )
                df[f"rolling_std_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).std()
                )
                df[f"rolling_max_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).max()
                )
                df[f"rolling_min_{window}"] = (
                    df["value"].rolling(window=window, min_periods=1).min()
                )

        # Разности
        df["diff_1"] = df["value"].diff(1)
        df["diff_7"] = df["value"].diff(7)
        df["diff_14"] = df["value"].diff(14)

        # Отношения
        if "rolling_mean_7" in df.columns:
            df["value_to_mean_7"] = df["value"] / (df["rolling_mean_7"] + 1e-8)

        # Внешние признаки (только если они есть)
        if self.use_external_features and external_data is not None:

            # Нефть - числовые признаки
            if "oil" in external_data:
                oil_cols = [
                    "oil_price",
                    "oil_price_lag1",
                    "oil_price_lag7",
                    "oil_price_change",
                ]
                existing_oil_cols = [
                    c for c in oil_cols if c in external_data["oil"].columns
                ]
                if existing_oil_cols:
                    oil_data = external_data["oil"][["date"] + existing_oil_cols]
                    df = df.merge(oil_data, on="date", how="left")
                    for col in existing_oil_cols:
                        if col in df.columns:
                            df[col] = (
                                df[col]
                                .fillna(method="ffill")
                                .fillna(method="bfill")
                                .fillna(0)
                            )

            # Транзакции - числовые признаки (только для кластеров)
            if (
                "transactions" in external_data
                and self.level_name
                and "cluster" in self.level_name
            ):
                import re

                cluster_match = re.search(r"cluster_(\d+)", self.level_name)
                if cluster_match:
                    cluster = int(cluster_match.group(1))
                    cluster_trans = external_data["transactions"][
                        external_data["transactions"]["cluster"] == cluster
                    ].copy()

                    if (
                        len(cluster_trans) > 0
                        and "transactions" in cluster_trans.columns
                    ):
                        # Берем только числовые колонки
                        trans_data = cluster_trans[["date", "transactions"]]
                        df = df.merge(trans_data, on="date", how="left")
                        df["transactions"] = (
                            df["transactions"].fillna(method="ffill").fillna(0)
                        )

                        # Лаги транзакций
                        df["transactions_lag1"] = df["transactions"].shift(1).fillna(0)
                        df["transactions_lag7"] = df["transactions"].shift(7).fillna(0)

        # Удаляем все нечисловые колонки, кроме явно указанных категориальных
        categorical_cols = [
            "day",
            "dayofweek",
            "weekend",
            "week_of_year",
            "is_holiday",
            "holiday_tomorrow",
            "holiday_yesterday",
        ]

        for col in df.columns:
            if col not in ["date", "value"] + categorical_cols:
                if df[col].dtype == "object":
                    # Пытаемся преобразовать в числовой, если не получается - удаляем
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                    except:
                        df = df.drop(columns=[col])

        return df

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        external_data: Dict = None,
    ):
        """
        Обучает CatBoost на агрегированных рядах.
        """
        if self.level_name:
            print(f"    Обучение CatBoost для {self.level_name}")

        # Создаем признаки
        train_features = self._create_features(train_data, external_data)

        # Удаляем строки с NaN (первые history дней)
        train_features = train_features.dropna().reset_index(drop=True)

        # Проверяем, что после удаления NaN остались данные
        if len(train_features) < 10:
            if self.level_name:
                print(
                    f"    ⚠️  Недостаточно данных для обучения (всего {len(train_features)} строк). Используем fallback."
                )
            self.model = DummyRegressor(strategy="mean")
            if len(train_features) > 0:
                self.model.fit(train_features[["value"]], train_features["value"])
            else:
                self.model.fit([[0]], [0])
            self.feature_names = ["value"]
            return

        # Целевая переменная
        target_col = "value"

        # Проверяем, не константная ли целевая переменная
        if train_features[target_col].nunique() == 1:
            if self.level_name:
                print(
                    f"    ⚠️  Все значения таргета одинаковы ({train_features[target_col].iloc[0]:.2f}). Используем fallback."
                )
            self.model = DummyRegressor(
                strategy="constant", constant=train_features[target_col].iloc[0]
            )
            self.model.fit(
                train_features[["value"]].iloc[:1], train_features[target_col].iloc[:1]
            )
            self.feature_names = ["value"]
            return

        # Признаки (все кроме date и value)
        feature_cols = [
            col for col in train_features.columns if col not in ["date", "value"]
        ]

        # Проверяем, что признаки не пустые
        if len(feature_cols) == 0:
            if self.level_name:
                print(f"    ⚠️  Нет признаков для обучения. Используем fallback.")
            self.model = DummyRegressor(strategy="mean")
            self.model.fit(train_features[["value"]], train_features["value"])
            self.feature_names = ["value"]
            return

        X_train = train_features[feature_cols]
        y_train = train_features[target_col]

        # Определяем категориальные признаки (индексы)
        categorical_features = [
            "day",
            "dayofweek",
            "weekend",
            "week_of_year",
            "is_holiday",
            "holiday_tomorrow",
            "holiday_yesterday",
        ]
        cat_features = []
        for i, col in enumerate(feature_cols):
            if col in categorical_features:
                cat_features.append(i)

        # Создаем пул для валидации, если есть данные
        eval_set = None
        if val_data is not None and len(val_data) > 10:
            val_features = self._create_features(val_data, external_data)
            val_features = val_features.dropna().reset_index(drop=True)

            if len(val_features) >= 5 and val_features[target_col].nunique() > 1:
                X_val = val_features[feature_cols]
                y_val = val_features[target_col]
                eval_set = (X_val, y_val)

        # Настраиваем параметры модели
        model_params = self.model_params.copy()

        # Важно: добавляем параметры для отображения прогресса
        if "verbose" not in model_params:
            model_params["verbose"] = False

        # Если таргет имеет очень маленькую дисперсию, уменьшаем learning rate
        if y_train.std() < 0.01:
            if self.level_name:
                print(
                    f"    ⚠️  Очень маленькая дисперсия таргета ({y_train.std():.6f}). Уменьшаем learning rate."
                )
            model_params["learning_rate"] = min(
                model_params.get("learning_rate", 0.03), 0.01
            )

        try:
            # Создаем модель
            self.model = cb.CatBoostRegressor(**model_params)

            # Создаем обучающий пул с указанием категориальных признаков
            train_pool = cb.Pool(
                data=X_train,
                label=y_train,
                cat_features=cat_features if cat_features else None,
            )

            # Обучаем
            if eval_set:
                val_pool = cb.Pool(
                    data=X_val,
                    label=y_val,
                    cat_features=cat_features if cat_features else None,
                )

                self.model.fit(
                    train_pool,
                    eval_set=val_pool,
                    use_best_model=True,
                    verbose_eval=False,
                    plot=False,
                )
            else:
                self.model.fit(
                    train_pool, use_best_model=False, verbose_eval=False, plot=False
                )

            self.feature_names = feature_cols

            if self.level_name:
                best_iter = self.model.get_best_iteration()
                if best_iter is not None:
                    print(f"    ✅ Обучение завершено. Лучшая итерация: {best_iter}")
                else:
                    print(
                        f"    ✅ Обучение завершено. Использовано итераций: {self.model.tree_count_}"
                    )

        except Exception as e:
            if "All train targets are equal" in str(e):
                if self.level_name:
                    print(f"    ⚠️  Все таргеты одинаковы. Используем fallback.")
                const_value = y_train.iloc[0] if len(y_train) > 0 else 0
                self.model = DummyRegressor(strategy="constant", constant=const_value)
                self.model.fit([[0]], [const_value])
                self.feature_names = ["value"]
            else:
                # Если другая ошибка, пробрасываем дальше
                raise e

    def predict(
        self, horizon: int, last_data: pd.DataFrame, external_data: Dict = None
    ) -> np.ndarray:
        """
        Делает прогноз на horizon шагов вперед.
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Для DummyRegressor или других простых моделей
        if hasattr(self.model, "predict") and not hasattr(
            self.model, "get_best_iteration"
        ):
            # Это простая модель, которая не умеет работать с признаками
            if hasattr(self.model, "constant_"):
                const_value = self.model.constant_[0]
            else:
                const_value = last_data["value"].iloc[-1] if len(last_data) > 0 else 0
            return np.full(horizon, const_value)

        # Для CatBoost
        predictions = []
        current_data = last_data.copy()

        # Определяем, какие признаки реально доступны
        available_features = []
        if hasattr(self, "feature_names") and self.feature_names is not None:
            # Проверяем каждый признак
            for f in self.feature_names:
                if f in ["date", "value"]:
                    continue
                available_features.append(f)

        if self.level_name:
            print(
                f"    Прогнозирование для {self.level_name}, признаков: {len(available_features)}"
            )

        for step in range(horizon):
            # Создаем признаки для текущего шага
            features = self._create_features(current_data, external_data)

            if len(features) == 0:
                # Если нет данных, возвращаем последнее известное значение
                pred = current_data["value"].iloc[-1] if len(current_data) > 0 else 0
            else:
                # Берем последнюю строку
                last_row = features.iloc[-1:].copy()

                # Выбираем только те признаки, которые есть и в модели, и в данных
                valid_features = []
                for f in available_features:
                    if f in last_row.columns:
                        valid_features.append(f)
                    else:
                        if self.level_name and step == 0:
                            print(
                                f"      Предупреждение: признак {f} отсутствует в данных"
                            )

                if len(valid_features) == 0:
                    # Если нет ни одного признака, используем последнее значение
                    pred = (
                        current_data["value"].iloc[-1] if len(current_data) > 0 else 0
                    )
                else:
                    last_features = last_row[valid_features]

                    # Проверяем, что все признаки есть
                    if last_features.isna().any().any():
                        # Заполняем пропуски
                        last_features = last_features.fillna(0)

                    # Прогнозируем
                    try:
                        pred = self.model.predict(last_features)[0]
                    except Exception as e:
                        if self.level_name and step == 0:
                            print(f"      Ошибка прогноза: {e}")
                        # В случае ошибки используем последнее значение
                        pred = (
                            current_data["value"].iloc[-1]
                            if len(current_data) > 0
                            else 0
                        )

            predictions.append(pred)

            # Добавляем прогноз в данные для следующего шага
            next_date = current_data["date"].max() + pd.Timedelta(days=1)
            new_row = pd.DataFrame({"date": [next_date], "value": [pred]})
            current_data = pd.concat([current_data, new_row], ignore_index=True)

        return np.array(predictions)
