import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class HierarchicalForecaster:
    """
    Иерархический прогнозист, который агрегирует данные по кластерам,
    обучает модели на агрегированных рядах и дизагрегирует прогнозы.
    """

    def __init__(self, stores_df: pd.DataFrame, items_df: pd.DataFrame):
        """
        Parameters:
        -----------
        stores_df : pd.DataFrame
            DataFrame с информацией о магазинах (store_nbr, cluster, etc.)
        items_df : pd.DataFrame
            DataFrame с информацией о товарах (item_nbr, family, etc.)
        """
        self.stores = stores_df
        self.items = items_df

        # Создаем маппинги
        self.store_to_cluster = stores_df.set_index("store_nbr")["cluster"].to_dict()
        self.item_to_family = items_df.set_index("item_nbr")["family"].to_dict()

        # Для хранения агрегированных рядов и прогнозов
        self.aggregated_series = {}
        self.cluster_family_proportions = {}
        self.trained_models = {}

        # Для нормализации
        self.scale_factors = {}
        self.store_proportions = {}
        self.item_proportions = {}

        # Информация о количестве для нормализации
        self.n_items_total = None
        self.n_stores_total = None

    def create_hierarchy(self, train_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Создает иерархию агрегированных временных рядов с использованием СРЕДНИХ значений.

        Returns:
        --------
        Dict с уровнями:
            - 'total': средние продажи по всем магазинам и товарам
            - 'by_cluster': средние продажи по кластерам магазинов
            - 'by_cluster_family': средние продажи по кластерам и семействам
        """
        print("Создание иерархии...")

        # Добавляем метаданные
        train_with_meta = train_data.copy()
        train_with_meta["cluster"] = train_with_meta["store_nbr"].map(
            self.store_to_cluster
        )
        train_with_meta["family"] = train_with_meta["item_nbr"].map(self.item_to_family)

        # Запоминаем общее количество уникальных товаров и магазинов
        self.n_items_total = train_data["item_nbr"].nunique()
        self.n_stores_total = train_data["store_nbr"].nunique()

        print(f"  Всего уникальных товаров: {self.n_items_total}")
        print(f"  Всего уникальных магазинов: {self.n_stores_total}")

        # Уровень 0: Total sales - используем СРЕДНИЕ продажи
        total_series = (
            train_with_meta.groupby("date")["unit_sales"].mean().reset_index()
        )
        total_series.columns = ["date", "total_sales"]
        print(f"  Средние продажи total: {total_series['total_sales'].mean():.2f}")

        # Уровень 1: By cluster - средние по кластерам
        cluster_series = (
            train_with_meta.groupby(["date", "cluster"])["unit_sales"]
            .mean()
            .reset_index()
        )

        # Уровень 2: By cluster + family - средние по кластерам и семействам
        cluster_family_series = (
            train_with_meta.groupby(["date", "cluster", "family"])["unit_sales"]
            .mean()
            .reset_index()
        )

        # Сохраняем пропорции для дизагрегации
        self._compute_proportions(train_with_meta)

        self.aggregated_series = {
            "total": total_series,
            "by_cluster": cluster_series,
            "by_cluster_family": cluster_family_series,
        }

        print(f"  Total: {len(total_series)} дней")
        print(f"  By cluster: {cluster_series['cluster'].nunique()} кластеров")
        print(
            f"  By cluster+family: {cluster_family_series.groupby(['cluster', 'family']).ngroups} групп"
        )

        return self.aggregated_series

    def _compute_proportions(self, train_with_meta: pd.DataFrame):
        """
        Вычисляет пропорции для дизагрегации.
        """
        # Пропорции для store внутри cluster (на основе количества, а не сумм)
        store_counts = train_with_meta.groupby(["cluster", "store_nbr"]).size()
        cluster_counts = train_with_meta.groupby("cluster").size()
        self.store_proportions = (store_counts / cluster_counts).fillna(0).to_dict()

        # Пропорции для item внутри (cluster, family) (на основе количества)
        item_counts = train_with_meta.groupby(["cluster", "family", "item_nbr"]).size()
        family_counts = train_with_meta.groupby(["cluster", "family"]).size()
        self.item_proportions = (item_counts / family_counts).fillna(0).to_dict()

        # Масштабы: средние продажи на каждом уровне
        self.scale_factors = {
            "total": train_with_meta["unit_sales"].mean(),
            "cluster": train_with_meta.groupby("cluster")["unit_sales"]
            .mean()
            .to_dict(),
            "cluster_family": train_with_meta.groupby(["cluster", "family"])[
                "unit_sales"
            ]
            .mean()
            .to_dict(),
        }

        print("Пропорции вычислены:")
        print(f"  Средние продажи (total): {self.scale_factors['total']:.2f}")
        print(f"  Уникальных store_proportions: {len(self.store_proportions)}")
        print(f"  Уникальных item_proportions: {len(self.item_proportions)}")

    def fit_aggregated(
        self,
        model_class,
        model_params: Dict = None,
        max_history: Optional[int] = 90,
        external_data: Dict = None,
        train_data: pd.DataFrame = None,  # исходные данные для создания признаков
        val_data: pd.DataFrame = None,  # валидационные данные для создания признаков
    ):
        """
        Обучает модели на агрегированных рядах с поддержкой внешних признаков.

        Parameters:
        -----------
        model_class : class
            Класс модели (может быть адаптером CatBoost с методом fit, принимающим external_data)
        model_params : Dict
            Параметры модели
        max_history : int, optional
            Максимальная длина истории
        external_data : Dict, optional
            Словарь с внешними данными (нефть, праздники, транзакции)
        train_data : pd.DataFrame, optional
            Исходные тренировочные данные для создания признаков (нужны для CatBoost)
        val_data : pd.DataFrame, optional
            Исходные валидационные данные для создания признаков (нужны для CatBoost)
        """
        print("Обучение агрегированных моделей...")

        # Проверяем, нужно ли использовать внешние признаки
        use_external = external_data is not None and train_data is not None

        # Функция для подготовки данных для конкретного уровня
        def prepare_level_data(series_df, value_col, level_name):
            """Подготавливает DataFrame с датой и значением для конкретного уровня"""
            df = series_df.copy()
            df = df.rename(columns={value_col: "value"})
            return df[["date", "value"]]

        # Обучаем на total уровне
        print("  Уровень Total...")
        total_model = model_class(**(model_params or {}))

        if (
            use_external
            and hasattr(total_model, "fit")
            and "external_data" in total_model.fit.__code__.co_varnames
        ):
            # Если модель умеет принимать external_data
            total_series = prepare_level_data(
                self.aggregated_series["total"], "total_sales", "total"
            )

            # Разделяем на train/val для этого уровня
            if val_data is not None:
                total_train = total_series[
                    total_series["date"] <= train_data["date"].max()
                ]
                total_val = total_series[
                    total_series["date"] > train_data["date"].max()
                ]
            else:
                total_train = total_series
                total_val = None

            total_model.fit(
                train_data=total_train, val_data=total_val, external_data=external_data
            )
        else:
            # Стандартное обучение на массиве
            y_total = self.aggregated_series["total"]["total_sales"].values
            if max_history:
                y_total = y_total[-max_history:]
            total_model.fit(y_total)

        self.trained_models["total"] = total_model

        # Обучаем на cluster уровне
        print("  Уровень Clusters...")
        cluster_models = {}

        for cluster in self.aggregated_series["by_cluster"]["cluster"].unique():
            cluster_data = self.aggregated_series["by_cluster"][
                self.aggregated_series["by_cluster"]["cluster"] == cluster
            ].sort_values("date")

            model = model_class(**(model_params or {}))

            if (
                use_external
                and hasattr(model, "fit")
                and "external_data" in model.fit.__code__.co_varnames
            ):
                # Используем внешние признаки
                cluster_series = prepare_level_data(
                    cluster_data, "unit_sales", f"cluster_{cluster}"
                )

                if val_data is not None:
                    cluster_train = cluster_series[
                        cluster_series["date"] <= train_data["date"].max()
                    ]
                    cluster_val = cluster_series[
                        cluster_series["date"] > train_data["date"].max()
                    ]
                else:
                    cluster_train = cluster_series
                    cluster_val = None

                # Добавляем информацию о кластере в external_data для этого уровня
                level_external = external_data.copy() if external_data else None
                if level_external and "transactions" in level_external:
                    # Фильтруем транзакции по кластеру
                    level_external["transactions"] = level_external["transactions"][
                        level_external["transactions"]["cluster"] == cluster
                    ]

                model.fit(
                    train_data=cluster_train,
                    val_data=cluster_val,
                    external_data=level_external,
                )
            else:
                # Стандартное обучение
                y_cluster = cluster_data["unit_sales"].values
                if max_history:
                    y_cluster = y_cluster[-max_history:]
                model.fit(y_cluster)

            cluster_models[cluster] = model

        self.trained_models["clusters"] = cluster_models

        # Обучаем на cluster+family уровне
        print("  Уровень Cluster+Family...")
        cf_models = {}
        cf_groups = self.aggregated_series["by_cluster_family"].groupby(
            ["cluster", "family"]
        )

        total_clusters = len(list(cf_groups))
        processed = 0

        for (cluster, family), group in cf_groups:
            processed += 1
            if processed % 100 == 0:
                print(f"    Обработано {processed}/{total_clusters} групп")

            group = group.sort_values("date")

            # Пропускаем группы с малым количеством данных
            if len(group) < 14:
                continue

            model = model_class(**(model_params or {}))

            if (
                use_external
                and hasattr(model, "fit")
                and "external_data" in model.fit.__code__.co_varnames
            ):
                # Используем внешние признаки
                cf_series = prepare_level_data(
                    group, "unit_sales", f"cluster_{cluster}_family_{family}"
                )

                if val_data is not None:
                    cf_train = cf_series[cf_series["date"] <= train_data["date"].max()]
                    cf_val = cf_series[cf_series["date"] > train_data["date"].max()]
                else:
                    cf_train = cf_series
                    cf_val = None

                # Подготавливаем external_data для этого уровня
                level_external = external_data.copy() if external_data else None
                if level_external:
                    if "transactions" in level_external:
                        level_external["transactions"] = level_external["transactions"][
                            level_external["transactions"]["cluster"] == cluster
                        ]
                    # Можно добавить специфичные для family признаки

                model.fit(
                    train_data=cf_train, val_data=cf_val, external_data=level_external
                )
            else:
                # Стандартное обучение
                y_cf = group["unit_sales"].values
                if max_history:
                    y_cf = y_cf[-max_history:]
                model.fit(y_cf)

            cf_models[(cluster, family)] = model

        self.trained_models["cluster_family"] = cf_models

        print(f"Обучено моделей:")
        print(f"  Total: 1")
        print(f"  Clusters: {len(cluster_models)}")
        print(f"  Cluster+Family: {len(cf_models)}")

        # Возвращаем статистику использования external_data
        if use_external:
            print(f"\nИспользованы внешние признаки: {list(external_data.keys())}")

    def _extract_forecast(self, pred, horizon: int) -> np.ndarray:
        """
        Универсальная функция для извлечения прогноза в виде массива.
        """
        # Если это уже список или массив
        if isinstance(pred, (list, np.ndarray)):
            if len(pred) >= horizon:
                return np.array(pred[:horizon])
            else:
                last_val = pred[-1] if len(pred) > 0 else 0
                return np.array(list(pred) + [last_val] * (horizon - len(pred)))

        # Если это pandas Series или DataFrame
        elif isinstance(pred, (pd.Series, pd.DataFrame)):
            try:
                values = pred.values
                if len(values) >= horizon:
                    return np.array(values[:horizon]).flatten()
                else:
                    last_val = values[-1] if len(values) > 0 else 0
                    return np.array(
                        list(values) + [last_val] * (horizon - len(values))
                    ).flatten()
            except:
                print(f"    Не удалось обработать pandas объект: {type(pred)}")
                return np.zeros(horizon)

        # Если это StatsForecast результат (объект с методом values)
        elif hasattr(pred, "values") and callable(getattr(pred, "values")):
            try:
                values = pred.values()
                if len(values) >= horizon:
                    return np.array(values[:horizon])
                else:
                    last_val = values[-1] if len(values) > 0 else 0
                    return np.array(list(values) + [last_val] * (horizon - len(values)))
            except:
                print(f"    Не удалось обработать StatsForecast объект")
                return np.zeros(horizon)

        # Если это словарь (например, AutoETS возвращает {'mean': [...]})
        elif isinstance(pred, dict):
            # Сначала ищем 'mean' (как в AutoETS)
            if "mean" in pred:
                return self._extract_forecast(pred["mean"], horizon)
            else:
                # Берем первое значение из словаря
                first_key = list(pred.keys())[0]
                return self._extract_forecast(pred[first_key], horizon)

        # Если это скаляр
        else:
            try:
                val = float(pred)
                return np.full(horizon, val)
            except:
                print(f"  Неизвестный тип прогноза: {type(pred)}")
                return np.zeros(horizon)

    def predict_aggregated(
        self, horizon: int, start_date: pd.Timestamp, external_data: Dict = None
    ) -> Dict:
        """
        Делает прогнозы на всех уровнях иерархии с использованием внешних данных.

        Parameters:
        -----------
        horizon : int
            Горизонт прогнозирования
        start_date : pd.Timestamp
            Дата начала прогноза
        external_data : Dict, optional
            Внешние данные для прогноза (нефть, праздники и т.д.)

        Returns:
        --------
        Dict
            Словарь с прогнозами
        """
        print(f"Прогнозирование на {horizon} дней...")

        # Проверяем, нужно ли использовать внешние признаки
        use_external = external_data is not None

        # Функция для получения последних данных для уровня
        def get_last_data(series_df, value_col, n_days=60):
            """Возвращает последние n_days дней для создания признаков"""
            df = series_df.copy()
            df = df.rename(columns={value_col: "value"})
            return df[["date", "value"]].tail(n_days)

        # Total level
        total_model = self.trained_models["total"]
        if (
            use_external
            and hasattr(total_model, "predict")
            and "external_data" in total_model.predict.__code__.co_varnames
        ):
            total_last = get_last_data(self.aggregated_series["total"], "total_sales")
            total_forecast = total_model.predict(horizon, total_last, external_data)
        else:
            total_forecast = self._extract_forecast(
                total_model.predict(horizon), horizon
            )

        # Cluster level
        cluster_forecasts = {}
        for cluster, model in self.trained_models["clusters"].items():
            try:
                if (
                    use_external
                    and hasattr(model, "predict")
                    and "external_data" in model.predict.__code__.co_varnames
                ):
                    cluster_last = get_last_data(
                        self.aggregated_series["by_cluster"][
                            self.aggregated_series["by_cluster"]["cluster"] == cluster
                        ],
                        "unit_sales",
                    )
                    # Фильтруем external_data для этого кластера
                    cluster_external = external_data.copy() if external_data else None
                    if cluster_external and "transactions" in cluster_external:
                        cluster_external["transactions"] = cluster_external[
                            "transactions"
                        ][cluster_external["transactions"]["cluster"] == cluster]
                    cluster_forecasts[cluster] = model.predict(
                        horizon, cluster_last, cluster_external
                    )
                else:
                    pred_raw = model.predict(horizon)
                    cluster_forecasts[cluster] = self._extract_forecast(
                        pred_raw, horizon
                    )
            except Exception as e:
                print(f"  Ошибка прогноза для кластера {cluster}: {e}")
                cluster_forecasts[cluster] = total_forecast.copy()

        # Cluster+Family level
        cf_forecasts = {}
        for (cluster, family), model in self.trained_models["cluster_family"].items():
            try:
                if (
                    use_external
                    and hasattr(model, "predict")
                    and "external_data" in model.predict.__code__.co_varnames
                ):
                    cf_last = get_last_data(
                        self.aggregated_series["by_cluster_family"][
                            (
                                self.aggregated_series["by_cluster_family"]["cluster"]
                                == cluster
                            )
                            & (
                                self.aggregated_series["by_cluster_family"]["family"]
                                == family
                            )
                        ],
                        "unit_sales",
                    )
                    # Фильтруем external_data
                    cf_external = external_data.copy() if external_data else None
                    if cf_external and "transactions" in cf_external:
                        cf_external["transactions"] = cf_external["transactions"][
                            cf_external["transactions"]["cluster"] == cluster
                        ]
                    cf_forecasts[(cluster, family)] = model.predict(
                        horizon, cf_last, cf_external
                    )
                else:
                    pred_raw = model.predict(horizon)
                    cf_forecasts[(cluster, family)] = self._extract_forecast(
                        pred_raw, horizon
                    )
            except Exception as e:
                print(f"  Ошибка прогноза для ({cluster}, {family}): {e}")
                cf_forecasts[(cluster, family)] = cluster_forecasts.get(
                    cluster, total_forecast.copy()
                )

        print(f"  Всего кластеров: {len(cluster_forecasts)}")
        print(f"  Всего cluster+family: {len(cf_forecasts)}")

        return {
            "total": total_forecast,
            "clusters": cluster_forecasts,
            "cluster_family": cf_forecasts,
            "dates": pd.date_range(start=start_date, periods=horizon),
        }

    def disaggregate_to_store_item(
        self, forecasts: Dict, test_pairs: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """
        Дизагрегирует прогнозы до уровня магазин-товар.
        """
        print("Дизагрегация прогнозов до уровня магазин-товар...")

        results = []
        dates = forecasts["dates"]
        horizon = len(dates)

        # Получаем средние значения для масштабирования
        total_avg = self.scale_factors.get("total", 1.0)
        cluster_avg = self.scale_factors.get("cluster", {})
        cf_avg = self.scale_factors.get("cluster_family", {})

        print(f"  Средние продажи (total): {total_avg:.2f}")
        print(f"  Форма total прогноза: {forecasts['total'].shape}")

        total_pred = forecasts["total"]
        cluster_preds = forecasts["clusters"]
        cf_preds = forecasts["cluster_family"]

        # Счетчики для отладки
        total_pairs = len(test_pairs)
        debug_interval = max(1, total_pairs // 10)

        for idx, (store, item) in enumerate(test_pairs):
            if idx % debug_interval == 0:
                pct = (idx / total_pairs) * 100
                print(f"  Прогресс: {pct:.1f}% ({idx}/{total_pairs})")

            cluster = self.store_to_cluster.get(store)
            family = self.item_to_family.get(item)

            if cluster is None or family is None:
                # Fallback
                for i, date in enumerate(dates):
                    predicted = total_pred[i] * (1.0 / self.n_items_total)
                    results.append(
                        {
                            "store_nbr": store,
                            "item_nbr": item,
                            "date": date,
                            "predicted": max(0, min(predicted, 100)),
                        }
                    )
                continue

            # Получаем прогнозы
            cluster_pred = cluster_preds.get(cluster, total_pred)
            cf_pred = cf_preds.get((cluster, family), cluster_pred)

            # Коэффициенты масштабирования
            cluster_factor = cluster_avg.get(cluster, total_avg) / total_avg
            cf_factor = cf_avg.get(
                (cluster, family), cluster_avg.get(cluster, total_avg)
            ) / cluster_avg.get(cluster, total_avg)

            # Количество товаров в семействе и магазинов в кластере
            items_in_family = len(
                [
                    k
                    for k in self.item_proportions.keys()
                    if k[0] == cluster and k[1] == family
                ]
            )
            stores_in_cluster = len(
                [k for k in self.store_proportions.keys() if k[0] == cluster]
            )

            item_share = (
                1.0 / items_in_family
                if items_in_family > 0
                else 1.0 / self.n_items_total
            )
            store_share = (
                1.0 / stores_in_cluster
                if stores_in_cluster > 0
                else 1.0 / self.n_stores_total
            )

            for i, date in enumerate(dates):
                predicted = (
                    cf_pred[i] * item_share * cf_factor * 0.7
                    + cluster_pred[i] * store_share * cluster_factor * 0.3
                )
                predicted = max(0, min(predicted, 100))

                results.append(
                    {
                        "store_nbr": store,
                        "item_nbr": item,
                        "date": date,
                        "predicted": predicted,
                    }
                )

        print(f"  Дизагрегация завершена. Всего прогнозов: {len(results)}")
        return pd.DataFrame(results)

    def select_representative_pairs(self, n_pairs: int = 1000) -> List[Tuple[int, int]]:
        """
        Выбирает N самых репрезентативных пар магазин-товар.
        """
        print(f"Выбор {n_pairs} репрезентативных пар...")
        # Пока заглушка
        return []
