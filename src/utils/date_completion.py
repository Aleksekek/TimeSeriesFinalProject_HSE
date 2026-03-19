from datetime import timedelta
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_missing_dates(
    df: pd.DataFrame,
    store_nbr: int = None,
    item_nbr: int = None,
    n_random_pairs: int = 5,
    date_col: str = "date",
    group_cols: List[str] = ["store_nbr", "item_nbr"],
    plot: bool = True,
):
    """
    Проверяет наличие пропусков в датах для выбранных пар.

    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм с данными
    store_nbr : int, optional
        Номер магазина (если не указан, берет случайные)
    item_nbr : int, optional
        Номер товара (если не указан, берет случайные)
    n_random_pairs : int
        Количество случайных пар для проверки
    date_col : str
        Название колонки с датами
    group_cols : list
        Колонки для группировки
    plot : bool
        Рисовать ли графики
    """

    # Если конкретная пара не указана, берем случайные
    if store_nbr is None or item_nbr is None:
        all_pairs = df.groupby(group_cols).size().reset_index()[group_cols]
        random_pairs = all_pairs.sample(min(n_random_pairs, len(all_pairs)))
        pairs_to_check = random_pairs.values.tolist()
    else:
        pairs_to_check = [(store_nbr, item_nbr)]

    results = []

    for store, item in pairs_to_check:
        # Фильтруем данные для пары
        pair_data = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
        pair_data = pair_data.sort_values(date_col)

        if len(pair_data) == 0:
            print(f"❌ Пара ({store}, {item}) не найдена")
            continue

        # Получаем диапазон дат
        min_date = pair_data[date_col].min()
        max_date = pair_data[date_col].max()

        # Создаем полный диапазон дат
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        # Проверяем пропуски
        existing_dates = set(pair_data[date_col].dt.date)
        expected_dates = set(d.date() for d in all_dates)
        missing_dates = expected_dates - existing_dates

        # Статистика
        n_expected = len(all_dates)
        n_existing = len(pair_data)
        n_missing = len(missing_dates)

        # Проверка на дубликаты
        duplicates = pair_data.duplicated(subset=[date_col]).sum()

        result = {
            "store": store,
            "item": item,
            "min_date": min_date,
            "max_date": max_date,
            "n_expected_days": n_expected,
            "n_existing_records": n_existing,
            "n_missing_days": n_missing,
            "missing_pct": (n_missing / n_expected) * 100 if n_expected > 0 else 0,
            "duplicates": duplicates,
            "missing_dates": sorted(missing_dates)[-10:],  # последние 10 пропусков
        }

        results.append(result)

        # Выводим информацию
        print(f"\n{'='*50}")
        print(f"Пара: магазин {store}, товар {item}")
        print(f"{'='*50}")
        print(f"Период: {min_date.date()} -> {max_date.date()}")
        print(f"Всего дней в периоде: {n_expected}")
        print(f"Записей в данных: {n_existing}")
        print(f"Пропущено дней: {n_missing} ({result['missing_pct']:.2f}%)")
        print(f"Дубликатов дат: {duplicates}")

        if n_missing > 0:
            print(f"Примеры пропущенных дат (последние 10): {result['missing_dates']}")

            # Проверяем, есть ли нулевые продажи в эти дни
            if "unit_sales" in pair_data.columns:
                print("\nПроверка наличия нулевых продаж:")
                zero_sales = (pair_data["unit_sales"] == 0).sum()
                print(f"  Записей с нулевыми продажами: {zero_sales}")

        # Визуализация
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))

            # График временного ряда
            ax1 = axes[0]
            ax1.plot(
                pair_data[date_col],
                pair_data["unit_sales"],
                "b-",
                linewidth=1,
                alpha=0.7,
                label="Продажи",
            )
            ax1.set_title(f"Магазин {store}, Товар {item} - Временной ряд")
            ax1.set_xlabel("Дата")
            ax1.set_ylabel("Продажи")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Heatmap пропусков по дням недели
            ax2 = axes[1]

            # Создаем календарь
            pair_data["day_of_week"] = pd.to_datetime(pair_data[date_col]).dt.dayofweek
            pair_data["week"] = (
                pd.to_datetime(pair_data[date_col]).dt.isocalendar().week
            )
            pair_data["year"] = pd.to_datetime(pair_data[date_col]).dt.year

            # Создаем полную матрицу дней
            all_days_df = pd.DataFrame({"date": all_dates})
            all_days_df["day_of_week"] = all_days_df["date"].dt.dayofweek
            all_days_df["week"] = all_days_df["date"].dt.isocalendar().week
            all_days_df["year"] = all_days_df["date"].dt.year

            # Отмечаем наличие данных
            all_days_df["has_data"] = all_days_df["date"].dt.date.isin(existing_dates)

            # Строим тепловую карту
            pivot = all_days_df.pivot_table(
                values="has_data", index="week", columns="day_of_week", aggfunc="mean"
            )

            im = ax2.imshow(pivot, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            ax2.set_yticks(range(len(pivot.index)))
            ax2.set_yticklabels(pivot.index)
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
            ax2.set_xlabel("День недели")
            ax2.set_ylabel("Неделя года")
            ax2.set_title("Тепловая карта наличия данных (зеленый - есть данные)")

            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


# Функция для проверки конкретной пары
def check_specific_pair(df, store, item, date_col="date"):
    """Проверяет конкретную пару магазин-товар"""
    return check_missing_dates(
        df, store_nbr=store, item_nbr=item, n_random_pairs=1, plot=True
    )


# Функция для проверки случайных пар
def check_random_pairs(df, n_pairs=5):
    """Проверяет случайные пары"""
    return check_missing_dates(df, n_random_pairs=n_pairs, plot=True)


def fill_missing_dates_recent(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: List[str] = ["store_nbr", "item_nbr"],
    target_col: str = "unit_sales",
    fill_value: float = 0.0,
    months_back: int = 3,
    end_date: Optional[pd.Timestamp] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Заполняет пропущенные даты нулями только для последних N месяцев.

    Parameters:
    -----------
    df : pd.DataFrame
        Исходный DataFrame с продажами
    date_col : str
        Название колонки с датами
    group_cols : List[str]
        Колонки, по которым группируем
    target_col : str
        Колонка с целевой переменной
    fill_value : float
        Значение для заполнения пропусков
    months_back : int
        Сколько месяцев назад от end_date брать для заполнения
    end_date : pd.Timestamp, optional
        Конечная дата периода. Если None, берется максимум из данных
    verbose : bool
        Печатать ли прогресс

    Returns:
    --------
    pd.DataFrame
        DataFrame с заполненными пропусками только для нужного периода
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Определяем период для заполнения
    if end_date is None:
        end_date = df[date_col].max()

    start_date = end_date - pd.DateOffset(months=months_back)

    if verbose:
        print(f"Период для заполнения: {start_date} -> {end_date}")
        print(f"Всего дней: {(end_date - start_date).days + 1}")

    # Разделяем данные на "старые" (до start_date) и "новые" (после start_date)
    old_data = df[df[date_col] < start_date].copy()
    recent_data = df[df[date_col] >= start_date].copy()

    if verbose:
        print(f"Старых записей (до {start_date}): {len(old_data)}")
        print(f"Новых записей (после {start_date}): {len(recent_data)}")

    # Полный диапазон дат для периода заполнения
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Получаем все уникальные пары из recent_data
    # (только те, что были активны в последние месяцы)
    active_pairs = recent_data[group_cols].drop_duplicates()

    if verbose:
        print(f"Активных пар в последние {months_back} месяцев: {len(active_pairs)}")

    # Заполняем пропуски только для активных пар
    filled_chunks = []

    # Обрабатываем чанками по 1000 пар для экономии памяти
    chunk_size = 1000
    n_pairs = len(active_pairs)

    for start_idx in tqdm(range(0, n_pairs, chunk_size), desc="Заполнение пропусков"):
        end_idx = min(start_idx + chunk_size, n_pairs)
        pairs_chunk = active_pairs.iloc[start_idx:end_idx]

        # Для каждой пары в чанке создаем полный ряд дат
        chunk_rows = []
        for _, pair in pairs_chunk.iterrows():
            for date in all_dates:
                row = {date_col: date}
                for col in group_cols:
                    row[col] = pair[col]
                row["_temp_flag"] = "filled"  # пометка, что строка добавлена
                chunk_rows.append(row)

        chunk_df = pd.DataFrame(chunk_rows)

        # Мерджим с реальными данными
        chunk_merged = chunk_df.merge(
            recent_data, on=[date_col] + group_cols, how="left"
        )

        # Заполняем пропуски
        chunk_merged[target_col] = chunk_merged[target_col].fillna(fill_value)

        # Убираем временную метку
        chunk_merged = chunk_merged.drop(columns=["_temp_flag"])

        filled_chunks.append(chunk_merged)

    # Объединяем все заполненные чанки
    recent_filled = pd.concat(filled_chunks, ignore_index=True)

    # Объединяем со старыми данными
    result = pd.concat([old_data, recent_filled], ignore_index=True)

    # Сортируем
    result = result.sort_values([date_col] + group_cols).reset_index(drop=True)

    if verbose:
        print(f"\nИтоговая статистика:")
        print(f"  Старых записей: {len(old_data)}")
        print(f"  Заполненных новых записей: {len(recent_filled)}")
        print(f"  Из них реальных: {len(recent_data)}")
        print(f"  Добавлено нулей: {len(recent_filled) - len(recent_data)}")
        print(f"  Всего записей: {len(result)}")

    return result
