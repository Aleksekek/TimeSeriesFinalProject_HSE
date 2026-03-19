from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


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
