from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def fill_missing_dates(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: List[str] = ["store_nbr", "item_nbr"],
    target_col: str = "unit_sales",
    fill_value: float = 0.0,
    chunksize: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Заполняет пропущенные даты для каждой группы нулями (поэтапная версия).
    """

    # Определяем диапазон дат
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    if verbose:
        print(f"Диапазон дат: {min_date} -> {max_date} ({len(all_dates)} дней)")
        print(f"Группировка по: {group_cols}")

    # Получаем все уникальные группы
    groups = df[group_cols].drop_duplicates().reset_index(drop=True)
    n_groups = len(groups)

    if verbose:
        print(f"Всего уникальных групп: {n_groups}")

    # Обрабатываем группы чанками
    result_chunks = []

    for start_idx in tqdm(range(0, n_groups, chunksize), desc="Обработка групп"):
        end_idx = min(start_idx + chunksize, n_groups)
        groups_chunk = groups.iloc[start_idx:end_idx]

        # Создаем сетку для этого чанка групп
        chunk_grid = pd.MultiIndex.from_product(
            [all_dates] + [groups_chunk[col].values for col in group_cols],
            names=[date_col] + group_cols,
        ).to_frame(index=False)

        # Мерджим с данными
        chunk_result = chunk_grid.merge(df, on=[date_col] + group_cols, how="left")

        # Заполняем пропуски
        chunk_result[target_col] = chunk_result[target_col].fillna(fill_value)

        result_chunks.append(chunk_result)

    result = pd.concat(result_chunks, ignore_index=True)

    if verbose:
        print(f"Исходный размер: {len(df)}")
        print(f"Новый размер: {len(result)}")
        print(f"Добавлено строк: {len(result) - len(df)}")

    return result
    return result
