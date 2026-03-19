import numpy as np
import pandas as pd
from tqdm import tqdm


def reduce_mem_usage(df) -> tuple:
    """
    Уменьшает использование памяти DataFrame путем оптимизации типов данных.
    Адаптировано для pandas 3.0+ с поддержкой новых типов.

    Args:
        df (pandas.DataFrame): Исходный DataFrame

    Returns:
        df (pandas.DataFrame): DataFrame с оптимизированными типами
        memory_reduction (dict): Словарь с информацией о сокращении памяти
    """

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Начальное использование памяти: {start_mem:.2f} MB")

    for col in tqdm(df.columns, desc="Оптимизация колонок"):
        col_type = df[col].dtype

        # Пропускаем, если уже оптимизировано
        if col_type in ["datetime64[ns]", "category"]:
            continue

        # Обработка строковых колонок (object или string)
        if pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(
            col_type
        ):
            # Проверяем, можно ли преобразовать в категорию
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # Если колонка содержит булевы значения как строки
            if num_unique <= 3 and set(df[col].dropna().unique()).issubset(
                {"True", "False", "true", "false"}
            ):
                df[col] = df[col].astype("boolean")  # pandas boolean with NA support
            # Если колонка содержит даты
            elif col == "date" or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            # Если количество уникальных значений менее 50%, используем категорию
            elif num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")
            # Иначе оставляем как string (pandas 2.0+ string type)
            else:
                df[col] = df[col].astype("string[pyarrow]")

        # Обработка числовых колонок
        elif pd.api.types.is_numeric_dtype(col_type):
            # Проверяем, можно ли преобразовать в целые числа
            col_min = df[col].min()
            col_max = df[col].max()

            # Проверяем, все ли значения целые (с учетом NaN)
            if not pd.api.types.is_integer_dtype(col_type):
                is_integer = True
                try:
                    # Проверяем на NaN и бесконечности
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        # Проверяем, равны ли значения их целой части
                        diff = (non_null - non_null.astype(np.int64)).abs().sum()
                        is_integer = diff < 0.01
                    else:
                        is_integer = True  # Пустая колонка
                except:
                    is_integer = False
            else:
                is_integer = True

            # Обработка целочисленных колонок
            if is_integer:
                # Заполняем NaN минимальным значением - 1 для корректного определения типа
                if df[col].isna().any():
                    fill_val = col_min - 1 if pd.notna(col_min) else -1
                    df[col] = df[col].fillna(fill_val)

                # Выбираем подходящий тип
                if col_min >= 0:
                    if col_max <= np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif col_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif col_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if (
                        col_min >= np.iinfo(np.int8).min
                        and col_max <= np.iinfo(np.int8).max
                    ):
                        df[col] = df[col].astype(np.int8)
                    elif (
                        col_min >= np.iinfo(np.int16).min
                        and col_max <= np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        col_min >= np.iinfo(np.int32).min
                        and col_max <= np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            # Обработка вещественных колонок
            else:
                # Проверяем, можно ли использовать float32
                if col_min > -np.inf and col_max < np.inf:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        # Обработка булевых колонок
        elif pd.api.types.is_bool_dtype(col_type):
            df[col] = df[col].astype("boolean")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Конечное использование памяти: {end_mem:.2f} MB")
    print(f"Сокращение памяти: {(1 - end_mem/start_mem)*100:.1f}%")

    memory_reduction = {
        "start_mb": start_mem,
        "end_mb": end_mem,
        "reduction_percent": (1 - end_mem / start_mem) * 100,
    }

    return df, memory_reduction
