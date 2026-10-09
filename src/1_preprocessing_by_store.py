#!/usr/bin/env python
"""按门店生成 M5 A1 特征，保持官方特征语义且避免全局长表。"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = RAW_DIR / "processed"
LOG_DIR = PROJECT_DIR / "logs"
END_TRAIN = 1941
TOTAL_DAYS = 1969
ENCODING_END = 1913
ALL_STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
INDEX_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FILES = ["grid_part_1.pkl", "grid_part_2.pkl", "grid_part_3.pkl", "lags_df_28.pkl", "mean_encoding_df.pkl"]
BASE_FILES = FILES[:4]
ENCODING_GROUPS = [
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]
SCHEMA_VERSION = 1


def configure_logging() -> logging.Logger:
    '''
    Configure the logger.
    Returns:
        logging.Logger: The configured logger.
    The output is json format.
    '''
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5_preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / "preprocessing_by_store.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


LOGGER = logging.getLogger("m5_preprocessing")


def log_event(event: str, **values: object) -> None:
    '''
    Log an event using json.
    '''
    global LOGGER
    if not LOGGER.handlers: LOGGER = configure_logging()
    payload = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **values}
    LOGGER.info(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))


def peak_rss_mb() -> float:
    '''
    Return the peak memory usage of the current process in MB.
    '''
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dataframe_mb(df: pd.DataFrame) -> float:
    '''
    Return the memory usage of a dataframe in MB.
    '''
    return float(df.memory_usage(index=True, deep=True).sum() / 1024**2)


def parse_stores() -> list[str]:
    '''
    Parse the stores to be processed.
    Returns:
        list[str]: The stores to be processed.
    '''
    value = os.environ.get("M5_STORES", "").strip()
    requested = ALL_STORES if not value else [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(ALL_STORES))
    if unknown:
        raise ValueError(f"M5_STORES 包含未知门店: {unknown}")
    requested_set = set(requested)
    return [store for store in ALL_STORES if store in requested_set]


def atomic_pickle(obj: object, path: Path) -> None:
    '''
    Save a pickle file atomically.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        pd.to_pickle(obj, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    '''
    Save a json file atomically.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def index_signature(index: pd.Index) -> str:
    '''
    Return a hex digest of the hash of the sorted index.
    Verify that the row order of the five tables is identical 
    and that the category mapping remains consistent across multiple runs.
    '''
    hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def category_signature(vocabs: dict[str, list[object]]) -> str:
    '''
    Return a hex digest of the hash of the sorted index.
    Verify that the row order of the five tables is identical 
    and that the category mapping remains consistent across multiple runs.
    '''
    serialized = json.dumps(vocabs, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def metadata_path(store: str) -> Path:
    '''
    Return the path to the metadata file for a store.
    '''
    return PROCESSED_DIR / store / "metadata.json"


def read_metadata(store: str) -> dict[str, object] | None:
    '''
    Read the metadata for a store.
    Returns:
        dict[str, object] | None: The metadata or None if the file does not exist or is invalid.
    '''
    path = metadata_path(store)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def validate_store_files(store: str, expected_stage: str, category_hash: str | None = None) -> bool:
    '''
    Validate the store files.
    Returns:
        bool: True if the files are valid, False otherwise.
    '''
     # 1. 读 metadata.json，检查 schema_version / store / stage
    # 2. 检查所有产物文件是否存在
    # 3. 加载所有 pkl，检查：
    #    - index 唯一
    #    - 五表 index 完全相同（行对齐！）
    #    - row_count 与 metadata 记录一致
    #    - index_signature 哈希一致
    # 4. 如果 stage="complete"，还检查 d>1941 的 sales 全为 NaN
    metadata = read_metadata(store)
    required = FILES if expected_stage == "complete" else BASE_FILES
    if not metadata or metadata.get("schema_version") != SCHEMA_VERSION:
        return False
    if metadata.get("store") != store or metadata.get("stage") not in ({"complete"} if expected_stage == "complete" else {"base", "complete"}):
        return False
    if category_hash is not None and metadata.get("category_signature") != category_hash:
        return False
    store_dir = PROCESSED_DIR / store
    if any(not (store_dir / name).is_file() for name in required):
        return False
    frames: list[pd.DataFrame] = []
    try:
        frames = [pd.read_pickle(store_dir / name) for name in required]
        reference = frames[0].index
        if not reference.is_unique or any(not frame.index.equals(reference) for frame in frames[1:]):
            return False
        if int(metadata.get("row_count", -1)) != len(reference):
            return False
        if metadata.get("index_signature") != index_signature(reference):
            return False
        if expected_stage == "complete":
            future = frames[0]["d"] > END_TRAIN
            if not frames[0].loc[future, "sales"].isna().all():
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
    finally:
        del frames
        gc.collect()


def make_vocabs(sales_meta: pd.DataFrame, calendar: pd.DataFrame) -> dict[str, list[object]]:
    vocabs: dict[str, list[object]] = {}
    for column in INDEX_COLUMNS:
        vocabs[column] = sorted(sales_meta[column].dropna().astype(str).unique().tolist())
    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        vocabs[column] = sorted(calendar[column].dropna().astype(str).unique().tolist())
    for column in ["snap_CA", "snap_TX", "snap_WI"]:
        vocabs[column] = sorted(calendar[column].dropna().unique().tolist())
    return vocabs


def as_global_category(series: pd.Series, categories: list[object]) -> pd.Series:
    if categories and isinstance(categories[0], str):
        values = series.astype("string")
    else:
        values = series
    return values.astype(pd.CategoricalDtype(categories=categories))


def load_global_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[object]], dict[str, int], int, pd.DataFrame]:
      # 1. 读 calendar.csv → 添加 d_num 列（int16）
    # 2. 只读 sales 的 6 列元数据 → 构建全局类别词表 vocabs
    # 3. 以 Categorical + float32 dtype 读完整 sales（省内存的关键！）
    # 4. 记录 id → 序号映射 id_ordinals（用于算全局行号）
    # 5. 读 sell_prices → 计算每个 (store, item) 的 release 周
    # 6. calendar 的事件/SNAP列转全局 Categorical
    
    sales_path = RAW_DIR / "sales_train_evaluation.csv"
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(np.int16)
    sales_meta = pd.read_csv(sales_path, usecols=INDEX_COLUMNS, dtype="string")
    vocabs = make_vocabs(sales_meta, calendar)
    dtype: dict[str, object] = {column: pd.CategoricalDtype(vocabs[column]) for column in INDEX_COLUMNS}
    dtype.update({f"d_{day}": np.float32 for day in range(1, END_TRAIN + 1)})
    sales = pd.read_csv(sales_path, dtype=dtype)
    id_ordinals = {identifier: ordinal for ordinal, identifier in enumerate(sales_meta["id"].astype(str))}
    del sales_meta
    price_dtype = {
        "store_id": pd.CategoricalDtype(vocabs["store_id"]),
        "item_id": pd.CategoricalDtype(vocabs["item_id"]),
        "wm_yr_wk": np.int16,
        "sell_price": np.float32,
    }
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv", dtype=price_dtype)
    release = prices.groupby(["store_id", "item_id"], observed=True)["wm_yr_wk"].min()
    release_min = int(release.min())
    release_df = release.rename("release").reset_index()
    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]:
        calendar[column] = as_global_category(calendar[column], vocabs[column])
    log_event("global_inputs_loaded", sales_rows=len(sales), price_rows=len(prices), peak_rss_mb=round(peak_rss_mb(), 1))
    return sales, prices, calendar, vocabs, id_ordinals, release_min, release_df


def build_base_grid(
    store: str,
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    vocabs: dict[str, list[object]],
    id_ordinals: dict[str, int],
    release_min: int,
    release_df: pd.DataFrame,
) -> pd.DataFrame:
    store_sales = sales.loc[sales["store_id"] == store].copy()
    day_columns = [f"d_{day}" for day in range(1, END_TRAIN + 1)]
    grid = store_sales.melt(id_vars=INDEX_COLUMNS, value_vars=day_columns, var_name="d", value_name="sales")
    dimensions = store_sales[INDEX_COLUMNS].drop_duplicates("id")
    future_parts = []
    for day in range(END_TRAIN + 1, TOTAL_DAYS + 1):
        future = dimensions.copy()
        future["d"] = f"d_{day}"
        future["sales"] = np.float32(np.nan)
        future_parts.append(future)
    grid = pd.concat([grid, *future_parts], ignore_index=True)
    grid["d"] = grid["d"].str.removeprefix("d_").astype(np.int16)
    ordinal = grid["id"].astype("string").map(id_ordinals).to_numpy(dtype=np.int64)
    row_id = ordinal * TOTAL_DAYS + grid["d"].to_numpy(dtype=np.int64)
    grid.index = pd.Index(row_id, name="row_id")
    grid["wm_yr_wk"] = grid["d"].map(calendar.set_index("d_num")["wm_yr_wk"]).astype(np.int16)
    grid = grid.reset_index().merge(release_df, on=["store_id", "item_id"], how="left", validate="many_to_one").set_index("row_id")
    grid = grid.loc[grid["wm_yr_wk"] >= grid["release"]].sort_index()
    grid["release"] = (grid["release"] - release_min).astype(np.int16)
    grid["sales"] = grid["sales"].astype(np.float32)
    grid.drop(columns="wm_yr_wk", inplace=True)
    for column in INDEX_COLUMNS:
        grid[column] = as_global_category(grid[column], vocabs[column])
    if not grid.index.is_unique:
        raise ValueError(f"{store} 生成了重复全局索引")
    return grid


def make_price_features(store: str, prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = prices.loc[prices["store_id"] == store].copy()
    item_group = frame.groupby(["store_id", "item_id"], observed=True)["sell_price"]
    frame["price_max"] = item_group.transform("max")
    frame["price_min"] = item_group.transform("min")
    frame["price_std"] = item_group.transform("std")
    frame["price_mean"] = item_group.transform("mean")
    frame["price_norm"] = frame["sell_price"] / frame["price_max"]
    frame["price_nunique"] = item_group.transform("nunique")
    frame["item_nunique"] = frame.groupby(["store_id", "sell_price"], observed=True)["item_id"].transform("nunique")
    calendar_prices = calendar[["wm_yr_wk", "month", "year"]].drop_duplicates("wm_yr_wk")
    frame = frame.merge(calendar_prices, on="wm_yr_wk", how="left", validate="many_to_one")
    frame["price_momentum"] = frame["sell_price"] / frame.groupby(["store_id", "item_id"], observed=True)["sell_price"].shift(1)
    frame["price_momentum_m"] = frame["sell_price"] / frame.groupby(["store_id", "item_id", "month"], observed=True)["sell_price"].transform("mean")
    frame["price_momentum_y"] = frame["sell_price"] / frame.groupby(["store_id", "item_id", "year"], observed=True)["sell_price"].transform("mean")
    feature_columns = [
        "sell_price", "price_max", "price_min", "price_std", "price_mean", "price_norm",
        "price_nunique", "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y",
    ]
    frame[feature_columns] = frame[feature_columns].astype(np.float32)
    return frame[["store_id", "item_id", "wm_yr_wk", *feature_columns]]


def build_price_table(grid: pd.DataFrame, price_features: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["id", "d", "store_id", "item_id"]].copy()
    keys["row_id"] = keys.index
    keys["wm_yr_wk"] = keys["d"].map(calendar.set_index("d_num")["wm_yr_wk"]).astype(np.int16)
    merged = keys.merge(price_features, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    merged.set_index("row_id", inplace=True)
    merged.index.name = "row_id"
    feature_columns = [column for column in price_features.columns if column not in {"store_id", "item_id", "wm_yr_wk"}]
    return merged.loc[grid.index, ["id", "d", *feature_columns]]


def build_calendar_table(grid: pd.DataFrame, calendar: pd.DataFrame, vocabs: dict[str, list[object]]) -> pd.DataFrame:
    columns = ["date", "d_num", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
    table = grid[["id", "d"]].copy()
    table["row_id"] = table.index
    table = table.reset_index(drop=True).merge(calendar[columns], left_on="d", right_on="d_num", how="left", validate="many_to_one")
    table.set_index("row_id", inplace=True)
    dates = pd.to_datetime(table.pop("date"))
    table.drop(columns="d_num", inplace=True)
    table["tm_d"] = dates.dt.day.astype(np.int8)
    table["tm_w"] = dates.dt.isocalendar().week.astype(np.int16)
    table["tm_m"] = dates.dt.month.astype(np.int8)
    table["tm_y"] = (dates.dt.year - dates.dt.year.min()).astype(np.int8)
    table["tm_wm"] = ((dates.dt.day - 1) // 7 + 1).astype(np.int8)
    table["tm_dw"] = dates.dt.dayofweek.astype(np.int8)
    table["tm_w_end"] = (table["tm_dw"] >= 5).astype(np.int8)
    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]:
        table[column] = as_global_category(table[column], vocabs[column])
    return table.loc[grid.index]


def rolling_feature(values: pd.Series, ids: pd.Series, shift: int, window: int, statistic: str = "mean") -> pd.Series:
    shifted = values.groupby(ids, observed=True).shift(shift)
    rolling = shifted.groupby(ids, observed=True).rolling(window)
    result = rolling.mean() if statistic == "mean" else rolling.std()
    return result.reset_index(level=0, drop=True).reindex(values.index).astype(np.float32)


def build_lag_table(grid: pd.DataFrame) -> pd.DataFrame:
    table = grid[["id", "d", "sales"]].copy()
    grouped = grid.groupby("id", observed=True)["sales"]
    for lag in range(28, 43):
        table[f"sales_lag_{lag}"] = grouped.shift(lag).astype(np.float32)
    for window in [7, 14, 30, 60, 180]:
        table[f"rolling_mean_{window}"] = rolling_feature(grid["sales"], grid["id"], 28, window, "mean")
        table[f"rolling_std_{window}"] = rolling_feature(grid["sales"], grid["id"], 28, window, "std")
    for shift in [1, 7, 14]:
        for window in [7, 14, 30, 60]:
            table[f"rolling_mean_tmp_{shift}_{window}"] = rolling_feature(grid["sales"], grid["id"], shift, window, "mean")
    return table


def write_base_outputs(store: str, grid: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame, vocabs: dict[str, list[object]], category_hash: str) -> None:
    store_dir = PROCESSED_DIR / store
    price_table = build_price_table(grid, make_price_features(store, prices, calendar), calendar)
    calendar_table = build_calendar_table(grid, calendar, vocabs)
    lag_table = build_lag_table(grid)
    frames = {
        "grid_part_1.pkl": grid,
        "grid_part_2.pkl": price_table,
        "grid_part_3.pkl": calendar_table,
        "lags_df_28.pkl": lag_table,
    }
    reference = grid.index
    if any(not frame.index.equals(reference) for frame in frames.values()):
        raise ValueError(f"{store} 基础特征索引不一致")
    for name, frame in frames.items():
        atomic_pickle(frame, store_dir / name)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "base",
        "store": store,
        "row_count": len(reference),
        "index_signature": index_signature(reference),
        "category_signature": category_hash,
        "columns": {name: list(frame.columns) for name, frame in frames.items()},
    }
    atomic_json(payload, metadata_path(store))
    total_bytes = sum((store_dir / name).stat().st_size for name in BASE_FILES)
    log_event("store_base_written", store=store, rows=len(grid), dataframe_mb=round(dataframe_mb(grid), 1), output_mb=round(total_bytes / 1024**2, 1), peak_rss_mb=round(peak_rss_mb(), 1))
    del price_table, calendar_table, lag_table, frames
    gc.collect()


def aggregate_encoding_stats(grids: Iterable[pd.DataFrame]) -> dict[tuple[str, ...], pd.DataFrame]:
    contributions: dict[tuple[str, ...], list[pd.DataFrame]] = {tuple(group): [] for group in ENCODING_GROUPS}
    for grid in grids:
        work = grid.loc[grid["d"] <= ENCODING_END, [*INDEX_COLUMNS[1:], "sales"]].copy()
        work["_sales_sq"] = work["sales"].astype(np.float64).pow(2)
        for columns in ENCODING_GROUPS:
            grouped = work.groupby(columns, observed=True, dropna=False).agg(
                count=("sales", "count"), total=("sales", "sum"), sumsq=("_sales_sq", "sum")
            ).reset_index()
            contributions[tuple(columns)].append(grouped)
        del work
        gc.collect()
    result: dict[tuple[str, ...], pd.DataFrame] = {}
    for columns in ENCODING_GROUPS:
        key = tuple(columns)
        combined = pd.concat(contributions[key], ignore_index=True)
        stats = combined.groupby(columns, observed=True, dropna=False)[["count", "total", "sumsq"]].sum().reset_index()
        stats["mean"] = stats["total"] / stats["count"]
        numerator = stats["sumsq"] - stats["total"].pow(2) / stats["count"]
        variance = numerator.clip(lower=0) / (stats["count"] - 1)
        stats["std"] = np.sqrt(variance).where(stats["count"] >= 2, np.nan)
        result[key] = stats[[*columns, "count", "mean", "std"]]
    return result


def map_stat(grid: pd.DataFrame, stats: pd.DataFrame, columns: list[str], value: str) -> np.ndarray:
    if len(columns) == 1:
        mapping = stats.set_index(columns[0])[value]
        return grid[columns[0]].map(mapping).to_numpy(dtype=np.float32, na_value=np.nan)
    mapping = stats.set_index(columns)[value]
    keys = pd.MultiIndex.from_frame(grid[columns])
    return mapping.reindex(keys).to_numpy(dtype=np.float32, na_value=np.nan)


def build_encoding_table(grid: pd.DataFrame, stats: dict[tuple[str, ...], pd.DataFrame]) -> pd.DataFrame:
    encoded = grid[["id", "d"]].copy()
    for columns in ENCODING_GROUPS:
        name = "_".join(columns)
        group_stats = stats[tuple(columns)]
        encoded[f"enc_{name}_mean"] = map_stat(grid, group_stats, columns, "mean")
        encoded[f"enc_{name}_std"] = map_stat(grid, group_stats, columns, "std")
    return encoded


def mark_complete(store: str, category_hash: str) -> None:
    store_dir = PROCESSED_DIR / store
    frames = [pd.read_pickle(store_dir / name) for name in FILES]
    reference = frames[0].index
    if not reference.is_unique or any(not frame.index.equals(reference) for frame in frames[1:]):
        raise ValueError(f"{store} 五表索引校验失败")
    if not frames[0].loc[frames[0]["d"] > END_TRAIN, "sales"].isna().all():
        raise ValueError(f"{store} 预测期 sales 未完全遮蔽")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "complete",
        "store": store,
        "row_count": len(reference),
        "index_signature": index_signature(reference),
        "category_signature": category_hash,
        "columns": {name: list(frame.columns) for name, frame in zip(FILES, frames)},
        "sizes": {name: (store_dir / name).stat().st_size for name in FILES},
    }
    atomic_json(payload, metadata_path(store))
    output_bytes = sum(payload["sizes"].values())
    log_event("store_complete", store=store, rows=len(reference), output_mb=round(output_bytes / 1024**2, 1), peak_rss_mb=round(peak_rss_mb(), 1))
    del frames
    gc.collect()


def main() -> None:
    global LOGGER
    LOGGER = logging.getLogger("m5_preprocessing")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    selected = parse_stores()
    if not selected:
        raise ValueError("M5_STORES 未选择任何门店")
    initially_complete = {store for store in selected if validate_store_files(store, "complete")}
    for store in sorted(initially_complete):
        log_event("store_skip_complete", store=store)
    pending = [store for store in selected if store not in initially_complete]
    if not pending:
        return

    sales, prices, calendar, vocabs, id_ordinals, release_min, release_df = load_global_inputs()
    category_hash = category_signature(vocabs)
    pending = [store for store in pending if not validate_store_files(store, "complete", category_hash)]

    for store in pending:
        if validate_store_files(store, "base", category_hash):
            log_event("store_base_resume", store=store)
            continue
        started = time.time()
        grid = build_base_grid(store, sales, calendar, vocabs, id_ordinals, release_min, release_df)
        write_base_outputs(store, grid, prices, calendar, vocabs, category_hash)
        log_event("store_base_elapsed", store=store, seconds=round(time.time() - started, 1))
        del grid
        gc.collect()

    def iter_global_grids() -> Iterable[pd.DataFrame]:
        for store_name in ALL_STORES:
            path = PROCESSED_DIR / store_name / "grid_part_1.pkl"
            if path.exists() and validate_store_files(store_name, "base", category_hash):
                grid_frame = pd.read_pickle(path)
            else:
                grid_frame = build_base_grid(store_name, sales, calendar, vocabs, id_ordinals, release_min, release_df)
            yield grid_frame
            del grid_frame
            gc.collect()

    stats = aggregate_encoding_stats(iter_global_grids())
    log_event("global_encoding_stats_ready", groups=len(stats), peak_rss_mb=round(peak_rss_mb(), 1))
    for store in pending:
        grid = pd.read_pickle(PROCESSED_DIR / store / "grid_part_1.pkl")
        encoded = build_encoding_table(grid, stats)
        if not encoded.index.equals(grid.index):
            raise ValueError(f"{store} 目标编码索引不一致")
        atomic_pickle(encoded, PROCESSED_DIR / store / "mean_encoding_df.pkl")
        del encoded, grid
        gc.collect()
        mark_complete(store, category_hash)

    log_event("preprocessing_finished", stores=selected, peak_rss_mb=round(peak_rss_mb(), 1))


if __name__ == "__main__":
    main()
