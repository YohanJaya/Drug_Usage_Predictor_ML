import pandas as pd
import os
import numpy as np

def load_kaggle_data(csv_path, target_column="Sales", date_column="Date", drop_closed=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[date_column])

    # Optional: remove closed stores (Open == 0) since Sales will be 0 or meaningless
    if drop_closed and "Open" in df.columns:
        df = df[df["Open"] != 0].copy()

    # Basic date feature engineering
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_year"] = df[date_column].dt.dayofyear
    # If week number useful:
    df["week"] = df[date_column].dt.isocalendar().week.astype(int)

    # Convert common boolean-ish columns to numeric if needed
    for col in ["Promo", "SchoolHoliday", "Open"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # StateHoliday may be '0' or letters; make dummies
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)
        state_dummies = pd.get_dummies(df["StateHoliday"], prefix="StateHoliday", drop_first=True)
        df = pd.concat([df, state_dummies], axis=1)
        df.drop(columns=["StateHoliday"], inplace=True)

    # DayOfWeek is numeric already; if you want dummies:
    if "DayOfWeek" in df.columns:
        dow_dummies = pd.get_dummies(df["DayOfWeek"].astype(int), prefix="Dow", drop_first=True)
        df = pd.concat([df, dow_dummies], axis=1)
        df.drop(columns=["DayOfWeek"], inplace=True)

    # Drop columns we don't want as raw features
    drop_cols = [date_column]
    if target_column in df.columns:
        drop_cols.append(target_column)
    # 'Customers' can be used as feature or dropped (depends on task). Keep it by default.
    # 'Store' might be categorical; you can encode it or drop it
    if "Store" in df.columns:
        # small example: treat Store as categorical with dummies if small number of stores
        try:
            store_dummies = pd.get_dummies(df["Store"].astype(str), prefix="Store", drop_first=True)
            df = pd.concat([df, store_dummies], axis=1)
        except Exception:
            pass
        df.drop(columns=["Store"], inplace=True)

    # Final features
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Remove any non-numeric columns if present
    feature_df = feature_df.select_dtypes(include=[np.number]).fillna(0)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    X = feature_df.values
    y = df[target_column].values
    feature_names = feature_df.columns.tolist()
    return X, y, feature_names