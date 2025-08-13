# TRXUSDT end-to-end: data collection (Binance), resampling, ARIMA & LSTM forecasting
# Requirements:
#   pip install requests pandas numpy scikit-learn pmdarima tensorflow
#
# Notes:
# - Uses Binance public REST endpoints (no API key needed).
# - Native OHLCV intervals pulled from klines; 10m and 3h are resampled; 1s/30s built from aggTrades.
# - LSTM uses window_size=8 on normalized close values.
# - Forecast horizons (1D, 1M) are converted to steps per timeframe.

import math
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras import layers, models


SYMBOL = "TRXUSDT"
BASE_URL = "https://api.binance.com"

# Desired timeframes
TIMEFRAMES = ["1S", "30S", "1T", "10T", "1H", "3H", "1D", "1W"]  # S=seconds, T=minutes, H=hours, D=days, W=weeks (pandas-style)
WINDOW_SIZE = 8  # for LSTM
TARGET_COLUMN = "close"

# Rolling ranges to materialize
RANGES = ["1D", "5D", "1M", "3M", "1Y"]

# Practical caps: building 1s/30s bars for very long ranges is massive.
# We'll cap 1S and 30S at 5D max
SUBMIN_CAP = {"1S": "5D", "30S": "5D"}



# Helpers: time & conversion
def now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def parse_range_to_delta(range_str: str) -> timedelta:
    if range_str.endswith("D"):
        return timedelta(days=int(range_str[:-1]))
    if range_str.endswith("M"):
        # approximate months as 30 days for data pull; modeling uses actual bar steps
        return timedelta(days=30 * int(range_str[:-1]))
    if range_str.endswith("Y"):
        return timedelta(days=365 * int(range_str[:-1]))
    raise ValueError(f"Unsupported range: {range_str}")

def timeframe_to_seconds(tf: str) -> int:
    tf = tf.upper()
    if tf.endswith("S"):
        return int(tf[:-1])
    if tf.endswith("T"):
        return int(tf[:-1]) * 60
    if tf.endswith("H"):
        return int(tf[:-1]) * 3600
    if tf.endswith("D"):
        return int(tf[:-1]) * 86400
    if tf.endswith("W"):
        return int(tf[:-1]) * 7 * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")

def horizon_to_steps(horizon: str, timeframe: str) -> int:
    # Convert calendar horizon to number of bars for the given timeframe
    horizon = horizon.upper()
    tf_sec = timeframe_to_seconds(timeframe)
    if horizon == "1D":
        seconds = 86400
    elif horizon == "1M":
        # approximate: 30 days
        seconds = 30 * 86400
    else:
        raise ValueError("Horizon must be '1D' or '1M'")
    return max(1, seconds // tf_sec)


# Binance REST: klines & trades
def get_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Pull OHLCV klines. interval in Binance-native strings: 1m, 1h, 1d, 1w, etc."""
    url = f"{BASE_URL}/api/v3/klines"
    rows = []
    limit = 1000
    cur = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": cur, "endTime": end_ms}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_open_time = data[-1][0]
        # Move forward; Binance klines are non-overlapping
        next_ms = last_open_time + 1
        if next_ms >= end_ms or len(data) < limit:
            break
        cur = next_ms
        time.sleep(0.1)  # be polite
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open","high","low","close","volume"]]

def get_agg_trades(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Pull aggregated trades and return a trades DataFrame (time, price, qty)."""
    url = f"{BASE_URL}/api/v3/aggTrades"
    trades = []
    # Binance aggTrades supports fromId or startTime/endTime, but limited per request.
    cur = start_ms
    while True:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        trades.extend(data)
        last_t = data[-1]["T"]
        next_ms = last_t + 1
        if next_ms >= end_ms or len(data) < 1000:
            break
        cur = next_ms
        time.sleep(0.1)
    if not trades:
        return pd.DataFrame(columns=["time","price","qty"])
    df = pd.DataFrame(trades)
    df["time"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["price"] = df["p"].astype(float)
    df["qty"] = df["q"].astype(float)
    return df[["time","price","qty"]].sort_values("time").reset_index(drop=True)

def trades_to_ohlcv(trades_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate trades to OHLCV using a pandas resample rule (e.g., '1S', '30S')."""
    if trades_df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    ts = trades_df.set_index("time")
    # price OHLC, volume as sum of qty
    ohlc = ts["price"].resample(rule).ohlc()
    vol = ts["qty"].resample(rule).sum()
    df = pd.concat([ohlc, vol], axis=1)
    df.columns = ["open","high","low","close","volume"]
    # drop empty periods
    df = df.dropna(subset=["open","high","low","close"])
    return df



# Data assembly per timeframe & range
def fetch_timeframe_range(symbol: str, timeframe: str, range_str: str) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by UTC time for (timeframe, range)."""
    timeframe = timeframe.upper()
    range_str = range_str.upper()

    # Apply caps for sub-minute data
    effective_range = range_str
    if timeframe in SUBMIN_CAP:
        # If requested longer than cap, enforce cap
        cap_delta = parse_range_to_delta(SUBMIN_CAP[timeframe])
        req_delta = parse_range_to_delta(range_str)
        if req_delta > cap_delta:
            effective_range = SUBMIN_CAP[timeframe]

    end_ms = now_utc_ms()
    start_ms = end_ms - int(parse_range_to_delta(effective_range).total_seconds() * 1000)

    # Native Binance kline coverage and resampling plan:
    # - 1T (1m), 1H, 1D, 1W -> native klines
    # - 10T (10m) from 1T resample
    # - 3H from 1H resample
    # - 1S, 30S from aggTrades -> aggregate
    if timeframe in ["1T", "1H", "1D", "1W"]:
        interval_map = {"1T": "1m", "1H": "1h", "1D": "1d", "1W": "1w"}
        df = get_klines(symbol, interval_map[timeframe], start_ms, end_ms)
        return df

    if timeframe == "10T":
        base = get_klines(symbol, "1m", start_ms, end_ms)
        if base.empty:
            return base
        return base.resample("10T").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(subset=["open","close"])

    if timeframe == "3H":
        base = get_klines(symbol, "1h", start_ms, end_ms)
        if base.empty:
            return base
        return base.resample("3H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(subset=["open","close"])

    if timeframe in ["1S", "30S"]:
        trades = get_agg_trades(symbol, start_ms, end_ms)
        rule = "1S" if timeframe == "1S" else "30S"
        return trades_to_ohlcv(trades, rule)

    raise ValueError(f"Unsupported timeframe requested: {timeframe}")

def build_all_data(symbol: str, timeframes: list, ranges: list) -> dict:
    """Return nested dict: data[timeframe][range] = OHLCV DataFrame"""
    out = {}
    for tf in timeframes:
        out[tf] = {}
        for r in ranges:
            try:
                df = fetch_timeframe_range(symbol, tf, r)
            except Exception as e:
                print(f"[WARN] Failed {tf} {r}: {e}")
                df = pd.DataFrame(columns=["open","high","low","close","volume"])
            out[tf][r] = df
    return out


# Modeling: ARIMA & LSTM
def forecast_arima(close_series: pd.Series, steps: int) -> pd.Series:
    """Auto-ARIMA on close series; returns out-of-sample forecast series length=steps."""
    # Ensure no missing values
    y = close_series.dropna().astype(float)
    if len(y) < 30:
        raise ValueError("Not enough data for ARIMA (need ~30+ points).")
    model = auto_arima(
        y,
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        max_order=10,
    )
    fc = model.predict(n_periods=steps)
    # Build a future index spaced like the input frequency (best effort)
    if y.index.inferred_freq is not None:
        idx = pd.date_range(start=y.index[-1] + pd.tseries.frequencies.to_offset(y.index.inferred_freq),
                            periods=steps, freq=y.index.inferred_freq)
    else:
        # fallback to equal spacing by seconds between last 2 points
        if len(y) >= 2:
            dt_sec = int((y.index[-1] - y.index[-2]).total_seconds())
            idx = pd.date_range(start=y.index[-1] + pd.to_timedelta(dt_sec, unit="s"),
                                periods=steps, freq=f"{dt_sec}S")
        else:
            idx = pd.RangeIndex(steps)
    return pd.Series(fc, index=idx, name="arima_forecast")

def make_supervised(values: np.ndarray, window: int) -> tuple:
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    X = np.array(X)
    y = np.array(y)
    return X, y

def forecast_lstm(close_series: pd.Series, steps: int, window: int = 8, epochs: int = 20) -> pd.Series:
    """LSTM iterative multi-step forecasting on normalized close series."""
    s = close_series.dropna().astype(float).values.reshape(-1, 1)
    if len(s) < max(40, window + 10):
        raise ValueError("Not enough data for LSTM.")

    scaler = MinMaxScaler()
    s_scaled = scaler.fit_transform(s)

    X, y = make_supervised(s_scaled.flatten(), window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build a simple LSTM
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # Iterative forecasting
    last_window = s_scaled[-window:].flatten().tolist()
    preds_scaled = []
    for _ in range(steps):
        x_in = np.array(last_window[-window:]).reshape(1, window, 1)
        pred = model.predict(x_in, verbose=0).flatten()[0]
        preds_scaled.append(pred)
        last_window.append(pred)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Build future index spaced like the original
    idx = close_series.index
    if idx.inferred_freq is not None:
        future_idx = pd.date_range(
            start=idx[-1] + pd.tseries.frequencies.to_offset(idx.inferred_freq),
            periods=steps, freq=idx.inferred_freq
        )
    else:
        if len(idx) >= 2:
            dt_sec = int((idx[-1] - idx[-2]).total_seconds())
            future_idx = pd.date_range(
                start=idx[-1] + pd.to_timedelta(dt_sec, unit="s"),
                periods=steps, freq=f"{dt_sec}S"
            )
        else:
            future_idx = pd.RangeIndex(steps)

    return pd.Series(preds, index=future_idx, name="lstm_forecast")



# Orchestration
def run_example():
    # 1) Pull data
    data = build_all_data(SYMBOL, TIMEFRAMES, RANGES)

    # 2) Choose a timeframe for modeling demo
    #    1H is a good compromise between detail
    tf_choice = "1H"
    # Choose a range providing enough history for training
    range_choice = "3M"
    df = data[tf_choice][range_choice]
    if df.empty:
        raise RuntimeError(f"No data for {tf_choice} {range_choice}")

    close = df[TARGET_COLUMN]
    print(f"Using {tf_choice} | {range_choice} | samples={len(close)} | last={close.index[-1]}")

    # 3) Forecast horizons (calendar): 1D and 1M
    for horizon in ["1D", "1M"]:
        steps = horizon_to_steps(horizon, tf_choice)
        print(f"\n=== Forecast horizon {horizon} => steps={steps} bars at {tf_choice} ===")

        # ARIMA
        try:
            arima_fc = forecast_arima(close, steps=steps)
            print(f"ARIMA: {len(arima_fc)} steps; last point: {arima_fc.iloc[-1]:.6f}")
        except Exception as e:
            print(f"ARIMA failed: {e}")
            arima_fc = None

        # LSTM
        try:
            lstm_fc = forecast_lstm(close, steps=steps, window=WINDOW_SIZE, epochs=20)
            print(f"LSTM: {len(lstm_fc)} steps; last point: {lstm_fc.iloc[-1]:.6f}")
        except Exception as e:
            print(f"LSTM failed: {e}")
            lstm_fc = None

        # Join previews
        if arima_fc is not None and lstm_fc is not None:
            preview = pd.concat([arima_fc.rename("ARIMA"), lstm_fc.rename("LSTM")], axis=1).head(10)
            print("\nPreview (first 10 rows):")
            print(preview)
        elif arima_fc is not None:
            print("\nARIMA preview (first 10 rows):")
            print(arima_fc.head(10))
        elif lstm_fc is not None:
            print("\nLSTM preview (first 10 rows):")
            print(lstm_fc.head(10))

    # 4) Access any of the prepared DataFrames
    #    e.g. 10-minute bars over 1M
    df_10m_1m = data["10T"]["1M"]
    print(f"\n10m bars (1M): {len(df_10m_1m)} rows; columns={list(df_10m_1m.columns)}")

if __name__ == "__main__":
    run_example()
