"""
Before Signal generation, with the help of preliminaries file, please download symbol_token_df.csv(necessary) and stocks and nifty data.

Signal Generation Pipeline — Nifty 500
Assumes all stock CSVs and NIFTY50.csv are already downloaded.

Output columns (master_signals_26.csv):
    date, ticker, stock_code, open, high, low, close, volume,
    index_close, signal, rv_10, bull_ratio, return, vol_44, roc_score,
    roc_rank, bull_rank, composite_score, final_rank,
    nifty_sma_50, nifty_sma_100, nifty_sma_200,
    nifty_ema_50, nifty_ema_100, nifty_ema_200
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import glob
import warnings
import time

import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL CONFIG  ← all tuneable parameters live here
# ============================================================

# ── Paths ────────────────────────────────────────────────────
DATA_DIR          = "data_2016_2026/nifty500"
NIFTY_CSV         = "NIFTY50.csv"
SYMBOL_TOKEN_CSV  = "symbol_token_df.csv"
MASTER_CSV        = "master_signals_final.csv"

# ── Date range for signal generation ─────────────────────────
SIGNAL_START      = pd.Timestamp("2026-03-01")
SIGNAL_END        = pd.Timestamp("2026-03-05")

# ── SMA windows (applied to each stock) ──────────────────────
SMA_WINDOWS       = [22, 66, 132]

# ── ROC periods (trading days) ────────────────────────────────
PERIODS_2M        = 44
PERIODS_3M        = 66
PERIODS_6M        = 132
PERIODS_9M        = 198

# ── Weighted momentum score weights ──────────────────────────
W1                = 0.40   # 0  → -3 months
W2                = 0.35   # -3 → -6 months
W3                = 0.25   # -6 → -9 months

# ── ARIMAX / rolling-window params ───────────────────────────
TRAIN_WINDOW      = 252    # rows of history fed into ARIMAX
RV_WINDOW         = 10     # days for realised-volatility calc
LOOKBACK_DAYS     = 22     # window for bull-ratio & signal fill
VOL_WINDOW        = 44     # days for annualised rolling vol

# ── Portfolio / ranking params ────────────────────────────────
TOP_N             = 30     # stocks per day after filters

# ── Nifty 50 SMA / EMA windows ───────────────────────────────
NIFTY_SMA_WINDOWS = [50, 100, 200]
NIFTY_EMA_WINDOWS = [50, 100, 200]

# ── Misc ──────────────────────────────────────────────────────
API_SLEEP         = 0.3    # seconds between API calls (if used)


# ============================================================
# STEP 1 — Compute SMAs & ROC for every stock CSV
# ============================================================

def compute_indicators(data_dir: str = DATA_DIR) -> None:
    """
    For each stock CSV in data_dir:
      • adds sma_22, sma_66, sma_132
      • adds roc_0_3m, roc_3m_6m, roc_6m_9m → roc_score
      • adds roc_0_2m, roc_2m_6m
    Overwrites files in-place.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"\n[Step 1] Found {len(csv_files)} CSV files → computing indicators")
    skipped = []

    for fpath in tqdm(csv_files, desc="Indicators", unit="file"):
        try:
            df = pd.read_csv(fpath)
            if "close" not in df.columns:
                skipped.append(fpath)
                continue

            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            # SMAs
            for w in SMA_WINDOWS:
                df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=w).mean().round(4)
            df["RSI_14"] = ta.rsi(df["close"], length=14).round(4)
            # ROC segments
            df["roc_0_3m"]  = df["close"].pct_change(periods=PERIODS_3M).mul(100).round(4)
            df["roc_3m_6m"] = df["close"].shift(PERIODS_3M).pct_change(periods=PERIODS_3M).mul(100).round(4)
            df["roc_6m_9m"] = df["close"].shift(PERIODS_6M).pct_change(periods=PERIODS_3M).mul(100).round(4)

            # Weighted momentum
            df["roc_score"] = (
                W1 * df["roc_0_3m"] +
                W2 * df["roc_3m_6m"] +
                W3 * df["roc_6m_9m"]
            ).round(4)

            # Additional ROC columns
            df["roc_0_2m"]  = df["close"].pct_change(periods=PERIODS_2M).mul(100).round(4)
            df["roc_2m_6m"] = df["close"].shift(PERIODS_2M).pct_change(periods=PERIODS_6M - PERIODS_2M).mul(100).round(4)

            df.to_csv(fpath, index=False)

        except Exception as e:
            tqdm.write(f"  ❌ {os.path.basename(fpath)}: {e}")
            skipped.append(fpath)

    print(f"  ✅ Done. Processed {len(csv_files) - len(skipped)} | Skipped {len(skipped)}")


# ============================================================
# STEP 2 — Nifty 50 SMA + EMA computation
# ============================================================

def compute_nifty_indicators(nifty_csv: str = NIFTY_CSV) -> pd.DataFrame:
    """
    Reads NIFTY50.csv, computes SMAs and EMAs, returns enriched DataFrame.
    Also saves NIFTY50_sma.csv for reference.
    """
    print("\n[Step 2] Computing Nifty 50 SMAs & EMAs")
    df = pd.read_csv(nifty_csv)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    for w in NIFTY_SMA_WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean().round(2)

    for w in NIFTY_EMA_WINDOWS:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean().round(2)

    out_cols = (
        ["date", "open", "high", "low", "close", "volume"]
        + [f"sma_{w}" for w in NIFTY_SMA_WINDOWS]
        + [f"ema_{w}" for w in NIFTY_EMA_WINDOWS]
    )
    df[out_cols].to_csv("NIFTY50_sma.csv", index=False)
    print("  ✅ NIFTY50_sma.csv saved")
    return df


# ============================================================
# HELPERS — used inside Step 3 (signal generation)
# ============================================================

def _parse_datetime_col(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format="mixed", errors="coerce")
    return df.dropna(subset=[col])


def arimax_signal(history: pd.DataFrame):
    """
    Fits an ARIMAX(0,0,1) on log-returns with RV & volume_diff as exogenous.
    Returns (signal: 0|1, realised_volatility: float) or (None, None).
    """
    df = history.copy()
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
    df["RV"]          = df["log_return"].rolling(RV_WINDOW).std() * np.sqrt(252)
    df["volume_diff"] = df["volume"].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 80:
        return None, None

    y = df["log_return"]
    X = df[["RV", "volume_diff"]]

    try:
        model = SARIMAX(
            y, exog=X, order=(0, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res       = model.fit(disp=False)
        last_exog = X.iloc[-1].values.reshape(1, -1)
        fc        = res.get_forecast(steps=1, exog=last_exog)
        next_ret  = float(fc.predicted_mean.iloc[0])
        signal    = 1 if next_ret > 0 else 0
        rv        = float(df["RV"].iloc[-1])
        return signal, rv
    except Exception:
        return None, None


def load_stock_csv(fpath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    df = _parse_datetime_col(df)
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("signal", "rv"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def save_stock_csv(fpath: str, df: pd.DataFrame) -> None:
    df.to_csv(fpath, index=False)


def load_nifty_close(date: pd.Timestamp, nifty_csv: str = NIFTY_CSV) -> float | None:
    if not os.path.exists(nifty_csv):
        return None
    try:
        nf = pd.read_csv(nifty_csv, usecols=["datetime", "close"])
        nf = _parse_datetime_col(nf)
        row = nf[nf["datetime"] == date]
        return float(row["close"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def get_top_n_for_date(all_fpaths: list, date: pd.Timestamp, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Filters stocks passing SMA trend + momentum filters, returns top-N by roc_score.
    """
    rows = []
    read_cols = ["datetime", "close", "sma_22", "sma_66", "sma_132",
                 "roc_score", "roc_0_2m", "roc_2m_6m"]

    for fpath in all_fpaths:
        ticker = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = pd.read_csv(fpath, usecols=read_cols)
        except Exception:
            continue

        df = _parse_datetime_col(df)
        row = df[df["datetime"] == date]
        if row.empty:
            continue
        r = row.iloc[0]

        # Filter 1: SMA trend alignment
        if any(pd.isna(r[c]) for c in ("sma_22", "sma_66", "sma_132", "roc_score")):
            continue
        if not (r["sma_22"] > r["sma_66"] > r["sma_132"]):
            continue

        # Filter 2: recent momentum stronger than older
        if pd.isna(r["roc_0_2m"]) or pd.isna(r["roc_2m_6m"]):
            continue
        if not (r["roc_0_2m"] > r["roc_2m_6m"]):
            continue

        rows.append({
            "ticker":    ticker,
            "fpath":     fpath,
            "close":     r["close"],
            "roc_score": r["roc_score"],
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("roc_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def fill_missing_signals(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Runs ARIMAX on any rows within the lookback window that are missing a signal."""
    window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS + 10)
    mask         = (df["datetime"] >= window_start) & (df["datetime"] <= as_of_date)
    missing_idx  = df[mask & df["signal"].isna()].index

    for idx in missing_idx:
        if idx < TRAIN_WINDOW:
            continue
        history = df.iloc[idx - TRAIN_WINDOW: idx][["close", "volume"]].copy()
        history.index = df.iloc[idx - TRAIN_WINDOW: idx]["datetime"]
        signal, rv = arimax_signal(history)
        df.at[idx, "signal"] = signal
        df.at[idx, "rv"]     = rv

    return df


def compute_bull_ratio(master_df: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Adds bull_ratio column: ratio of bullish to bearish signals over LOOKBACK_DAYS."""
    master_df["bull_ratio"] = np.nan

    for idx, row in tqdm(master_df.iterrows(), total=len(master_df), desc="Bull Ratio"):
        ticker = row["ticker"]
        date   = row["date"]
        fpath  = os.path.join(data_dir, f"{ticker}.csv")

        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=["datetime", "signal"])
        except (pd.errors.EmptyDataError, ValueError):
            continue

        df = _parse_datetime_col(df)
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)

        date_idx = df[df["datetime"] == date].index
        if len(date_idx) == 0:
            continue

        end_idx   = date_idx[0]
        start_idx = max(0, end_idx - LOOKBACK_DAYS + 1)
        window    = df.loc[start_idx:end_idx, "signal"].dropna()

        bulls = int((window == 1).sum())
        bears = int((window == 0).sum())

        if bulls == 0 and bears == 0:
            master_df.at[idx, "bull_ratio"] = np.nan
        elif bears == 0:
            master_df.at[idx, "bull_ratio"] = float(bulls)
        else:
            master_df.at[idx, "bull_ratio"] = round(bulls / bears, 4)

    return master_df


def compute_rankings(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-date percentile ranks:
      roc_rank, bull_rank → composite_score → final_rank
    """
    master_df["roc_rank"]        = np.nan
    master_df["bull_rank"]       = np.nan
    master_df["composite_score"] = np.nan
    master_df["final_rank"]      = np.nan

    for date, grp in master_df.groupby("date"):
        idx = grp.index
        n   = len(grp)
        if n == 0:
            continue

        roc_ranks  = grp["roc_score"].rank(pct=True).round(4)
        bull_ranks = grp["bull_ratio"].rank(pct=True).round(4)
        composite  = ((roc_ranks + bull_ranks) / 2).round(4)
        final_rank = composite.rank(ascending=False, method="min").astype(int)

        master_df.loc[idx, "roc_rank"]        = roc_ranks
        master_df.loc[idx, "bull_rank"]        = bull_ranks
        master_df.loc[idx, "composite_score"]  = composite
        master_df.loc[idx, "final_rank"]       = final_rank

    return master_df


def merge_nifty_indicators(master_df: pd.DataFrame) -> pd.DataFrame:
    """Left-joins nifty SMA & EMA columns onto master_df, forward-fills gaps."""
    nifty = pd.read_csv("NIFTY50_sma.csv")
    nifty["date"] = pd.to_datetime(nifty["date"]).dt.normalize()

    rename_map = {}
    for w in NIFTY_SMA_WINDOWS:
        rename_map[f"sma_{w}"] = f"nifty_sma_{w}"
    for w in NIFTY_EMA_WINDOWS:
        rename_map[f"ema_{w}"] = f"nifty_ema_{w}"

    nifty_slim = nifty[["date"] + list(rename_map.keys())].rename(columns=rename_map)
    nifty_slim = nifty_slim.rename(columns={"date": "date_key"})

    master_df["date_key"] = pd.to_datetime(master_df["date"]).dt.normalize()
    master_df = master_df.merge(nifty_slim, on="date_key", how="left")

    for col in rename_map.values():
        master_df[col] = master_df[col].ffill()

    master_df = master_df.drop(columns=["date_key"])
    return master_df


# ============================================================
# STEP 3 — Main signal-generation loop
# ============================================================

def generate_signals(data_dir: str = DATA_DIR) -> None:
    print("\n[Step 3] Generating signals")

    # Stock file list (exclude index & temp files)
    all_fpaths = sorted([
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
        and os.path.basename(f) != "NIFTY50.csv"
    ])
    print(f"  Found {len(all_fpaths)} stock CSVs")

    # Symbol → stock_code mapping
    symbol_token_df  = pd.read_csv(SYMBOL_TOKEN_CSV)
    symbol_to_code   = dict(zip(symbol_token_df["Symbol"], symbol_token_df["Stock_Code"]))

    # Derive signal dates from first available stock file
    _tmp = pd.read_csv(all_fpaths[0], usecols=["datetime"])
    _tmp = _parse_datetime_col(_tmp)
    all_dates = sorted(_tmp[
        (_tmp["datetime"] >= SIGNAL_START) &
        (_tmp["datetime"] <= SIGNAL_END)
    ]["datetime"].unique())
    print(f"  Signal dates: {all_dates[0].date()} → {all_dates[-1].date()} ({len(all_dates)} days)\n")

    # Load or initialise master
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df["date"] = pd.to_datetime(master_df["date"])
        print(f"  Loaded {MASTER_CSV} | rows: {len(master_df)}")
    else:
        master_df = pd.DataFrame(columns=[
            "date", "ticker", "stock_code",
            "open", "high", "low", "close", "volume",
            "index_close", "signal", "rv_10",
            "bull_ratio", "return", "vol_44", "roc_score",
        ])
        print("  Initialised fresh master")

    master_rows = []

    # ── Outer loop: dates ──────────────────────────────────────
    for date in tqdm(all_dates, desc="Dates", unit="day"):
        done_tickers = set(master_df[master_df["date"] == date]["ticker"].tolist())
        if len(done_tickers) >= TOP_N:
            tqdm.write(f"  ⏭  {date.date()} already complete")
            continue

        top_df = get_top_n_for_date(all_fpaths, date, TOP_N)
        if top_df.empty:
            tqdm.write(f"  ⚠  {date.date()} | no stocks passed filters")
            continue

        nifty_close = load_nifty_close(date)
        day_results = []

        for _, row in tqdm(top_df.iterrows(), total=len(top_df),
                           desc=f"  {date.date()}", leave=False, unit="ticker"):

            ticker    = row["ticker"]
            fpath     = row["fpath"]
            roc_score = row["roc_score"]

            # Reuse if already computed
            if ticker in done_tickers:
                existing = master_df[
                    (master_df["date"] == date) &
                    (master_df["ticker"] == ticker)
                ].iloc[0]
                day_results.append(existing.to_dict())
                continue

            df = load_stock_csv(fpath)
            if df is None:
                continue

            df = fill_missing_signals(df, date)
            save_stock_csv(fpath, df)

            today_row = df[df["datetime"] == date]
            if today_row.empty:
                continue
            tr        = today_row.iloc[0]
            today_idx = today_row.index[0]

            signal = int(tr["signal"]) if not pd.isna(tr["signal"]) else None
            rv_val = float(tr["rv"])   if not pd.isna(tr["rv"])     else None

            # Log return on this date
            if today_idx >= 1:
                prev_close = df.at[today_idx - 1, "close"]
                ret = (round(float(np.log(tr["close"] / prev_close)) * 100, 4)
                       if prev_close and prev_close > 0 else None)
            else:
                ret = None

            # 44-day annualised rolling vol
            if today_idx >= VOL_WINDOW:
                log_rets = np.log(
                    df.loc[today_idx - VOL_WINDOW + 1: today_idx, "close"].values /
                    df.loc[today_idx - VOL_WINDOW    : today_idx - 1, "close"].values
                )
                vol_44 = round(float(np.std(log_rets) * np.sqrt(252)) * 100, 4)
            else:
                vol_44 = None

            day_results.append({
                "date":        date,
                "ticker":      ticker,
                "stock_code":  symbol_to_code.get(ticker, None),
                "open":        float(tr["open"])   if "open"   in tr.index else None,
                "high":        float(tr["high"])   if "high"   in tr.index else None,
                "low":         float(tr["low"])    if "low"    in tr.index else None,
                "close":       float(tr["close"]),
                "volume":      float(tr["volume"]) if "volume" in tr.index else None,
                "index_close": nifty_close,
                "signal":      signal,
                "rv_10":       rv_val,
                "return":      ret,
                "vol_44":      vol_44,
                "roc_score":   roc_score,
            })

        master_rows.extend(day_results)
        tqdm.write(f"  ✅ {date.date()} | {len(day_results)} tickers: "
                   f"{[r['ticker'] for r in day_results]}")

    # ── Merge new rows into master ──────────────────────────────
    new_df = pd.DataFrame(master_rows)
    if not new_df.empty:
        master_df = (
            pd.concat([master_df, new_df], ignore_index=True)
            .drop_duplicates(subset=["date", "ticker"])
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
        master_df["signal"] = pd.to_numeric(master_df["signal"], errors="coerce")

    # ── Bull ratio ──────────────────────────────────────────────
    master_df = compute_bull_ratio(master_df)

    # ── Drop rows without sufficient data ──────────────────────
    master_df = master_df.dropna(subset=["bull_ratio", "rv_10", "signal"])

    # ── Rankings ────────────────────────────────────────────────
    master_df = compute_rankings(master_df)

    # ── Merge Nifty indicators ──────────────────────────────────
    master_df = merge_nifty_indicators(master_df)

    # ── Final column order (matches target format) ───────────────
    final_cols = [
        "date", "ticker", "stock_code",
        "open", "high", "low", "close", "volume",
        "index_close", "signal", "rv_10",
        "bull_ratio", "return", "vol_44", "roc_score",
        "roc_rank", "bull_rank", "composite_score", "final_rank",
        "nifty_sma_50",  "nifty_sma_100",  "nifty_sma_200",
        "nifty_ema_50",  "nifty_ema_100",  "nifty_ema_200",
    ]
    master_df = (
        master_df
        .sort_values(["date", "final_rank"])
        .reset_index(drop=True)
    )[final_cols]

    master_df.to_csv(MASTER_CSV, index=False)
    print(f"\n✅ {MASTER_CSV} saved | total rows: {len(master_df)}")
    print(master_df.tail(10).to_string())


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    compute_indicators()          # Step 1: SMAs & ROC for all stocks
    compute_nifty_indicators()    # Step 2: Nifty SMAs & EMAs
    generate_signals()            # Step 3: ARIMAX → master CSV
