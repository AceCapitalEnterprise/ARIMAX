import os
import glob
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from datetime import datetime, time as dt_time
from breeze_connect import BreezeConnect
import pandas_ta_classic as ta
warnings.filterwarnings("ignore")

# mohit sir
API_KEY       = "=qw3v81645C94339h387K4461_520l05"
API_SECRET    = "1h87H%27q23626t448M55J5605P532y5"
SESSION_TOKEN = "56135572"

# #DS
# API_KEY       = "3G5x135e8v07iz8873X11941JO1104D6"
# API_SECRET    = "7z47O2524285172^9562Q1812Iq96R*8"
# SESSION_TOKEN = "55889351"

# #RP Singhal
# API_KEY     = "w617H90t&3_01jb06ja6015(0nt6y65W"
# API_SECRET  = "1s3a9251f(3079273g3xf3t2zsr*h284"
# SESSION_TOKEN = "55840902"  

DATA_DIR         = "data_live/nifty500"
MASTER_CSV       = "master_signals_live.csv"
SYMBOL_TOKEN_CSV = "symbol_token_df.csv"
PORTFOLIO_CSV = "portfolio.csv"
EXIT_TRADES_CSV = "exit_trades.csv"
EQUITY_CURVE_CSV = "equity_curve.csv"
#write can generate signals



# ── GLOBAL PARAMETERS ─────────────────────────────────────────────────────────
N             = 15
RV_MULTIPLE   = 2
INITIAL_CAP   = 1_000_000.0          # ₹10 lakh
portfolio_today = None
NO_CANDIDATES_TODAY = False
SL = 15
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(breeze):
    # ═══════════════════════════════════════════════════════════════
    # PARAMETERS — modify freely
    # ═══════════════════════════════════════════════════════════════


    TODAY      = pd.Timestamp.today().normalize()
    DATA_START = TODAY - pd.DateOffset(months=15)

    INTERVAL  = "1day"
    EXCHANGE  = "NSE"
    PRODUCT   = "cash"

    SMA_WINDOWS = [22, 66, 132]
    PERIODS_2M  = 44
    PERIODS_3M  = 66
    PERIODS_6M  = 132

    W1 = 0.40   # roc_0_3m
    W2 = 0.35   # roc_3m_6m
    W3 = 0.25   # roc_6m_9m

    TRAIN_WINDOW  = 252
    RV_WINDOW     = 10
    LOOKBACK_DAYS = 22
    TOP_N         = 30
    VOL_WINDOW    = 44

    W_ROC  = 0.50
    W_BULL = 0.50
    # ═══════════════════════════════════════════════════════════════

    os.makedirs(DATA_DIR, exist_ok=True)

    symbol_token_df = pd.read_csv(SYMBOL_TOKEN_CSV)
    symbol_to_code  = dict(zip(symbol_token_df["Symbol"], symbol_token_df["Stock_Code"]))

    from_dt = DATA_START.strftime("%Y-%m-%dT09:15:00")
    to_dt   = TODAY.strftime("%Y-%m-%dT15:30:00")

    # ──────────────────────────────────────────────────────────────
    # STEP 1 — Download Nifty 500 stocks
    #          retry 2x → relogin → 1 final try
    # ──────────────────────────────────────────────────────────────
    print("\n━━━ STEP 1: Downloading stock data ━━━")
    for _, row in tqdm(symbol_token_df.iterrows(),
                       total=len(symbol_token_df), desc="Stocks", unit="stock"):
        sym  = row["Symbol"]
        code = row["Stock_Code"]
        if pd.isna(code):
            continue
        save_path = os.path.join(DATA_DIR, f"{sym}.csv")
        # if os.path.exists(save_path):
        #     continue

        downloaded = False
        for attempt in range(1, 3):
            try:
                resp = breeze.get_historical_data(
                    interval=INTERVAL, from_date=from_dt, to_date=to_dt,
                    stock_code=code, exchange_code=EXCHANGE, product_type=PRODUCT,
                )
                if resp and "Success" in resp and resp["Success"]:
                    df = pd.DataFrame(resp["Success"])
                    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
                    df = df.sort_values("datetime").reset_index(drop=True)
                    df.to_csv(save_path, index=False)
                    tqdm.write(f"  ✅ {sym} | {len(df)} rows")
                    downloaded = True
                    break
            except Exception as e:
                tqdm.write(f"  ⚠ {sym} attempt {attempt}/2 | {e}")
    # ──────────────────────────────────────────────────────────────
    # STEP 2 — Download Nifty 50 index
    #          retry 2x → relogin → 1 final try
    # ──────────────────────────────────────────────────────────────
    print("\n━━━ STEP 2: Downloading Nifty 50 index ━━━")
    nifty_path = os.path.join(DATA_DIR, "NIFTY50.csv")
    # if not os.path.exists(nifty_path):
    downloaded = False
    for attempt in range(1, 3):
        try:
            resp = breeze.get_historical_data(
                interval=INTERVAL, from_date=from_dt, to_date=to_dt,
                stock_code="NIFTY", exchange_code="NSE", product_type="cash",
            )
            if resp and "Success" in resp and resp["Success"]:
                nf = pd.DataFrame(resp["Success"])
                nf["datetime"] = pd.to_datetime(nf["datetime"]).dt.normalize()
                nf = nf.sort_values("datetime").reset_index(drop=True)
                nf.to_csv(nifty_path, index=False)
                print(f"  ✅ NIFTY50 | {len(nf)} rows")
                downloaded = True
                break
        except Exception as e:
            print(f"  ⚠ NIFTY50 attempt {attempt}/2 | {e}")

    if not downloaded:
        try:
            breeze.generate_session(api_secret=API_SECRET, session_token=SESSION_TOKEN)
            resp = breeze.get_historical_data(
                interval=INTERVAL, from_date=from_dt, to_date=to_dt,
                stock_code="NIFTY", exchange_code="NSE", product_type="cash",
            )
            if resp and "Success" in resp and resp["Success"]:
                nf = pd.DataFrame(resp["Success"])
                nf["datetime"] = pd.to_datetime(nf["datetime"]).dt.normalize()
                nf = nf.sort_values("datetime").reset_index(drop=True)
                nf.to_csv(nifty_path, index=False)
                print(f"  ✅ NIFTY50 | {len(nf)} rows (after relogin)")
            else:
                print("  ❌ NIFTY50 failed after relogin")
        except Exception as e:
            print(f"  ❌ NIFTY50 relogin failed | {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 3 — Compute SMAs + ROC on every stock CSV
    # ──────────────────────────────────────────────────────────────
    print("\n━━━ STEP 3: Computing SMAs + ROC on stock CSVs ━━━")
    all_fpaths = sorted([
        f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not os.path.basename(f).startswith("_")
        and os.path.basename(f) != "NIFTY50.csv"
    ])

    for fpath in tqdm(all_fpaths, desc="Indicators", unit="file"):
        try:
            df = pd.read_csv(fpath)
            if "close" not in df.columns:
                continue
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df = df.sort_values("datetime").reset_index(drop=True)

            for w in SMA_WINDOWS:
                df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=w).mean().round(4)

            df["roc_0_3m"]  = df["close"].pct_change(PERIODS_3M).mul(100).round(4)
            df["roc_3m_6m"] = df["close"].shift(PERIODS_3M).pct_change(PERIODS_3M).mul(100).round(4)
            df["roc_6m_9m"] = df["close"].shift(PERIODS_6M).pct_change(PERIODS_3M).mul(100).round(4)
            df["roc_score"] = (W1*df["roc_0_3m"] + W2*df["roc_3m_6m"] + W3*df["roc_6m_9m"]).round(4)
            df["roc_0_2m"]  = df["close"].pct_change(PERIODS_2M).mul(100).round(4)
            df["roc_2m_6m"] = df["close"].shift(PERIODS_2M).pct_change(PERIODS_6M - PERIODS_2M).mul(100).round(4)
            # After your existing ROC lines, add:
            df["rsi_14"] = ta.rsi(df["close"], length=14).round(4)

            for col in ["signal", "rv"]:
                if col not in df.columns:
                    df[col] = np.nan

            df.to_csv(fpath, index=False)
        except Exception as e:
            tqdm.write(f"  ⚠ {os.path.basename(fpath)}: {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 4 — Nifty50 SMAs + EMAs, derive LATEST_DATE
    # ──────────────────────────────────────────────────────────────
    print("\n━━━ STEP 4: Computing Nifty50 SMAs + EMAs ━━━")
    nifty_df = pd.read_csv(nifty_path)
    nifty_df["datetime"] = pd.to_datetime(nifty_df["datetime"]).dt.normalize()
    nifty_df = nifty_df.sort_values("datetime").reset_index(drop=True)
    nifty_df["date"] = nifty_df["datetime"]

    for w in [50, 100, 200]:
        nifty_df[f"nifty_sma_{w}"] = nifty_df["close"].rolling(w).mean().round(2)
        nifty_df[f"nifty_ema_{w}"] = nifty_df["close"].ewm(span=w, adjust=False).mean().round(2)

    nifty_slim = nifty_df[["date",
                            "nifty_sma_50", "nifty_sma_100", "nifty_sma_200",
                            "nifty_ema_50", "nifty_ema_100", "nifty_ema_200"]].copy()

    LATEST_DATE     = nifty_df["date"].max()
    _nrow           = nifty_df[nifty_df["date"] == LATEST_DATE]
    nifty_close_day = float(_nrow["close"].iloc[0]) if not _nrow.empty else None
    print(f"  📅 Latest date: {LATEST_DATE.date()}")

    # ──────────────────────────────────────────────────────────────
    # STEP 5 — ARIMAX signal (strictly same as original)
    # ──────────────────────────────────────────────────────────────
    def arimax_signal(history: pd.DataFrame):
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
            model    = SARIMAX(y, exog=X, order=(0, 0, 1),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            res      = model.fit(disp=False)
            last_exog = X.iloc[-1].values.reshape(1, -1)
            fc       = res.get_forecast(steps=1, exog=last_exog)
            next_ret = float(fc.predicted_mean.iloc[0])
            signal   = 1 if next_ret > 0 else 0
            rv       = float(df["RV"].iloc[-1])
            return signal, rv
        except Exception:
            return None, None

    def fill_missing_signals(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS + 15)
        window_mask  = (df["datetime"] >= window_start) & (df["datetime"] <= as_of_date)
        missing_idx  = df[window_mask & df["signal"].isna()].index
        if len(missing_idx) == 0:
            return df
        for idx in missing_idx:
            if idx < TRAIN_WINDOW:
                continue
            history       = df.iloc[idx - TRAIN_WINDOW: idx][["close", "volume"]].copy()
            history.index = df.iloc[idx - TRAIN_WINDOW: idx]["datetime"]
            signal, rv    = arimax_signal(history)
            df.at[idx, "signal"] = signal
            df.at[idx, "rv"]     = rv
        return df

    # ──────────────────────────────────────────────────────────────
    # STEP 6 — get_top_n for LATEST_DATE (strictly same as original)
    # ──────────────────────────────────────────────────────────────
    def get_top_n_for_date(date: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for fpath in all_fpaths:
            ticker = os.path.splitext(os.path.basename(fpath))[0]
            try:
                df = pd.read_csv(fpath, usecols=[
                    "datetime", "close",
                    "sma_22", "sma_66", "sma_132",
                    "roc_score", "roc_0_2m", "roc_2m_6m","rsi_14"
                ])
            except Exception:
                continue
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df = df.dropna(subset=["datetime"])

            row = df[df["datetime"] == date]
            if row.empty:
                continue
            r = row.iloc[0]

            if pd.isna(r["sma_22"]) or pd.isna(r["sma_66"]) or \
               pd.isna(r["sma_132"]) or pd.isna(r["roc_score"]):
                continue
            if not (r["sma_22"] > r["sma_66"] > r["sma_132"]):
                continue
            if pd.isna(r["roc_0_2m"]) or pd.isna(r["roc_2m_6m"]):
                continue
            if not (r["roc_0_2m"] > r["roc_2m_6m"]):
                continue
            if pd.isna(r["rsi_14"]) or r["rsi_14"] <= 65:        # ← add this
                continue


            rows.append({
                "ticker":    ticker,
                "fpath":     fpath,
                "close":     r["close"],
                "roc_score": r["roc_score"],
                "rsi_14": r["rsi_14"],
            })

        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                .sort_values("roc_score", ascending=False)
                .head(TOP_N)
                .reset_index(drop=True))

    print(f"\n━━━ STEP 5+6: Selecting top stocks & signals for {LATEST_DATE.date()} ━━━")
    top_df = get_top_n_for_date(LATEST_DATE)
    if top_df.empty:
        print(f"⚠ No stocks passed filters for {LATEST_DATE.date()}")
        return pd.DataFrame()

    day_results = []

    for _, row in tqdm(top_df.iterrows(), total=len(top_df),
                       desc=f"  {LATEST_DATE.date()}", leave=False, unit="ticker"):
        ticker    = row["ticker"]
        fpath     = row["fpath"]
        roc_score = row["roc_score"]

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        for col in ["signal", "rv"]:
            if col not in df.columns:
                df[col] = np.nan

        df = fill_missing_signals(df, LATEST_DATE)
        df.to_csv(fpath, index=False)

        today_row = df[df["datetime"] == LATEST_DATE]
        if today_row.empty:
            continue
        tr        = today_row.iloc[0]
        today_idx = today_row.index[0]

        signal = int(tr["signal"]) if not pd.isna(tr["signal"]) else None
        rv_val = float(tr["rv"])   if not pd.isna(tr["rv"])     else None

        ret = None
        if today_idx >= 1:
            prev_close = df.at[today_idx - 1, "close"]
            if prev_close and prev_close > 0:
                ret = round(float(np.log(tr["close"] / prev_close)) * 100, 4)

        vol_44 = None
        if today_idx >= VOL_WINDOW:
            log_rets = np.log(
                df.loc[today_idx - VOL_WINDOW + 1: today_idx, "close"].values /
                df.loc[today_idx - VOL_WINDOW    : today_idx - 1, "close"].values
            )
            vol_44 = round(float(np.std(log_rets) * np.sqrt(252)) * 100, 4)

        day_results.append({
            "date":        LATEST_DATE,
            "ticker":      ticker,
            "stock_code":  symbol_to_code.get(ticker, None),
            "open":        float(tr["open"])   if "open"   in tr.index else None,
            "high":        float(tr["high"])   if "high"   in tr.index else None,
            "low":         float(tr["low"])    if "low"    in tr.index else None,
            "close":       float(tr["close"]),
            "volume":      float(tr["volume"]) if "volume" in tr.index else None,
            "index_close": nifty_close_day,
            "roc_score":   roc_score,
            "signal":      signal,
            "rv_10":       rv_val,
            "return":      ret,
            "vol_44":      vol_44,
            "rsi_14": float(tr["rsi_14"]) if "rsi_14" in tr.index else None,
        })

    tqdm.write(f"✅ {LATEST_DATE.date()} | filtered: {len(top_df)} | "
               f"tickers: {[r['ticker'] for r in day_results]}")

    # ──────────────────────────────────────────────────────────────
    # STEP 7 — Bull ratio (strictly same as original)
    # ──────────────────────────────────────────────────────────────
    print("\n━━━ STEP 7: Computing bull ratios ━━━")
    master_df = pd.DataFrame(day_results)
    master_df["signal"]     = pd.to_numeric(master_df["signal"], errors="coerce")
    master_df["bull_ratio"] = np.nan

    for idx, row in tqdm(master_df.iterrows(), total=len(master_df), desc="Bull Ratio"):
        ticker = row["ticker"]
        date   = row["date"]
        fpath  = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=["datetime", "signal"])
        except (pd.errors.EmptyDataError, ValueError):
            tqdm.write(f"⚠ Skipping {ticker}: no signal column or empty file")
            continue

        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.dropna(subset=["datetime"])
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

    # ──────────────────────────────────────────────────────────────
    # STEP 8 — Composite ranking (strictly same as original)
    # ──────────────────────────────────────────────────────────────
    master_df = master_df[master_df["rv_10"] <= 0.7]
    master_df = master_df[master_df["rsi_14"] > 65]
    master_df = master_df.dropna(subset=["bull_ratio", "rv_10", "signal", "rsi_14"])

    master_df["roc_rank"]  = master_df["roc_score"].rank(pct=True).round(4)
    master_df["bull_rank"] = master_df["bull_ratio"].rank(pct=True).round(4)
    master_df["composite_score"] = (
        W_ROC  * master_df["roc_rank"] +
        W_BULL * master_df["bull_rank"]
    ).round(4)
    master_df["final_rank"] = master_df["composite_score"].rank(ascending=False).astype(int)

    # ──────────────────────────────────────────────────────────────
    # STEP 9 — Merge Nifty SMAs / EMAs
    # ──────────────────────────────────────────────────────────────
    master_df["date_key"]  = pd.to_datetime(master_df["date"]).dt.normalize()
    nifty_slim["date_key"] = pd.to_datetime(nifty_slim["date"]).dt.normalize()
    master_df = master_df.merge(nifty_slim.drop(columns=["date"]), on="date_key", how="left")
    for col in ["nifty_sma_50","nifty_sma_100","nifty_sma_200",
                "nifty_ema_50","nifty_ema_100","nifty_ema_200"]:
        master_df[col] = master_df[col].ffill()
    master_df = master_df.drop(columns=["date_key"])

    # ──────────────────────────────────────────────────────────────
    # STEP 10 — Final column order & save
    # ──────────────────────────────────────────────────────────────
    final_cols = [
        "date", "ticker", "stock_code",
        "open", "high", "low", "close", "volume",
        "index_close",
        "nifty_sma_50", "nifty_sma_100", "nifty_sma_200",
        "nifty_ema_50", "nifty_ema_100", "nifty_ema_200",
        "signal", "rv_10", "rsi_14", "bull_ratio", "return", "vol_44", "roc_score",
        "roc_rank", "bull_rank", "composite_score", "final_rank",
    ]
    master_df = (master_df
                 .sort_values("composite_score", ascending=False)
                 .reset_index(drop=True))[final_cols]

    master_df.to_csv(MASTER_CSV, index=False)
    print(f"\n✅ Saved {MASTER_CSV} | {len(master_df)} rows for {LATEST_DATE.date()}")
    print(master_df[["ticker","signal","bull_ratio","composite_score","final_rank"]].to_string())

    return master_df



def build_initial_portfolio(
    signals_file: str = MASTER_CSV,
    portfolio_csv: str = PORTFOLIO_CSV,
    n: int = N,
    rv_multiple: float = RV_MULTIPLE,
    initial_cap: float = INITIAL_CAP,
) -> pd.DataFrame:
    global SL
    # ── 1. Load & filter to latest date ───────────────────────────────────────
    # signals = pd.read_csv(signals_file)
    signals = generate_signals(breeze)
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()

    latest_date = signals["date"].max()
    day_signals = (
        signals[signals["date"] == latest_date]
        .copy()
        .reset_index(drop=True)
    )
    print(f"[portfolio] Signal date : {latest_date.date()}")
    print(f"[portfolio] Stocks available : {len(day_signals)}")

    # load token map
    symbol_token_df = pd.read_csv("symbol_token_df.csv")
    token_map = dict(zip(symbol_token_df["Stock_Code"], symbol_token_df["Token"]))
    # ── 2. Top-N by final_rank ────────────────────────────────────────────────
    top = (
        day_signals
        .sort_values("final_rank")
        .head(n)
        .reset_index(drop=True)
    )
    print(f"[portfolio] Top-{n} selected : {top['ticker'].tolist()}")

    # ── 3. Inverse-volatility weights (vol_44) ────────────────────────────────
    vols = top["vol_44"].values.astype(float)

    # guard: if any vol is 0 or NaN, replace with cross-sectional mean
    mean_vol = np.nanmean(vols[vols > 0]) if np.any(vols > 0) else 1.0
    vols = np.where((vols <= 0) | np.isnan(vols), mean_vol, vols)

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()           # sum to 1

    # ── 4. Allocate capital ───────────────────────────────────────────────────
    rows = []
    total_invested = 0.0
    effective_cap = initial_cap * (len(top) / n)

    for i, (_, sig) in enumerate(top.iterrows()):
        alloc      = effective_cap * weights[i]
        rows.append({
            "signal_date"           : latest_date.date(),
            "ticker"         : sig["ticker"],
            "stock_code"     : sig["stock_code"],
            "breeze_token" : token_map.get(sig["stock_code"], None),
            "capital_allocated": round(alloc, 2),
            # "sl_pct"         : round(float(sig["rv_10"]) * 10 * rv_multiple, 4),
            "sl_pct"         : SL,
            "entry_price"      : None,
            "qty"              : None,
            "cmp"              : None,
            "peak_price"       : None,
            "trailing_sl"      : None,
            "buy_date"         : None,
            "buy_time"         : None,
        })
    portfolio_df = pd.DataFrame(rows)
    print(portfolio_df.to_string(index=False))
    write_header = not os.path.exists(portfolio_csv)
    portfolio_df.to_csv(portfolio_csv, mode="a", header=write_header, index=False)
    return portfolio_df


def on_ticks(ticks):
    # print("RAW TICK:", ticks) 
    symbol = ticks['symbol']
    ltp    = ticks['last']
    latest_ticks[symbol] = {
        "ltp" : ltp,
        "time": datetime.now()
    }
latest_ticks={}

# breeze.on_ticks = on_ticks
# breeze.ws_connect()

def startup(breeze):
    global portfolio_today

    if os.path.exists(PORTFOLIO_CSV):
        print("📂 Live portfolio found")
        portfolio_today = pd.read_csv(PORTFOLIO_CSV)
    else:
        print("🆕 No portfolio found, generating initial portfolio")
        portfolio_today = build_initial_portfolio()

    # subscribe all stock codes
    for code in portfolio_today['stock_code']:
        # breeze.subscribe_feeds(stock_code=code, exchange_code="NSE")
        breeze.subscribe_feeds(
            exchange_code="NSE",
            stock_code=code,
            product_type="cash",
            get_exchange_quotes=True,
            get_market_depth=False
        )
        time.sleep(0.2)
    
    print(f"✅ Subscribed {len(portfolio_today)} stocks")

def update_cmp():
    global portfolio_today
    for i, row in portfolio_today.iterrows():
        token = row['breeze_token']
        if token in latest_ticks:
            portfolio_today.at[i, 'cmp'] = latest_ticks[token]['ltp']

def handle_entries():
    global portfolio_today
    for i, row in portfolio_today.iterrows():
        if pd.notna(row['entry_price']):
            continue
        if pd.isna(row['cmp']):
            continue
        entry = row['cmp']
        qty   = int(row['capital_allocated'] // entry)
        if qty <= 0:
            continue
        portfolio_today.at[i, 'entry_price'] = entry
        portfolio_today.at[i, 'qty']         = qty
        portfolio_today.at[i, 'peak_price']  = entry
        portfolio_today.at[i, 'trailing_sl'] = round(entry - entry * row['sl_pct'] / 100, 4)
        portfolio_today.at[i, 'buy_date']    = datetime.now().date()
        portfolio_today.at[i, 'buy_time']    = datetime.now().time()
        print(f"🟢 ENTRY {row['ticker']} @ {entry}  qty={qty}  sl={portfolio_today.at[i, 'trailing_sl']:.2f}")
    
    portfolio_today.to_csv(PORTFOLIO_CSV, index=False)

def update_trailing_sl_and_exits(breeze):
    global portfolio_today
    exit_indices = []

    for i, row in portfolio_today.iterrows():
        if pd.isna(row['entry_price']) or pd.isna(row['cmp']):
            continue

        # ── update peak & trailing sl ──────────────────────────────
        if row['cmp'] > row['peak_price']:
            portfolio_today.at[i, 'peak_price']  = row['cmp']
            portfolio_today.at[i, 'trailing_sl'] = round(
                row['cmp'] - row['entry_price'] * row['sl_pct'] / 100, 4
            )

        # ── check exit ────────────────────────────────────────────
        if row['cmp'] <= portfolio_today.at[i, 'trailing_sl']:
            exit_row = portfolio_today.loc[i].to_dict()
            exit_row.update({
                "sell_price": row['cmp'],
                "sell_date" : datetime.now().date(),
                "sell_time" : datetime.now().time(),
                "pnl"       : round((row['cmp'] - row['entry_price']) * row['qty'], 2),
            })
            pd.DataFrame([exit_row]).to_csv(
                EXIT_TRADES_CSV,
                mode='a',
                header=not os.path.exists(EXIT_TRADES_CSV),
                index=False
            )
            try:
                breeze.unsubscribe_feeds(
                    exchange_code="NSE",
                    stock_code=row['stock_code'],
                    product_type="cash",
                    get_exchange_quotes=True,
                    get_market_depth=False
                )
            except:
                pass
            print(f"🔴 EXIT {row['ticker']} @ {row['cmp']}  pnl=₹{exit_row['pnl']:,.0f}")
            exit_indices.append(i)

    if exit_indices:
        portfolio_today.drop(exit_indices, inplace=True)
        portfolio_today.reset_index(drop=True, inplace=True)
    portfolio_today.to_csv(PORTFOLIO_CSV, index=False)

def get_cash_available():
    global INITIAL_CAP
    deployed = 0.0
    if os.path.exists(PORTFOLIO_CSV):
        df = pd.read_csv(PORTFOLIO_CSV)
        if not df.empty:
            deployed = (
                df['entry_price'].fillna(0) *
                df['qty'].fillna(0)
            ).sum()

    realized = 0.0
    if os.path.exists(EXIT_TRADES_CSV):
        exits = pd.read_csv(EXIT_TRADES_CSV)
        if not exits.empty:
            realized = ((exits['sell_price'] - exits['entry_price']) * exits['qty']).sum()

    cash = INITIAL_CAP + realized - deployed
    return round(float(cash), 2)
    
def get_recently_sold_tickers():
    if not os.path.exists(EXIT_TRADES_CSV):
        return set()

    exits = pd.read_csv(EXIT_TRADES_CSV)
    if exits.empty:
        return set()

    exits['sell_date'] = pd.to_datetime(exits['sell_date']).dt.normalize()
    today              = pd.Timestamp.today().normalize()

    # ── find last actual trading day before today ──────────────────
    last_trading_day = None
    for i in range(1, 11):
        d = today - pd.Timedelta(days=i)
        try:
            resp = breeze.get_historical_data(
                interval="1day",
                from_date=(d - pd.Timedelta(days=10)).strftime("%Y-%m-%dT00:00:00"),
                to_date=d.strftime("%Y-%m-%dT23:59:59"),
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash"
            )
            data = resp.get("Success", []) if isinstance(resp, dict) else []
            if not data:
                continue
            df           = pd.DataFrame(data)
            traded_dates = pd.to_datetime(df["datetime"]).dt.normalize()
            if d in set(traded_dates):
                last_trading_day = d
                break
        except Exception:
            continue

    if last_trading_day is None:
        print("❌ Could not determine last trading day, blocking today only")
        block_days = [today]
    else:
        print(f"✅ Last trading day: {last_trading_day.date()}")
        block_days = [today, last_trading_day]

    blocked = exits[exits['sell_date'].isin(block_days)]['ticker'].unique()
    return set(blocked)

def fill_vacancies(breeze):
    global portfolio_today, NO_CANDIDATES_TODAY,N

    # ── guards ────────────────────────────────────────────────────
    if NO_CANDIDATES_TODAY:
        return

    open_positions = len(portfolio_today)
    if open_positions <= 0:
        return
    if open_positions >= N:
        return

    cash = get_cash_available()
    if cash <= 0:
        print("❌ No cash available")
        return

    vacancies = N - open_positions
    print(f"🔍 Vacancies: {vacancies}  |  Cash: ₹{cash:,.2f}")

    # ── load signals ──────────────────────────────────────────────
    signals = pd.read_csv(MASTER_CSV)
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    latest_date = signals["date"].max()
    today_df = signals[signals["date"] == latest_date].copy().reset_index(drop=True)

    if today_df.empty:
        print("❌ No signals found for today")
        NO_CANDIDATES_TODAY = True
        return

    # ── block held + recently sold ────────────────────────────────
    held          = set(portfolio_today['ticker'].tolist())
    recently_sold = get_recently_sold_tickers()
    blocked       = held.union(recently_sold)
    print(f"🚫 Blocked: {blocked}")

    candidates = today_df[~today_df['ticker'].isin(blocked)].copy()

    if candidates.empty:
        print("❌ No candidates after filtering held/sold")
        NO_CANDIDATES_TODAY = True
        return

    # ── pick top by final_rank ────────────────────────────────────
    candidates = (candidates
                  .sort_values("final_rank")
                  .head(vacancies)
                  .reset_index(drop=True))

    if candidates.empty:
        print("❌ No candidates after ranking")
        NO_CANDIDATES_TODAY = True
        return

    print(f"✅ {len(candidates)} new candidates: {candidates['ticker'].tolist()}")

    # ── inverse-vol weighting ─────────────────────────────────────
    vols     = candidates["vol_44"].values.astype(float)
    mean_vol = np.nanmean(vols[vols > 0]) if np.any(vols > 0) else 1.0
    vols     = np.where((vols <= 0) | np.isnan(vols), mean_vol, vols)
    inv_vol  = 1.0 / vols
    weights  = inv_vol / inv_vol.sum()

    # deployable = available cash scaled to how many slots we're filling
    deployable = cash * (len(candidates) / vacancies)

    # ── token map ─────────────────────────────────────────────────
    symbol_token_df = pd.read_csv("symbol_token_df.csv")
    token_map       = dict(zip(symbol_token_df["Stock_Code"], symbol_token_df["Token"]))

    # ── build new rows ────────────────────────────────────────────
    new_rows = []
    for i, (_, sig) in enumerate(candidates.iterrows()):
        alloc = deployable * weights[i]
        new_rows.append({
            "signal_date"      : latest_date.date(),
            "ticker"           : sig["ticker"],
            "stock_code"       : sig["stock_code"],
            "breeze_token"     : token_map.get(sig["stock_code"], None),
            "capital_allocated": round(alloc, 2),
            "sl_pct"           : SL,
            "entry_price"      : None,
            "qty"              : None,
            "cmp"              : None,
            "peak_price"       : None,
            "trailing_sl"      : None,
            "buy_date"         : None,
            "buy_time"         : None,
        })

    new_df = pd.DataFrame(new_rows)

    # ── append to portfolio ───────────────────────────────────────
    portfolio_today = pd.concat([portfolio_today, new_df], ignore_index=True)
    portfolio_today.to_csv(PORTFOLIO_CSV, index=False)
    print(f"📋 Portfolio updated: {len(portfolio_today)} positions")

    # ── subscribe new stocks ──────────────────────────────────────
    for _, row in new_df.iterrows():
        breeze.subscribe_feeds(
            exchange_code="NSE",
            stock_code=row['stock_code'],
            product_type="cash",
            get_exchange_quotes=True,
            get_market_depth=False
        )
        time.sleep(0.2)
        print(f"📡 Subscribed {row['ticker']}")

try:
    breeze = BreezeConnect(api_key=API_KEY)
    #rps enterprises
    breeze.generate_session(
        api_secret=API_SECRET,
        session_token=SESSION_TOKEN
    )
    print("BreezeConnect initialized successfully")
    
except Exception as e:
    # logger.error(f"Failed to initialize BreezeConnect: {str(e)}")
    print(f"Failed to initialize BreezeConnect: {str(e)}")
    exit(1)
def market_is_bullish():
    signals = pd.read_csv(MASTER_CSV)

    latest = pd.to_datetime(signals["date"]).max()
    today = signals[pd.to_datetime(signals["date"]) == latest]

    return (
        not today.empty
        and today["index_close"].iloc[0] >
            today["nifty_ema_200"].iloc[0]
    )

breeze.on_ticks = on_ticks
breeze.ws_connect()

def run_engine():
    global LAST_PORTFOLIO_PRINT, NO_CANDIDATES_TODAY
    startup(breeze)
    signals_done = False
    LAST_PRINT_MINUTE = None
    LAST_EQUITY_MINUTE = None
    while True:
        # ── generate signals after market close ───────────────────
        if datetime.now().time() >= dt_time(15, 32) and not signals_done:
            print("🧠 Generating signals for tomorrow...")
            generate_signals(breeze)
            signals_done = True
            NO_CANDIDATES_TODAY = False
            print("✅ Signals generated for tomorrow")
        if not dt_time(9, 15) <= datetime.now().time() <= dt_time(15, 30):
            time.sleep(30)
            print("🔒 Market closed, waiting...")
            continue
        # if datetime.now().time() == dt_time(10, 00):
        if market_is_bullish():# if we dont want nifty ema filter, then remove this condition
            fill_vacancies(breeze)
        # fill_vacancies(breeze)
        # ── core engine loop ──────────────────────────────────────
        update_cmp()
        handle_entries()
        update_trailing_sl_and_exits(breeze)
        

        # ── portfolio snapshot every 2 mins ──────────────────────
        now = datetime.now()
        
        if now.minute % 15 == 0:
            if LAST_PRINT_MINUTE != now.minute:
                print(f"\n📊 PORTFOLIO SNAPSHOT @ {now.strftime('%H:%M:%S')}")
                print(
                    portfolio_today[[
                        'ticker', 'cmp', 'entry_price', 'qty',
                        'peak_price', 'trailing_sl'
                    ]].to_string(index=False)
                )
                portfolio_today.to_csv(PORTFOLIO_CSV, index=False)
                LAST_PRINT_MINUTE = now.minute
        else:
            LAST_PRINT_MINUTE = None
        # time.sleep(2)
        # ── EQUITY TRACKING (ONCE PER 5 MINUTES) ──────────────────────────
        
        if LAST_EQUITY_MINUTE != now.minute:
            try:
                # ── REALIZED PnL ─────────────────────────────
                realized = 0.0
                if os.path.exists(EXIT_TRADES_CSV):
                    exits = pd.read_csv(EXIT_TRADES_CSV)
                    if not exits.empty:
                        realized = (
                            (exits['sell_price'] - exits['entry_price']) * exits['qty']
                        ).sum()
        
                # ── UNREALIZED PnL ───────────────────────────
                unrealized = 0.0
                if portfolio_today is not None and not portfolio_today.empty:
                    temp = portfolio_today.dropna(subset=['entry_price', 'cmp', 'qty'])
                    if not temp.empty:
                        unrealized = (
                            (temp['cmp'] - temp['entry_price']) * temp['qty']
                        ).sum()
        
                # ── EQUITY ───────────────────────────────────
                equity = INITIAL_CAP + realized + unrealized
        
                # ── SAVE ROW ────────────────────────────────
                row = pd.DataFrame([{
                    "timestamp": now,
                    "equity": round(equity, 2)
                }])
        
                write_header = not os.path.exists(EQUITY_CURVE_CSV)
                row.to_csv(EQUITY_CURVE_CSV, mode='a', header=write_header, index=False)
                LAST_EQUITY_MINUTE = now.minute
        
            except Exception as e:
                print(f"⚠ Equity error: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    print("🚀 Starting live trading engine...")
    run_engine()
