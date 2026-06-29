"""
Nifty 500 Data Download Pipeline
==================================
1. Resolve Symbol -> (Token, Stock_Code) via Breeze, save symbol_token_df.csv
2. Download daily OHLCV history for every Nifty 500 stock
3. Download daily OHLCV history for the NIFTY50 index itself

Run order matters: Step 1 must succeed before Step 2 (needs stock_code).
Step 3 is independent of Steps 1-2 and can run any time.
"""

import os
import time
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from breeze_connect import BreezeConnect


BREEZE_API_KEY       = "=qw3v81645C94339h387K4461_520l05"
BREEZE_API_SECRET    = "1h87H%27q23626t448M55J5605P532y5"
BREEZE_SESSION_TOKEN = "56119530"

# ── Paths ────────────────────────────────────────────────────
DATA_DIR          = "data_2026/nifty500"     # single source of truth for ALL output
NIFTY_LIST_CSV    = "ind_nifty50list.csv"   # input: official Nifty 500 constituent list
SYMBOL_TOKEN_CSV  = "symbol_token_df.csv"    # output: Symbol -> Token/Stock_Code map
NIFTY_CSV         = os.path.join(DATA_DIR, "NIFTY50.csv")

# ── Date range / API params ──────────────────────────────────
INTERVAL    = "1day"
EXCHANGE    = "NSE"
PRODUCT     = "cash"
START_DATE  = pd.Timestamp("2026-01-01")
END_DATE    = pd.Timestamp("2026-01-31")
CHUNK_DAYS  = 365 * 2     # Breeze's per-request history limit
API_SLEEP   = 0.6         # seconds between calls, to avoid rate limiting

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# BREEZE SESSION
# ============================================================

def init_breeze() -> BreezeConnect:
    """Creates and authenticates a BreezeConnect session."""
    try:
        breeze = BreezeConnect(api_key=BREEZE_API_KEY)
        breeze.generate_session(
            api_secret=BREEZE_API_SECRET,
            session_token=BREEZE_SESSION_TOKEN,
        )
        print("✅ BreezeConnect initialized successfully")
        return breeze
    except Exception as e:
        print(f"❌ Failed to initialize BreezeConnect: {e}")
        raise SystemExit(1)


# ============================================================
# STEP 1 — Symbol → Token / Stock_Code mapping
# ============================================================

def fetch_symbol_token_code(breeze: BreezeConnect, symbol: str):
    """Looks up the Breeze token + stock_code for one symbol. Prefers level1 token."""
    try:
        resp = breeze.get_names(exchange_code="NSE", stock_code=symbol)
        if not resp:
            return None, None, resp

        token      = resp.get("isec_token_level1")
        stock_code = resp.get("isec_stock_code")

        if not token:
            return None, None, resp
        if isinstance(token, str) and " " in token:
            token = token.split()[0]   # take first level1 token if multiple

        return token, stock_code, resp
    except Exception as e:
        return None, None, str(e)


def build_symbol_token_map(breeze: BreezeConnect, force: bool = False) -> pd.DataFrame:
    """
    Resolves Symbol -> (Token, Stock_Code) for every ticker in NIFTY_LIST_CSV.
    Skips the API entirely if SYMBOL_TOKEN_CSV already exists, unless force=True.
    """
    if os.path.exists(SYMBOL_TOKEN_CSV) and not force:
        print(f"[Step 1] {SYMBOL_TOKEN_CSV} already exists, skipping. (force=True to redo)")
        return pd.read_csv(SYMBOL_TOKEN_CSV)

    print("\n[Step 1] Resolving symbol → token/stock_code map")
    equity_details = pd.read_csv(NIFTY_LIST_CSV)
    symbols = equity_details["Symbol"].astype(str).str.strip().tolist()

    rows, failed = [], {}

    for sym in tqdm(symbols, desc="Symbols", unit="symbol"):
        token, stock_code, raw = fetch_symbol_token_code(breeze, sym)

        if token and stock_code:
            rows.append({"Symbol": sym, "Token": token, "Stock_Code": stock_code})
        else:
            failed[sym] = raw
            tqdm.write(f"❌ {sym} failed")

        time.sleep(0.3)   # gentler rate limit for this lighter endpoint

    symbol_token_df = pd.DataFrame(rows)
    symbol_token_df.to_csv(SYMBOL_TOKEN_CSV, index=False)

    print(f"  ✅ SUCCESS: {len(symbol_token_df)} | ❌ FAILED: {len(failed)}")
    if failed:
        print(f"  Failed symbols: {list(failed.keys())}")

    return symbol_token_df


# ============================================================
# STEP 2 — Per-stock historical data download
# ============================================================

def fetch_history_chunk(breeze: BreezeConnect, stock_code: str,
                         from_date: pd.Timestamp, to_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches one date-range chunk of daily OHLCV for a single stock_code."""
    resp = breeze.get_historical_data(
        interval=INTERVAL,
        from_date=from_date.strftime("%Y-%m-%dT09:15:00"),
        to_date=to_date.strftime("%Y-%m-%dT15:30:00"),
        stock_code=stock_code,
        exchange_code=EXCHANGE,
        product_type=PRODUCT,
    )

    if not resp or "Success" not in resp:
        return pd.DataFrame()

    df = pd.DataFrame(resp["Success"])
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def download_one_stock(breeze: BreezeConnect, symbol: str, stock_code: str) -> None:
    """Downloads full history for one symbol across all date chunks, saves to CSV."""
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if os.path.exists(file_path):
        return   # already downloaded — skip

    all_chunks = []
    cur_start  = START_DATE
    total_steps = ((END_DATE - START_DATE).days // CHUNK_DAYS) + 1

    with tqdm(total=total_steps, desc=symbol, leave=False, unit="chunk") as pbar:
        while cur_start <= END_DATE:
            cur_end = min(cur_start + timedelta(days=CHUNK_DAYS), END_DATE)
            try:
                df = fetch_history_chunk(breeze, stock_code, cur_start, cur_end)
                if not df.empty:
                    all_chunks.append(df)
            except Exception as e:
                tqdm.write(f"⚠ {symbol} {cur_start.date()} → {cur_end.date()} | {e}")

            cur_start = cur_end + timedelta(days=1)
            pbar.update(1)
            time.sleep(API_SLEEP)

    if not all_chunks:
        tqdm.write(f"❌ No data for {symbol}")
        return

    final_df = (
        pd.concat(all_chunks)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    final_df.to_csv(file_path, index=False)
    tqdm.write(f"✅ Saved {symbol} | rows: {len(final_df)}")


def download_all_stocks(breeze: BreezeConnect, symbol_token_df: pd.DataFrame) -> None:
    print("\n[Step 2] Downloading per-stock historical data")
    for _, row in tqdm(symbol_token_df.iterrows(), total=len(symbol_token_df),
                        desc="Nifty 500 download", unit="stock"):
        symbol, stock_code = row["Symbol"], row["Stock_Code"]
        if pd.isna(stock_code):
            tqdm.write(f"⚠ Missing stock_code for {symbol}")
            continue
        download_one_stock(breeze, symbol, stock_code)

    print("  🎯 ALL STOCK DOWNLOADS COMPLETE")


# ============================================================
# STEP 3 — NIFTY 50 index download
# ============================================================

def download_nifty_index(breeze: BreezeConnect) -> None:
    print("\n[Step 3] Downloading NIFTY50 index data")

    if os.path.exists(NIFTY_CSV):
        print(f"  {NIFTY_CSV} already exists, skipping.")
        return

    all_chunks = []
    cur_start  = START_DATE
    total_steps = ((END_DATE - START_DATE).days // CHUNK_DAYS) + 1

    with tqdm(total=total_steps, desc="NIFTY50 INDEX", unit="chunk") as pbar:
        while cur_start <= END_DATE:
            cur_end = min(cur_start + timedelta(days=CHUNK_DAYS), END_DATE)
            try:
                df = fetch_history_chunk(breeze, "NIFTY", cur_start, cur_end)
                if not df.empty:
                    all_chunks.append(df)
            except Exception as e:
                tqdm.write(f"⚠ {cur_start.date()} → {cur_end.date()} | {e}")

            cur_start = cur_end + timedelta(days=1)
            pbar.update(1)
            time.sleep(API_SLEEP)

    if not all_chunks:
        print("  ❌ No data fetched for NIFTY50")
        return

    final_df = (
        pd.concat(all_chunks)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    final_df.to_csv(NIFTY_CSV, index=False)
    print(f"  ✅ Saved NIFTY50 | rows: {len(final_df)}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    breeze = init_breeze()

    symbol_token_df = build_symbol_token_map(breeze)   # Step 1
    download_all_stocks(breeze, symbol_token_df)       # Step 2
    download_nifty_index(breeze)                       # Step 3
