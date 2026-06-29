
#Before running this code, it is necessary to have master signals csv.


# # code for multiple backtests
# # if market above 200 ema , fix sl , else dont buy


# #DS
# API_KEY     = "3G5x135e8v07iz8873X11941JO1104D6"
# API_SECRET  = "7z47O2524285172^9562Q1812Iq96R*8"
# SESSION_TOKEN = "55330776"   # typed once at start

#RP Singhal
API_KEY     = "w617H90t&3_01jb06ja6015(0nt6y65W"
API_SECRET  = "1s3a9251f(3079273g3xf3t2zsr*h284"
SESSION_TOKEN = "56117835"   # typed once at start

# #mohit sir
# API_KEY     = "=qw3v81645C94339h387K4461_520l05"
# API_SECRET  = "1h87H%27q23626t448M55J5605P532y5"
# SESSION_TOKEN = "55725098"   # typed once at start



# #RPS Enterprises
# API_KEY     = "n592008%0y805x37369Gj%h83735o83T"
# API_SECRET  = "696r47U1T5wWz6448w1gd7819iRZm777"
# SESSION_TOKEN = "55330899"   # typed once at start




import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import requests
import os
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SIGNALS_FILE = "master_df_2020_2026.csv"
INITIAL_CAP  = 1_000_000.0
MAX_POSITIONS = 15
BUY_TIME     = pd.to_datetime("10:00").time()
START_DATE   = pd.Timestamp("2021-01-01")
END_DATE     = pd.Timestamp("2026-03-18")
FETCH_DAYS   = 70
API_SLEEP    = 0.1
RV_MULTIPLES  = [15] #Fix SL
# breeze already logged in globally

def init_breeze():
    global breeze
    from breeze_connect import BreezeConnect
    breeze = BreezeConnect(api_key=API_KEY)
    breeze.generate_session(api_secret=API_SECRET, session_token=SESSION_TOKEN)
    print("BreezeConnect session initialized")

try:
    init_breeze()
except Exception as e:
    print(f"Failed to initialize BreezeConnect: {e}")
    exit(1)



import glob

PROCESSED_DATA_FOLDER = "data_2016_2026/nifty500/"  # adjust if needed

# Cache so we don't re-read the same CSV multiple times
_indicator_cache = {}

INDICATOR_COLS = ["RSI_14"]
def get_indicators_on_date(stock_code, date):
    """
    STRICT: returns LAST candle of the SAME date (no shifting)
    """
    if stock_code not in _indicator_cache:
        pattern = os.path.join(PROCESSED_DATA_FOLDER, f"{stock_code}.csv")
        matches = glob.glob(pattern)
        if not matches:
            return {}

        df = pd.read_csv(matches[0], parse_dates=["datetime"])
        df["date_only"] = df["datetime"].dt.normalize()
        _indicator_cache[stock_code] = df

    df = _indicator_cache[stock_code]
    dt = pd.Timestamp(date).normalize()

    day_df = df[df["date_only"] == dt]

    if day_df.empty:
        return {}

    row = day_df.iloc[-1]  # ✅ SAME DAY last candle (EOD)

    return {col: row[col] for col in INDICATOR_COLS if col in row.index}
    
def passes_indicator_filter(stock_code, signal_date):
    # ind = get_indicators_on_buy_date(stock_code, signal_date)
    ind = get_indicators_on_date(stock_code, signal_date)
    if not ind:
        return False

    conditions = []

    # 1. RSI
    if "RSI_14" in ind:
        conditions.append(ind["RSI_14"] > 65)

    # # 2. CCI
    # if "CCI_20_0.015" in ind:
    #     conditions.append(ind["CCI_20_0.015"] > 50)

    # # 3. WILLR
    # if "WILLR_14" in ind:
    #     conditions.append(ind["WILLR_14"] > -40)

    # # 4. Supertrend bullish
    # if "SUPERTd_10_3.0" in ind:
    #     conditions.append(ind["SUPERTd_10_3.0"] == 1)

    # # 5. PSAR bullish
    # if "PSARl_0.02_0.2" in ind and "PSARs_0.02_0.2" in ind:
    #     conditions.append(ind["PSARl_0.02_0.2"] > 0)  # long active

    # Require at least 4 True
    return sum(conditions) ==1
# ── FETCH WITH RETRY + RELOGIN ─────────────────────────────────────────────────
def fetch_with_retry(stock_code, from_date, to_date, retries=5, wait=10):
    for attempt in range(1, retries + 1):
        try:
            df = fetch_5min(stock_code, from_date, to_date)
            return df
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                Exception) as e:
            if attempt == retries:
                print(f"  [FAILED after {retries} attempts] {stock_code}: {e}")
                print(f"  [RELOGIN] wifi may be down, attempting reconnect...")
                for relogin_attempt in range(1, 21):
                    try:
                        time.sleep(15)
                        init_breeze()   # reuses same SESSION_TOKEN, no re-typing
                        print(f"  [RELOGIN] success on attempt {relogin_attempt}, retrying fetch...")
                        return fetch_5min(stock_code, from_date, to_date)
                    except Exception as re:
                        print(f"  [RELOGIN {relogin_attempt}/20] failed: {re}")
                print(f"  [RELOGIN EXHAUSTED] giving up on {stock_code}")
                return pd.DataFrame()
            print(f"  [retry {attempt}/{retries}] {stock_code} — {e} — waiting {wait}s")
            time.sleep(wait)

            
# ── FETCH ──────────────────────────────────────────────────────────────────────
def fetch_5min(stock_code, from_date, to_date):
    from_str = pd.Timestamp(from_date).strftime("%Y-%m-%dT09:00:00")
    to_str   = pd.Timestamp(to_date).strftime("%Y-%m-%dT15:30:00")
    try:
        resp = breeze.get_historical_data(
            interval="30minute",
            from_date=from_str,
            to_date=to_str,
            stock_code=stock_code,
            exchange_code="NSE",
            product_type="cash",
        )
        if not resp or "Success" not in resp or not resp["Success"]:
            return pd.DataFrame()
        df = pd.DataFrame(resp["Success"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        print(f"  [fetch error] {stock_code}: {e}")
        return pd.DataFrame()


# ── TRACK ONE STOCK ────────────────────────────────────────────────────────────
def track_stock(stock_code, buy_date, buy_price, sl_dist,entry_df = None):
    """
    Track from buy_date to END_DATE in 15-day chunks.
    Returns (exit_price, exit_date, exit_reason, peak_price)
    """
    peak    = buy_price
    curr_sl = round(buy_price - sl_dist, 4)
    from_dt = pd.Timestamp(buy_date)
    df      = pd.DataFrame()
    eod_prices = {}

    while from_dt <= END_DATE:
        to_dt = min(from_dt + pd.Timedelta(days=FETCH_DAYS - 1), END_DATE)
        print(f"    chunk {from_dt.date()} → {to_dt.date()}")

        # ── reuse already-fetched df for first chunk instead of re-fetching ──
        if entry_df is not None and from_dt.date() == buy_date:
            df       = entry_df
            entry_df = None          # use only once, fetch normally after
            df = df[df["datetime"].dt.date <= END_DATE.date()].reset_index(drop=True)# add this
        else:
            df = fetch_with_retry(stock_code, from_dt, to_dt)
            time.sleep(API_SLEEP)
            

        if df.empty:
            from_dt = to_dt + pd.Timedelta(days=1)
            continue

        df = df[df["datetime"].dt.date <= END_DATE.date()].reset_index(drop=True)#add this

        # on buy_date: skip entry candle and anything before it
        if from_dt.date() == buy_date:
            df = df[
                (df["datetime"].dt.date > buy_date) |
                ((df["datetime"].dt.date == buy_date) &
                 (df["datetime"].dt.time > BUY_TIME))
            ].reset_index(drop=True)

        for _, row in df.iterrows():
            if row["high"] > peak:
                peak    = row["high"]
                curr_sl = round(peak - sl_dist, 4)
            if row["low"] <= curr_sl:
                return curr_sl, row["datetime"].date(), "SL_HIT", peak, eod_prices
            if row["datetime"].time() == pd.to_datetime("15:30").time():
                eod_prices[row["datetime"].date()] = row["close"]

        from_dt = to_dt + pd.Timedelta(days=1)

    last_close = float(df["close"].iloc[-1]) if not df.empty else buy_price
    return last_close, to_dt.date(), "END_OF_PERIOD", peak, eod_prices

# ── TRACK A BATCH ─────────────────────────────────────────────────────────────
def track_batch(positions):
    results = []
    for pos in positions:
        print(f"\n  tracking {pos['ticker']}...")
        exit_price, exit_date, reason, peak,eod_prices  = track_stock(
            pos["stock_code"], pos["buy_date"], pos["buy_price"], pos["sl_dist"],entry_df = pos.pop("entry_df",None),
        )
        pnl = round((exit_price - pos["buy_price"]) * pos["qty"], 2)
        print(f"    exit @ {exit_price:.2f} on {exit_date} [{reason}]  peak={peak:.2f}  PnL=₹{pnl:,.0f}")
        results.append({**pos,
            "exit_price"  : round(exit_price, 4),
            "exit_date"   : exit_date,
            "exit_reason" : reason,
            "peak_price"  : round(peak, 4),
            "pnl"         : pnl,
            "pnl_pct"     : round((exit_price - pos["buy_price"]) / pos["buy_price"] * 100, 2),
            "eod_prices" : eod_prices,
        })
    return results

def log_trade(pos):
    base = {
        "Ticker"      : pos["ticker"],
        "Stock_Code"  : pos["stock_code"],
        "Final_Rank"  : pos["final_rank"],
        "Invested"    : pos["invested"],
        "Qty"         : pos["qty"],
        "Buy_Date"    : pos["buy_date"],
        "Buy_Price"   : pos["buy_price"],
        "SL_Distance" : pos["sl_dist"],
        "Initial_SL"  : pos["initial_sl"],
        "Peak_Price"  : pos["peak_price"],
        "RV_10"       : pos["rv_10"],
        "Vol_44"      : pos["vol_44"],
        "Exit_Date"   : pos["exit_date"],
        "Exit_Price"  : pos["exit_price"],
        "Exit_Reason" : pos["exit_reason"],
        "PnL"         : pos["pnl"],
        "PnL_Pct"     : pos["pnl_pct"],
        "Regime_Mult" : pos.get("regime_mult", np.nan),
    }
    
    indicators = get_indicators_on_date(pos["ticker"], pos["signal_date"])
    base.update(indicators)
    return base
    
# ── METRICS HELPER ────────────────────────────────────────────────────────────
def compute_metrics(summary, trade_log, portfolio, equity_series):
    daily_returns = equity_series.pct_change().dropna()

    def sharpe_ratio_daily(daily_returns, risk_free_rate=0.07):
        TRADING_DAYS = 252
        rfr_per_day  = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
        excess_ret   = daily_returns - rfr_per_day
        return (excess_ret.mean() / excess_ret.std(ddof=0)) * np.sqrt(TRADING_DAYS)

    total_pnl = summary["PnL"].sum()
    sharpe    = sharpe_ratio_daily(daily_returns)
    ann_vol   = daily_returns.std(ddof=0) * np.sqrt(252) * 100
    roll_max  = equity_series.cummax()
    max_dd    = ((equity_series - roll_max) / roll_max * 100).min()

    summary["Holding_Days"] = (
        pd.to_datetime(summary["Exit_Date"]) - pd.to_datetime(summary["Buy_Date"])
    ).dt.days
    winners  = summary[summary["PnL"] > 0]
    losers   = summary[summary["PnL"] <= 0]
    win_rate = len(winners) / len(summary) * 100

    start = pd.to_datetime(summary["Buy_Date"].min())
    end   = pd.to_datetime(summary["Exit_Date"].max())
    years = (end - start).days / 365.25
    cagr  = ((INITIAL_CAP + total_pnl) / INITIAL_CAP) ** (1 / years) - 1

    return {
        "No_Trades"       : len(summary),
        "Total_PnL"       : round(total_pnl, 2),
        "Final_Value"     : round(INITIAL_CAP + total_pnl, 2),
        "Return_Pct"      : round(total_pnl / INITIAL_CAP * 100, 2),
        "CAGR_Pct"        : round(cagr * 100, 2),
        "Sharpe"          : round(sharpe, 2),
        "Ann_Vol_Pct"     : round(ann_vol, 2),
        "Max_DD_Pct"      : round(max_dd, 2),
        "Win_Rate_Pct"    : round(win_rate, 2),
        "Avg_PnL"         : round(summary["PnL"].mean(), 0),
        "Avg_PnL_Pct"     : round(summary["PnL_Pct"].mean(), 2),
        "Avg_Profit"      : round(winners["PnL"].mean(), 0) if len(winners) else 0,
        "Avg_Loss"        : round(losers["PnL"].mean(), 0)  if len(losers)  else 0,
        "Avg_Holding_Days": round(summary["Holding_Days"].mean(), 1),
        "Profit_Factor"   : round(abs(winners["PnL"].sum() / losers["PnL"].sum()), 2)
                            if len(losers) and losers["PnL"].sum() != 0 else np.nan,
    }


# ── SINGLE-MULTIPLE BACKTEST ──────────────────────────────────────────────────
def run_backtest(rv_multiple):
    # reload + filter master fresh for each run
    master = pd.read_csv(SIGNALS_FILE)
    master["date"] = pd.to_datetime(master["date"]).dt.normalize()
    master = master[(master["date"] >= START_DATE) & (master["date"] <= END_DATE)]
    master = master[master["rv_10"] <= 0.7].reset_index(drop=True)

    trading_dates = sorted(master["date"].drop_duplicates().tolist())

    def signals_on(date):
        return master[master["date"] == pd.Timestamp(date)].sort_values("final_rank")

    # ── inline buy_batch that uses this run's rv_multiple ────────────────────
    def _buy_batch(candidates, cash_available, buy_date, vacancy):
        if not candidates or vacancy <= 0:
            return [], cash_available

        deployable = cash_available * (len(candidates) / vacancy)
        vols       = np.array([float(r["vol_44"]) for r in candidates])
        vols       = np.where(vols <= 0, np.nanmean(vols), vols)
        inv_vol    = 1.0 / vols
        weights    = inv_vol / inv_vol.sum()

        positions = []
        cash_used = 0.0

        # ── regime check once per batch (all candidates share same date/index) ──
        sample_sig   = candidates[0]
        index_close  = float(sample_sig["index_close"])
        sma_100      = float(sample_sig["nifty_ema_200"])
        # regime_mult  = 1 if index_close < sma_100 else rv_multiple   # ← key line
        # print(f"  [regime] index={index_close:.0f}  sma100={sma_100:.0f}  → multiple={regime_mult}")
        if index_close < sma_100:
            print(f"  [regime] index={index_close:.0f}  sma100={sma_100:.0f}  → BELOW SMA, skipping batch")
            return [], cash_available
        regime_mult = rv_multiple
        print(f"  [regime] index={index_close:.0f}  sma100={sma_100:.0f}  → ABOVE SMA, multiple={regime_mult}")
                
        for i, sig in enumerate(candidates):
            alloc    = deployable * weights[i]
            # AFTER — fetches full first 15-day chunk
            first_chunk_end = min(
                pd.Timestamp(buy_date) + pd.Timedelta(days=FETCH_DAYS - 1), END_DATE
                    ).date()
            df_entry = fetch_with_retry(sig["stock_code"], buy_date, first_chunk_end)
            time.sleep(API_SLEEP)

            if df_entry.empty:
                continue
            candle = df_entry[df_entry["datetime"].dt.time == BUY_TIME]
            if candle.empty:
                continue

            buy_price   = float(candle.iloc[0]["open"])
            qty         = int(alloc // buy_price)
            if qty == 0:
                continue

            actual_cost = round(qty * buy_price, 2)
            # sl_dist     = round(buy_price * sig["rv_10"] * 10 * regime_mult  / 100, 4)
            sl_dist     = round(buy_price * regime_mult  / 100, 4)
            initial_sl  = round(buy_price - sl_dist, 4)
            cash_used  += actual_cost

            positions.append({
                "ticker"    : sig["ticker"],
                "stock_code": sig["stock_code"],
                "final_rank": sig["final_rank"],
                "rv_10"     : sig["rv_10"],
                "vol_44"    : sig["vol_44"],
                "buy_date"  : buy_date,
                "signal_date": sig["date"], 
                "buy_price" : buy_price,
                "qty"       : qty,
                "invested"  : actual_cost,
                "sl_dist"   : sl_dist,
                "initial_sl": initial_sl,
                "regime_mult" : regime_mult, 
                "entry_df"    : df_entry, 
            })

        return positions, cash_available - cash_used

    def _get_candidates(sig_date, in_portfolio, exclude_tickers, vacancy):
        candidates = []
        for _, sig in signals_on(sig_date).iterrows():
            if sig["ticker"] in in_portfolio or sig["ticker"] in exclude_tickers:
                continue
            # ✅ APPLY FILTER HERE
            if not passes_indicator_filter(sig["ticker"], sig_date):
                continue

            
            candidates.append(sig)
            if len(candidates) == vacancy:
                break
        return candidates

    def _try_fill_vacancies(from_signal_ts, current_cash):
        idx = trading_dates.index(from_signal_ts) if from_signal_ts in trading_dates else -1
        for sig_ts in trading_dates[idx + 1:]:
            vacancy = MAX_POSITIONS - len(portfolio)
            if vacancy <= 0 or sig_ts > END_DATE:
                break
            next_buy_idx = trading_dates.index(sig_ts) + 1
            if next_buy_idx >= len(trading_dates):
                break
            next_buy_ts = trading_dates[next_buy_idx]
            if next_buy_ts > END_DATE:
                break
            sold_this_day = {t for t, d in sold_on.items() if d == sig_ts.date()}
            candidates    = _get_candidates(sig_ts, set(portfolio.keys()), sold_this_day, vacancy)
            if not candidates:
                continue
            new_positions, current_cash = _buy_batch(candidates, current_cash, next_buy_ts.date(), vacancy)
            if new_positions:
                for pos in new_positions:
                    portfolio[pos["ticker"]] = pos
                for r in track_batch(new_positions):
                    portfolio[r["ticker"]].update(r)
        return current_cash

    # ── state ─────────────────────────────────────────────────────────────────
    cash      = INITIAL_CAP
    portfolio = {}
    sold_on   = {}
    trade_log = []

    # round 1
    first_signal_ts = next((d for d in trading_dates if d >= START_DATE), None)
    # day1_sigs       = signals_on(first_signal_ts)
    # candidates      = [row for _, row in day1_sigs.head(MAX_POSITIONS).iterrows()]
    candidates = _get_candidates(
        first_signal_ts,
        in_portfolio=set(),
        exclude_tickers=set(),
        vacancy=MAX_POSITIONS
    )
    next_buy_ts     = next((d for d in trading_dates if d > first_signal_ts), None)
    buy_date        = next_buy_ts.date()

    positions, cash = _buy_batch(candidates, cash, buy_date, MAX_POSITIONS)
    for pos in positions:
        portfolio[pos["ticker"]] = pos
    for r in track_batch(list(portfolio.values())):
        portfolio[r["ticker"]].update(r)

    cash = _try_fill_vacancies(next_buy_ts, cash)

    # rolling loop — wrap trading_dates in tqdm
    with tqdm(total=len(trading_dates), desc=f"RV×{rv_multiple}", leave=False) as pbar:
        processed = set()
        while True:
            tracked = {t: p for t, p in portfolio.items() if "exit_date" in p}
            if not tracked:
                break

            earliest_exit = min(pd.Timestamp(p["exit_date"]) for p in tracked.values())
            exiting       = {t: p for t, p in tracked.items()
                             if pd.Timestamp(p["exit_date"]) == earliest_exit}

            for ticker, pos in exiting.items():
                cash += round(pos["exit_price"] * pos["qty"], 2)
                sold_on[ticker] = earliest_exit.date()
                logged = log_trade(pos)
                logged["_eod_prices"] = pos.get("eod_prices", {})   # ← carry eod_prices
                logged["_buy_price"]  = pos["buy_price"]
                logged["_qty"]        = pos["qty"]
                logged["_buy_date"]   = pos["buy_date"]
                logged["_exit_date"]  = pos["exit_date"]
                trade_log.append(logged)
                del portfolio[ticker]
            # advance tqdm to reflect dates processed
            newly_done = {d for d in trading_dates if d <= earliest_exit} - processed
            pbar.update(len(newly_done))
            processed |= newly_done

            vacancy = MAX_POSITIONS - len(portfolio)
            if earliest_exit >= END_DATE:
                break

            sig_date     = earliest_exit.date()
            sold_today   = {t for t, d in sold_on.items() if d == sig_date}
            candidates   = _get_candidates(earliest_exit, set(portfolio.keys()), sold_today, vacancy)
            next_buy_ts  = next((d for d in trading_dates if d > earliest_exit), None)
            if next_buy_ts is None:
                break

            if candidates:
                new_positions, cash = _buy_batch(candidates, cash, next_buy_ts.date(), vacancy)
                for pos in new_positions:
                    portfolio[pos["ticker"]] = pos
                for r in track_batch(new_positions):
                    portfolio[r["ticker"]].update(r)

            if MAX_POSITIONS - len(portfolio) > 0:
                cash = _try_fill_vacancies(next_buy_ts, cash)

    for ticker, pos in portfolio.items():
        if "exit_date" in pos:
            logged = log_trade(pos)
            logged["_eod_prices"] = pos.get("eod_prices", {})
            logged["_buy_price"]  = pos["buy_price"]
            logged["_qty"]        = pos["qty"]
            logged["_buy_date"]   = pos["buy_date"]
            logged["_exit_date"]  = pos["exit_date"]
            trade_log.append(logged)

    summary = (pd.DataFrame(trade_log)
               .sort_values(["Buy_Date", "Final_Rank"])
               .reset_index(drop=True))
    summary.to_csv(f"trade_summary_sl{rv_multiple}_ARIMAX_FINALIZED.csv", index=False)

    # ── EQUITY CURVE WITH DAILY BREAKDOWN ────────────────────────────────────────
    all_dates = pd.date_range(START_DATE, END_DATE, freq="B")
    
    rows = []
    for d in all_dates:
        d = d.date()
    
        realized = sum(
            t["PnL"] for t in trade_log
            if pd.Timestamp(t["Exit_Date"]).date() <= d
        )
        unrealized = 0.0
        for t in trade_log:
            buy_d  = pd.Timestamp(t["_buy_date"]).date()
            exit_d = pd.Timestamp(t["_exit_date"]).date()
            # position was open on this day
            if buy_d <= d < exit_d:
                eod_px = t["_eod_prices"].get(d)
                if eod_px is not None:
                    unrealized += (eod_px - t["_buy_price"]) * t["_qty"]
                else:
                    # forward-fill: find last known eod price before d
                    past = {k: v for k, v in t["_eod_prices"].items() if k <= d}
                    if past:
                        last_px = past[max(past)]
                        unrealized += (last_px - t["_buy_price"]) * t["_qty"]
        
        equity = INITIAL_CAP + realized + unrealized
        rows.append({
            "Date":        d,
            "Equity":      round(equity, 2),
            "Realized":    round(realized, 2),
            "Unrealized":  round(unrealized, 2),
            "Total_PnL":   round(realized + unrealized, 2),
        })
    
    equity_df = pd.DataFrame(rows).set_index("Date")
    equity_df["Peak_Equity"] = equity_df["Equity"].cummax()
    equity_df["DD_Pct"]      = ((equity_df["Equity"] - equity_df["Peak_Equity"])
                                / equity_df["Peak_Equity"] * 100).round(2)
    
    equity_df.to_csv(f"equity_curve_sl{rv_multiple}.csv")
    print(f"  → saved equity_curve_sl{rv_multiple}.csv")
    
    equity_series = equity_df["Equity"]



    metrics       = compute_metrics(summary, trade_log, portfolio, equity_series)
    metrics["RV_Multiple"] = rv_multiple
    print(metrics)
    return metrics


# ── RUN ALL MULTIPLES ─────────────────────────────────────────────────────────
all_metrics = []

for rv_mult in tqdm(RV_MULTIPLES, desc="RV Multiples"):
    m = run_backtest(rv_mult)
    all_metrics.append(m)

# ── SINGLE FINAL PRINT ────────────────────────────────────────────────────────
metrics_df = (pd.DataFrame(all_metrics)
              .set_index("RV_Multiple")
              .sort_index())

print("\n" + "="*70)
print("BACKTEST RESULTS ACROSS RV MULTIPLES")
print("="*70)
print(metrics_df.T.to_string())
