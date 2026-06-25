# credit + debit + naked
from breeze_connect import BreezeConnect
import csv, time, json, functools, logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

@dataclass
class Config:
    start_date:       datetime = datetime(2025, 9, 1)
    end_date:         datetime = datetime(2026, 6, 2)
    entry_start:      str      = "10:45:00"
    exit_cutoff:      str      = "15:15:00"
    lot_size:         int      = 65
    capital:          float    = 80000.0
    cost_per_trade:   float    = 120.0
    risk_free_rate:   float    = 0.07
    dividend:         float    = 0.012
    ema_short:        int      = 5
    ema_long:         int      = 15
    adx_period:       int      = 14
    rv_window:        int      = 10
    strike_step:      int      = 50
    expiry_json:      str      = "expiries_nifty.json"
    csv_file:         str      = "ts_25_26_credit_6avg_rv.csv"
    fut_csv:          str      = "nifty_fut_5m_2023-2026_filtered.csv"
    hourly_csv: str = "nifty_fut_1h_2023-2026.csv"
    breakout_lookback:int      = 6
    trade_days:       tuple    = (0,1,4)         
    chunk_seconds:    int      = 1000
    use_credit: bool = True
    use_debit:  bool = False
    use_naked:  bool = False
    use_h6_highlow:  bool = False   
    use_h6_avg:      bool = True   
    use_ma75:        bool = False
    sl_pct_of_premium: float = 0.10

CFG = Config()

GREEK_COLS     = ["iv", "delta", "gamma", "vega", "theta"]
INDICATOR_COLS = ["ema_short", "ema_long", "adx", "rsi", "rv", "atr", "macd", "macd_signal", "macd_hist", "stoch_k", "cci"]
_breeze: Optional[BreezeConnect] = None

def get_breeze() -> BreezeConnect:
    global _breeze
    if _breeze is None:
        # #DS
        # _breeze = BreezeConnect(api_key="3G5x135e8v07iz8873X11941JO1104D6")
        # _breeze.generate_session(
        #     api_secret="7z47O2524285172^9562Q1812Iq96R*8",
        #     session_token="55853655"
        # )
        #rp singhal
        _breeze = BreezeConnect(api_key="w617H90t&3_01jb06ja6015(0nt6y65W")
        _breeze.generate_session(
            api_secret="1s3a9251f(3079273g3xf3t2zsr*h284",
            session_token="55889394"
        )
        # # mohit sir
        # _breeze = BreezeConnect(api_key="=qw3v81645C94339h387K4461_520l05")
        # _breeze.generate_session(
        #     api_secret="1h87H%27q23626t448M55J5605P532y5",
        #     session_token="55876822"
        # )
    return _breeze

with open(CFG.expiry_json) as f:
    _expiry_list: List[datetime] = sorted(
        datetime.strptime(d, "%Y-%m-%d") for d in json.load(f)["Nifty"]
    )

def get_expiry_for_date(d: datetime) -> str:
    for e in _expiry_list:
        if e >= d:
            return e.strftime("%Y-%m-%dT00:00:00.000Z")
    return _expiry_list[-1].strftime("%Y-%m-%dT00:00:00.000Z")

def expiry_as_datetime(expiry: str) -> datetime:
    return datetime.strptime(expiry[:10], "%Y-%m-%d")

def with_retry(retries=3, delay=1.0):
    def dec(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for att in range(retries):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    if att == retries - 1:
                        raise
                    log.warning(f"[RETRY {att+1}] {fn.__name__}: {e}")
                    time.sleep(delay)
        return wrapper
    return dec

def _fetch_chunked(
    stock_code: str,
    exchange_code: str,
    product_type: str,
    from_dt: datetime,
    to_dt: datetime,
    expiry_date: str = "",
    right: str = "",
    strike_price: int = 0,
) -> pd.DataFrame:
    all_rows = []
    cursor   = from_dt
    step     = timedelta(seconds=CFG.chunk_seconds)

    while cursor < to_dt:
        chunk_end = min(cursor + step, to_dt)
        try:
            kw = dict(
                interval   = "1second",
                from_date  = cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                to_date    = chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                stock_code = stock_code,
                exchange_code = exchange_code,
                product_type  = product_type,
            )
            if expiry_date:  kw["expiry_date"]  = expiry_date
            if right:        kw["right"]        = right
            if strike_price: kw["strike_price"] = strike_price

            resp = get_breeze().get_historical_data_v2(**kw)
            if resp and resp.get("Success"):
                all_rows.extend(resp["Success"])
            else:
                log.warning(f"  empty chunk {stock_code} {right} {strike_price} @ {cursor:%H:%M:%S}")
        except Exception as e:
            log.warning(f"  chunk error {stock_code} {right} {strike_price} @ {cursor:%H:%M:%S}: {e}")

        cursor = chunk_end + timedelta(seconds=1)
        time.sleep(0.05)

    if not all_rows:
        log.warning(f"  no data at all for {stock_code} {right} {strike_price}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # reindex to full 1-second grid and ffill
    full_idx = pd.date_range(start=from_dt, end=to_dt, freq="1s")
    full_idx = full_idx[(full_idx.time >= pd.Timestamp("09:15").time()) &
                        (full_idx.time <= pd.Timestamp("15:29:59").time())]
    df = df.reindex(full_idx).ffill()
    return df
def fetch_one_leg(
    entry_ts: datetime,
    exit_cutoff: datetime,
    expiry: str,
    strike: int,
    right: str,
) -> pd.DataFrame:
    return _fetch_chunked(
        stock_code="NIFTY", exchange_code="NFO", product_type="options",
        from_dt=entry_ts - timedelta(seconds=5), to_dt=exit_cutoff,
        expiry_date=expiry, right=right, strike_price=strike,
    )
def fetch_two_legs(
    entry_ts: datetime,
    exit_cutoff: datetime,
    expiry: str,
    strike1: int,
    strike2: int,
    right: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tasks = {
        "leg1": dict(
            stock_code="NIFTY", exchange_code="NFO", product_type="options",
            from_dt=entry_ts, to_dt=exit_cutoff,
            expiry_date=expiry, right=right, strike_price=strike1,
        ),
        "leg2": dict(
            stock_code="NIFTY", exchange_code="NFO", product_type="options",
            from_dt=entry_ts, to_dt=exit_cutoff,
            expiry_date=expiry, right=right, strike_price=strike2,
        ),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_fetch_chunked, **kw): name
                   for name, kw in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
                log.info(f"  [fetch done] {name}  rows={len(results[name])}")
            except Exception as e:
                log.error(f"  [fetch failed] {name}: {e}")
                results[name] = pd.DataFrame()

    return results["leg1"], results["leg2"]
    
# ── Indicators (on 5-min futures CSV) ────────────────────────────────────────
@with_retry()
def _fetch_ohlcv(stock_code, exchange_code, product_type, interval,
                 from_dt, to_dt, expiry_date="", right="", strike_price=0) -> pd.DataFrame:
    kw = dict(interval=interval,
              from_date=from_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
              to_date=to_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
              stock_code=stock_code, exchange_code=exchange_code,
              product_type=product_type)
    if expiry_date:  kw["expiry_date"]  = expiry_date
    if right:        kw["right"]        = right
    if strike_price: kw["strike_price"] = strike_price
    resp = get_breeze().get_historical_data_v2(**kw)
    if not resp or not resp.get("Success"):
        raise ValueError(f"Empty: {stock_code} {interval}")
    df = pd.DataFrame(resp["Success"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]

@with_retry()
def fetch_spot_price(ts: datetime, window: int = 30) -> Optional[float]:
    try:
        df = _fetch_ohlcv("NIFTY", "NSE", "cash", "1second",
                          ts - timedelta(seconds=window), ts + timedelta(seconds=5))
        if df.empty: return None
        if ts in df.index: return float(df.loc[ts, "close"])
        idx = df.index.searchsorted(ts, side="right") - 1
        return float(df.iloc[idx]["close"]) if idx >= 0 else None
    except Exception as e:
        log.warning(f"  spot fetch error @ {ts.time()}: {e}")
        return None

def load_fut_csv() -> pd.DataFrame:
    df = pd.read_csv(CFG.fut_csv, index_col="datetime", parse_dates=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    c, h, l = df["close"], df["high"], df["low"]
    df[f"ema_{CFG.ema_short}"] = ta.ema(c, length=CFG.ema_short)
    df[f"ema_{CFG.ema_long}"]  = ta.ema(c, length=CFG.ema_long)
    adx_df = ta.adx(h, l, c, length=CFG.adx_period)
    df["adx"]     = adx_df[f"ADX_{CFG.adx_period}"] if adx_df is not None else np.nan
    df["rsi"]     = ta.rsi(c, length=14)
    df["log_ret"] = np.log(c / c.shift(1))
    df["rv"]      = df["log_ret"].rolling(CFG.rv_window).std() * np.sqrt(375 * 252)
    n = CFG.breakout_lookback
    df["breakout_high"] = h.shift(1).rolling(n).max()
    df["breakout_low"]  = l.shift(1).rolling(n).min()
    df["vol_sma_5"]  = df["volume"].rolling(5).mean()
    df["vol_sma_15"] = df["volume"].rolling(15).mean()
    # ATR
    atr_df = ta.atr(h, l, c, length=14)
    df["atr"] = atr_df if atr_df is not None else np.nan
    
    # MACD (default 12, 26, 9)
    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df["macd"]        = macd_df["MACD_12_26_9"]
        df["macd_signal"] = macd_df["MACDs_12_26_9"]
        df["macd_hist"]   = macd_df["MACDh_12_26_9"]
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan
    
    # Stochastic K (default k=14, d=3, smooth_k=3)
    stoch_df = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    if stoch_df is not None:
        df["stoch_k"] = stoch_df["STOCHk_14_3_3"]
    else:
        df["stoch_k"] = np.nan
    
    # CCI
    cci_df = ta.cci(h, l, c, length=20)
    df["cci"] = cci_df if cci_df is not None else np.nan
    return df.drop(columns=["log_ret"])

def indic_snapshot(df: pd.DataFrame, ts) -> dict:
    try:
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0: return {}
        row = df.iloc[idx]
        
        snap = {
            "ema_short": float(row[f"ema_{CFG.ema_short}"]) if pd.notna(row.get(f"ema_{CFG.ema_short}")) else None,
            "ema_long":  float(row[f"ema_{CFG.ema_long}"])  if pd.notna(row.get(f"ema_{CFG.ema_long}"))  else None,
        }
        # all other INDICATOR_COLS fetched generically
        for k in ["adx", "rsi", "rv", "atr", "macd", "macd_signal", "macd_hist", "stoch_k", "cci"]:
            snap[k] = float(row[k]) if pd.notna(row.get(k)) else None
        return snap
    except:
        return {}
def check_entry_signal(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    candle_ts,
) -> Optional[str]:

    pos = df_5m.index.searchsorted(candle_ts, side="right")
    if pos < 4: return None

    cur   = df_5m.iloc[pos - 1]
    close = cur.get("close")
    ema_s = cur.get(f"ema_{CFG.ema_short}")
    ema_l = cur.get(f"ema_{CFG.ema_long}")
    vol5  = cur.get("vol_sma_5")
    vol15 = cur.get("vol_sma_15")
    ma75  = cur.get("ma75")
    rv    = cur.get("rv")

    if rv is None or np.isnan(float(rv)) or not float(rv) <= 0.2:
        return None
    # always-required values
    always_required = [close, ema_s, ema_l, vol5, vol15]
    if CFG.use_ma75:
        always_required.append(ma75)

    if any(v is None or (isinstance(v, float) and np.isnan(float(v)))
           for v in always_required):
        return None

    close = float(close); ema_s = float(ema_s); ema_l = float(ema_l)
    vol5  = float(vol5);  vol15 = float(vol15)
    ma75  = float(ma75) if ma75 is not None else None

    # ── always: ADX rising 3 bars ─────────────────────────────────────────
    adx_vals = df_5m["adx"].iloc[pos - 3: pos].values
    if any(np.isnan(adx_vals)): return None
    if not (adx_vals[0] < adx_vals[1] < adx_vals[2]): return None

    # ── always: volume filter ─────────────────────────────────────────────
    if not (vol5 > vol15): return None

    # ── optional: hourly h6 values (only compute if needed) ──────────────
    h6_high_max = h6_low_min = h6_close_avg = None
    if CFG.use_h6_highlow or CFG.use_h6_avg:
        h_closed = df_1h[df_1h.index < candle_ts - pd.Timedelta(hours=1)]
        if len(h_closed) < 6: return None
        h6 = h_closed.iloc[-6:]
        if CFG.use_h6_highlow:
            h6_high_max = float(h6["high"].max())
            h6_low_min  = float(h6["low"].min())
        if CFG.use_h6_avg:
            h6_close_avg = float(h6["close"].mean())

    bull = (close > ema_s > ema_l)                                         
    if CFG.use_ma75:        bull = bull and (close > ma75)
    if CFG.use_h6_highlow:  bull = bull and (close > h6_high_max)
    if CFG.use_h6_avg:      bull = bull and (close > h6_close_avg)

    if bull:
        log.info(
            f"  [BULL] @ {candle_ts} close={close:.2f} "
            f"ema5={ema_s:.2f} ema15={ema_l:.2f} "
            f"ma75={ma75 if CFG.use_ma75 else 'OFF'} "
            f"h6_high={h6_high_max if CFG.use_h6_highlow else 'OFF'} "
            f"h6_avg={h6_close_avg if CFG.use_h6_avg else 'OFF'} "
            f"adx=[{adx_vals[0]:.1f},{adx_vals[1]:.1f},{adx_vals[2]:.1f}] "
            f"vol5={vol5:.0f} vol15={vol15:.0f}"
        )
        return "bull"

    # ── BEAR ──────────────────────────────────────────────────────────────
    bear = (close < ema_s < ema_l)                                          # always
    if CFG.use_ma75:        bear = bear and (close < ma75)
    if CFG.use_h6_highlow:  bear = bear and (close < h6_low_min)
    if CFG.use_h6_avg:      bear = bear and (close < h6_close_avg)

    if bear:
        log.info(
            f"  [BEAR] @ {candle_ts} close={close:.2f} "
            f"ema5={ema_s:.2f} ema15={ema_l:.2f} "
            f"ma75={ma75 if CFG.use_ma75 else 'OFF'} "
            f"h6_low={h6_low_min if CFG.use_h6_highlow else 'OFF'} "
            f"h6_avg={h6_close_avg if CFG.use_h6_avg else 'OFF'} "
            f"adx=[{adx_vals[0]:.1f},{adx_vals[1]:.1f},{adx_vals[2]:.1f}] "
            f"vol5={vol5:.0f} vol15={vol15:.0f}"
        )
        return "bear"

    return None

def _d1d2(S, K, T, r, q, sig):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return d1, d1 - sig * np.sqrt(T)

def bs_price(S, K, T, r, q, sig, opt):
    if T <= 0 or sig <= 0:
        return max(0., S - K) if opt == "call" else max(0., K - S)
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def calc_iv(price, S, K, T, r, q, opt):
    try:
        if T <= 0 or price <= 0: return np.nan
        return brentq(lambda s: bs_price(S, K, T, r, q, s, opt) - price,
                      1e-4, 5.0, maxiter=50)
    except:
        return np.nan

def compute_greeks(spot, strike, opt_type, ts, expiry, opt_price) -> dict:
    expiry_dt = expiry_as_datetime(expiry).replace(hour=15, minute=30)
    T = (expiry_dt - ts).total_seconds() / (365 * 24 * 3600)
    r, q = CFG.risk_free_rate, CFG.dividend
    iv = calc_iv(opt_price, spot, strike, T, r, q, opt_type)
    if np.isnan(iv) or iv <= 0:
        return {g: np.nan for g in GREEK_COLS}
    d1, d2 = _d1d2(spot, strike, T, r, q, iv)
    Npd1  = norm.pdf(d1)
    gamma = np.exp(-q*T) * Npd1 / (spot * iv * np.sqrt(T))
    vega  = spot * np.exp(-q*T) * Npd1 * np.sqrt(T) / 100
    if opt_type == "call":
        delta = np.exp(-q*T) * norm.cdf(d1)
        theta = (
            -np.exp(-q*T)*spot*Npd1*iv / (2*np.sqrt(T))
            + q*spot*np.exp(-q*T)*norm.cdf(d1)
            - r*strike*np.exp(-r*T)*norm.cdf(d2)
        ) / 365
    else:
        delta = np.exp(-q*T) * (norm.cdf(d1) - 1)
        theta = (
            -np.exp(-q*T)*spot*Npd1*iv / (2*np.sqrt(T))
            - q*spot*np.exp(-q*T)*norm.cdf(-d1)
            + r*strike*np.exp(-r*T)*norm.cdf(-d2)
        ) / 365
    if T < 1/365:
        intrinsic = max(0., spot-strike) if opt_type == "call" else max(0., strike-spot)
        theta = -min(abs(theta), max(0., opt_price - intrinsic))
    return {"iv": iv, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def run_exit_loop(
    df_leg1, df_leg2,
    strike1, strike2, expiry,
    entry_ts, entry_net_prem, exit_cutoff, rpt
):
    sl_level = 2*entry_net_prem - 3
    trade_max_dd = 0.0
    max_profit = 0.0
    last_ts = entry_ts
    last_p1 = last_p2 = np.nan 
    common_idx = df_leg1.index.intersection(df_leg2.index)
    common_idx = common_idx[common_idx > entry_ts]

    for ts in common_idx:
        p1 = df_leg1.loc[ts, "close"]
        p2 = df_leg2.loc[ts, "close"]
        if np.isnan(p1) or np.isnan(p2) :
            continue

        last_ts, last_p1, last_p2 = ts, p1, p2
        current_net = p1 - p2
        pnl = entry_net_prem - current_net  
        if pnl > max_profit: max_profit = pnl
        adverse_pct = ((max_profit - pnl) / max_profit * 100) if max_profit > 0.1 else 0.0
        trade_max_dd = max(trade_max_dd, adverse_pct)
        rpt.track_unrealized(entry_net_prem,current_net )

        # SL check first (cheap)
        if current_net >= sl_level:
            return _exit_result(ts, p1, p2, "SL", trade_max_dd, sl_level)
            
        if ts >= exit_cutoff:
            return _exit_result(ts, p1, p2, "EOD", trade_max_dd, sl_level)

    # fallback
    if not np.isnan(last_p1):
        return _exit_result(last_ts, last_p1, last_p2, "EOD", trade_max_dd, sl_level)
    return _exit_result(entry_ts, entry_net_prem, 0.0, "NO_EXIT", trade_max_dd, sl_level)
def run_exit_loop_bull_call(
    df_leg1, df_leg2,
    strike1, strike2, expiry,
    entry_ts, entry_net_prem, exit_cutoff, rpt
):
    sl_level     = 3.0
    take_profit  = 2.0 * entry_net_prem
    trade_max_dd = 0.0
    max_profit   = 0.0
    last_ts = entry_ts
    last_p1 = last_p2 = np.nan

    common_idx = df_leg1.index.intersection(df_leg2.index)
    common_idx = common_idx[common_idx > entry_ts]

    for ts in common_idx:
        p1 = df_leg1.loc[ts, "close"]
        p2 = df_leg2.loc[ts, "close"]
        if np.isnan(p1) or np.isnan(p2):
            continue

        last_ts, last_p1, last_p2 = ts, p1, p2
        current_net = p1 - p2
        pnl = current_net - entry_net_prem          # debit grows = profit
        if pnl > max_profit: max_profit = pnl
        adverse_pct = ((max_profit - pnl) / max_profit * 100) if max_profit > 0.1 else 0.0
        trade_max_dd = max(trade_max_dd, adverse_pct)
        rpt.track_unrealized(current_net, entry_net_prem)

        if current_net <= sl_level:
            return _exit_result(ts, p1, p2, "SL", trade_max_dd, sl_level)
        if current_net >= take_profit:
            return _exit_result(ts, p1, p2, "TP", trade_max_dd, sl_level)

        if ts >= exit_cutoff:
            return _exit_result(ts, p1, p2, "EOD", trade_max_dd, sl_level)

    if not np.isnan(last_p1):
        return _exit_result(last_ts, last_p1, last_p2, "EOD", trade_max_dd, sl_level)
    return _exit_result(entry_ts, entry_net_prem, 0.0, "NO_EXIT", trade_max_dd, sl_level)


def _exit_result(ts, p1, p2, exit_type, max_dd, sl_level) -> dict:
    return {
        "exit_time":       ts,
        "exit_leg1_price": p1,
        "exit_leg2_price": p2,
        "exit_type":       exit_type,
        "trade_max_dd":    max_dd,
        "initial_sl":      sl_level,
        "final_trail_sl":  sl_level,
    }
def run_exit_loop_naked(df_leg1, entry_ts, entry_premium, exit_cutoff, rpt) -> dict:
    sl_dist      = entry_premium * CFG.sl_pct_of_premium
    trail_sl     = entry_premium - sl_dist
    initial_sl   = trail_sl
    peak_price   = entry_premium
    max_profit   = 0.0
    trade_max_dd = 0.0
    last_ts      = entry_ts
    last_close   = entry_premium

    idx = df_leg1.index[df_leg1.index > entry_ts]

    for ts in idx:
        c = df_leg1.loc[ts, "close"]
        if np.isnan(c): continue
        last_ts, last_close = ts, c

        if c > peak_price:
            peak_price = c
            max_profit = peak_price - entry_premium
            trail_sl   = peak_price - sl_dist

        dd = (peak_price - c) / peak_price * 100 if peak_price > 0 else 0.0
        trade_max_dd = max(trade_max_dd, dd)
        rpt.track_unrealized(c, entry_premium)

        if c <= trail_sl:
            exit_type = "Initial SL" if max_profit == 0.0 else "Trailing SL"
            return {
                "exit_time": ts, "exit_price": c, "exit_type": exit_type,
                "max_profit": max_profit, "trade_max_dd": trade_max_dd,
                "trade_pnl": (c - entry_premium) * CFG.lot_size,
                "initial_sl": initial_sl, "final_trail_sl": trail_sl,
            }

        if ts >= exit_cutoff:
            return {
                "exit_time": ts, "exit_price": c, "exit_type": "EOD",
                "max_profit": max_profit, "trade_max_dd": trade_max_dd,
                "trade_pnl": (c - entry_premium) * CFG.lot_size,
                "initial_sl": initial_sl, "final_trail_sl": trail_sl,
            }

    return {
        "exit_time": last_ts, "exit_price": last_close, "exit_type": "EOD",
        "max_profit": max_profit, "trade_max_dd": trade_max_dd,
        "trade_pnl": (last_close - entry_premium) * CFG.lot_size,
        "initial_sl": initial_sl, "final_trail_sl": trail_sl,
    }

# ── CSV ───────────────────────────────────────────────────────────────────────
_G1_ENTRY  = [f"entry_leg1_{g}" for g in GREEK_COLS]
_G2_ENTRY  = [f"entry_leg2_{g}" for g in GREEK_COLS]
_G1_EXIT   = [f"exit_leg1_{g}"  for g in GREEK_COLS]
_G2_EXIT   = [f"exit_leg2_{g}"  for g in GREEK_COLS]
_IND_ENTRY = [f"entry_{c}"      for c in INDICATOR_COLS]
_IND_EXIT  = [f"exit_{c}"       for c in INDICATOR_COLS]

_CSV_HEADER = [
    "Date", "Trade_Num_Today",
    "Strike1_Sell", "Strike2_Buy", "Strategy",
    "Entry_Time",
    "Entry_Leg1_Premium", "Entry_Leg2_Premium", "Entry_Net_Premium",
    "Exit_Time",
    "Exit_Leg1_Premium", "Exit_Leg2_Premium", "Exit_Net_Premium",
    "Exit_Type",
    "Max_Profit_Per_Lot", "Max_Loss_Per_Lot",
    "Initial_SL", "Final_Trail_SL", "Trade_Max_DD_Pct",
    "PnL_Per_qty", "Quantity", "Total_PnL",
    "Cumulative_PnL", "Capital", "Return_Pct","vix",
    *_G1_ENTRY, *_G2_ENTRY,
    *_G1_EXIT,  *_G2_EXIT,
    *_IND_ENTRY, *_IND_EXIT,
]

_fmt = lambda v: (
    "NA" if (v is None or (isinstance(v, float) and np.isnan(v)))
    else f"{v:.4f}"
)

def _init_csv():
    try:
        with open(CFG.csv_file, "x", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)
    except FileExistsError:
        pass

def log_csv(
    trade_date, trade_num,
    strike1, strike2,strategy,
    entry_ts,
    e_leg1_prem, e_leg2_prem, entry_net_prem,
    exit_ts,
    x_leg1_prem, x_leg2_prem, exit_net_prem,
    exit_type,
    max_profit_lot, max_loss_lot,
    isl, fsl, trade_max_dd,
    pnl_lot, qty, total_pnl,
    cum_pnl,
    vix_open, 
    eg1, eg2,
    xg1, xg2,
    ei,  xi,
):
    eg1 = eg1 or {}; eg2 = eg2 or {}
    xg1 = xg1 or {}; xg2 = xg2 or {}
    row = [
        trade_date.strftime("%Y-%m-%d"), trade_num,
        strike1, strike2, strategy,
        entry_ts.strftime("%H:%M:%S"),
        _fmt(e_leg1_prem), _fmt(e_leg2_prem), _fmt(entry_net_prem),
        exit_ts.strftime("%H:%M:%S"),
        _fmt(x_leg1_prem), _fmt(x_leg2_prem), _fmt(exit_net_prem),
        exit_type,
        _fmt(max_profit_lot), _fmt(max_loss_lot),
        _fmt(isl), _fmt(fsl), _fmt(trade_max_dd),
        _fmt(pnl_lot), qty, _fmt(total_pnl),
        _fmt(cum_pnl),
        _fmt(CFG.capital + cum_pnl),
        _fmt(cum_pnl / CFG.capital * 100),
        _fmt(vix_open),
        *[_fmt(eg1.get(g)) for g in GREEK_COLS],
        *[_fmt(eg2.get(g)) for g in GREEK_COLS],
        *[_fmt(xg1.get(g)) for g in GREEK_COLS],
        *[_fmt(xg2.get(g)) for g in GREEK_COLS],
        *[_fmt(ei.get(c)) for c in INDICATOR_COLS],
        *[_fmt(xi.get(c)) for c in INDICATOR_COLS],
    ]
    with open(CFG.csv_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Report ────────────────────────────────────────────────────────────────────
@dataclass
class Report:
    cumulative_pnl: float = 0.
    peak_equity:    float = 0.
    max_drawdown:   float = 0.
    total:  int = 0
    wins:   int = 0
    losses: int = 0
    win_pnls:  list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def record_eod(self, trade_date: datetime):
        eq = CFG.capital + self.cumulative_pnl
        self._track_equity(eq)
        self.equity_curve.append((trade_date.date(), eq))

    def _cagr(self) -> float:
        if len(self.equity_curve) < 2: return np.nan
        s_eq, e_eq = self.equity_curve[0][1], self.equity_curve[-1][1]
        s_dt = pd.Timestamp(self.equity_curve[0][0])
        e_dt = pd.Timestamp(self.equity_curve[-1][0])
        years = (e_dt - s_dt).days / 365.25
        if years <= 0 or s_eq <= 0: return np.nan
        return (e_eq / s_eq) ** (1 / years) - 1

    def _track_equity(self, eq):
        if eq > self.peak_equity: self.peak_equity = eq
        if self.peak_equity > 0:
            dd = (self.peak_equity - eq) / self.peak_equity * 100
            if dd > self.max_drawdown: self.max_drawdown = dd

    def update(self, pnl):
        net = pnl - CFG.cost_per_trade
        self.cumulative_pnl += net
        self.total += 1
        if net >= 0: self.wins += 1;   self.win_pnls.append(net)
        else:        self.losses += 1; self.loss_pnls.append(net)

    def track_unrealized(self, net_spread_price, entry_net_prem):
        unrealized = (net_spread_price - entry_net_prem ) * CFG.lot_size
        eq = CFG.capital + self.cumulative_pnl + unrealized
        self._track_equity(eq)

    def _sharpe_and_vol(self):
        if len(self.equity_curve) < 2: return np.nan, np.nan
        equities   = np.array([e for _, e in self.equity_curve])
        daily_rets = np.diff(equities) / equities[:-1]
        std = np.std(daily_rets, ddof=1)
        if std == 0: return np.nan, np.nan
        vol    = std * np.sqrt(252)
        sharpe = (np.mean(daily_rets) - CFG.risk_free_rate / 252) / std * np.sqrt(252)
        return sharpe, vol

    def summary(self):
        sharpe, vol = self._sharpe_and_vol()
        cagr        = self._cagr()
        avg_win     = np.mean(self.win_pnls)  if self.win_pnls  else 0.
        avg_loss    = np.mean(self.loss_pnls) if self.loss_pnls else 0.
        avg_pnl     = np.mean(self.win_pnls + self.loss_pnls) if self.total else 0.
        win_rate    = self.wins / self.total * 100 if self.total else 0.
        log.info("=" * 60)
        log.info(f"  CAGR         : {cagr*100:.2f}%")
        log.info(f"  Total Trades : {self.total}  |  W/L : {self.wins}/{self.losses}")
        log.info(f"  Win Rate     : {win_rate:.1f}%")
        log.info(f"  Avg PnL      : ₹{avg_pnl:.2f}")
        log.info(f"  Avg Win      : ₹{avg_win:.2f}")
        log.info(f"  Avg Loss     : ₹{avg_loss:.2f}")
        log.info(f"  Net PnL      : ₹{self.cumulative_pnl:.2f}")
        log.info(f"  Max DD       : {self.max_drawdown:.2f}%")
        log.info(f"  Ann. Vol     : {vol*100:.2f}%")
        log.info(f"  Ann. Sharpe  : {sharpe:.2f}")
        log.info("=" * 60)

    def save_equity_curve(self, path="equity_curve.csv"):
        if not self.equity_curve: return
        df = pd.DataFrame(self.equity_curve, columns=["date", "equity"])
        df["daily_ret_pct"] = df["equity"].pct_change() * 100
        df["cum_ret_pct"]   = (df["equity"] / CFG.capital - 1) * 100
        df.to_csv(path, index=False)
        log.info(f"Equity curve → {path}  ({len(df)} rows)")


def load_hourly_csv() -> pd.DataFrame:
    df = pd.read_csv(CFG.hourly_csv, index_col="datetime", parse_dates=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]


def run_backtest():
    rpt = Report(peak_equity=CFG.capital)
    _init_csv()

    df_all = compute_indicators(load_fut_csv())
    df_all["ma75"] = df_all["close"].rolling(75).mean()   # needed for entry signal
    
    df_1h = load_hourly_csv()

    vix_df = pd.read_csv("vix_daily_3yr_new.csv", parse_dates=["datetime"])
    vix_df["date"] = vix_df["datetime"].dt.date
    vix_map = vix_df.set_index("date")["open"].to_dict()

    current_date = CFG.start_date
    while current_date <= CFG.end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1); continue

        if current_date.weekday() not in CFG.trade_days:
            rpt.record_eod(current_date)
            current_date += timedelta(days=1); continue

        vix_open = vix_map.get(current_date.date(), np.nan)

        entry_start  = datetime.combine(
            current_date, datetime.strptime(CFG.entry_start,  "%H:%M:%S").time()
        )
        exit_cutoff  = datetime.combine(
            current_date, datetime.strptime(CFG.exit_cutoff,  "%H:%M:%S").time()
        )
        entry_cutoff = current_date.replace(hour=15, minute=0, second=0, microsecond=0)
        expiry       = get_expiry_for_date(current_date)

        df_day = df_all[df_all.index.date == current_date.date()]
        if df_day.empty:
            current_date += timedelta(days=1); continue

        log.info(f"\n{'='*52}\n{current_date.date()}  expiry={expiry[:10]}\n{'='*52}")

        scan_bars       = [ts for ts in df_day.index if entry_start <= ts < entry_cutoff]
        daily_trade_num = 0
        i               = 0

        while i < len(scan_bars):
            candle_ts = scan_bars[i]

            # ── entry signal: returns "bull", "bear", or None ─────────────────
            side = check_entry_signal(df_all, df_1h, candle_ts)
            if side is None:
                i += 1; continue

            entry_ts = candle_ts + timedelta(minutes=5)
            if entry_ts > exit_cutoff:
                i += 1; continue

            spot = fetch_spot_price(entry_ts)
            if spot is None:
                i += 1; continue
            atm = int(round(spot / CFG.strike_step)) * CFG.strike_step
            
            if CFG.use_credit:
                if side == "bull":
                    strike1 = atm - 50
                    strike2 = atm - 250
                    right   = "put"
                    strategy = "Bull_Put_Spread"
                else:
                    strike1 = atm + 50
                    strike2 = atm + 250
                    right   = "call"
                    strategy = "Bear_Call_Spread"
            
            elif CFG.use_debit:
                if side == "bull":
                    strike1 = atm          # buy ATM call
                    strike2 = atm + 150    # sell OTM call
                    right   = "call"
                    strategy = "Bull_Call_Spread"
                else:
                    strike1 = atm          # buy ATM put
                    strike2 = atm - 150    # sell OTM put
                    right   = "put"
                    strategy = "Bear_Put_Spread"
            
            elif CFG.use_naked:
                strike1  = atm
                strike2  = None            # no second leg
                right    = "call" if side == "bull" else "put"
                strategy = "Naked_Call" if side == "bull" else "Naked_Put"
                            
            if CFG.use_naked:
                df_leg1 = fetch_one_leg(entry_ts, exit_cutoff, expiry, strike1, right)
                df_leg2 = pd.DataFrame()
            else:
                df_leg1, df_leg2 = fetch_two_legs(
                    entry_ts - timedelta(seconds=5), exit_cutoff,
                    expiry, strike1, strike2, right
                )
                
            e_leg1_prem = float(df_leg1["close"].loc[entry_ts]) if entry_ts in df_leg1.index \
                          else float(df_leg1["close"].asof(entry_ts))
            
            if CFG.use_naked:
                e_leg2_prem    = np.nan
                entry_net_prem = e_leg1_prem   # cost of buying the option
            else:
                e_leg2_prem    = float(df_leg2["close"].loc[entry_ts]) if entry_ts in df_leg2.index \
                                 else float(df_leg2["close"].asof(entry_ts))
                entry_net_prem = e_leg1_prem - e_leg2_prem

         
            if np.isnan(e_leg1_prem):
                log.warning("NaN leg1 premium, skipping"); i += 1; continue
            if e_leg1_prem <= 10 :
                log.warning(f"  Bad premiums: {e_leg1_prem}, {e_leg2_prem}")
                i += 1; continue  
            if not CFG.use_naked and np.isnan(e_leg2_prem):
                log.warning("NaN leg2 premium, skipping"); i += 1; continue
            if not CFG.use_naked and entry_net_prem <= 0:
                log.warning(f"Net premium <= 0 ({entry_net_prem:.2f}), skipping"); i += 1; continue

                
            if entry_net_prem <= 0:
                log.warning(f"  Net premium <= 0 ({entry_net_prem:.2f}), skipping")
                i += 1; continue
            
            if CFG.use_naked:
                max_profit_lot = np.nan
                max_loss_lot   = entry_net_prem * CFG.lot_size
            
            elif CFG.use_credit:
                strike_width   = abs(strike2 - strike1)
                max_profit_lot = entry_net_prem * CFG.lot_size
                max_loss_lot   = (strike_width - entry_net_prem) * CFG.lot_size
            
            elif CFG.use_debit:
                strike_width   = abs(strike2 - strike1)
                max_profit_lot = (strike_width - entry_net_prem) * CFG.lot_size
                max_loss_lot   = entry_net_prem * CFG.lot_size
                

            entry_g1 = compute_greeks(spot, strike1, right, entry_ts, expiry, e_leg1_prem)
            entry_g2 = compute_greeks(spot, strike2, right, entry_ts, expiry, e_leg2_prem) \
                       if not CFG.use_naked else {}

            entry_indic = indic_snapshot(df_all, candle_ts)
            
            if CFG.use_credit:
                res = run_exit_loop(
                    df_leg1, df_leg2, strike1, strike2, expiry,
                    entry_ts, entry_net_prem, exit_cutoff, rpt
                )
                exit_p1 = res["exit_leg1_price"]
                exit_p2 = res["exit_leg2_price"]
                exit_net_prem = exit_p1 - exit_p2
                trade_pnl = (entry_net_prem - exit_net_prem) * CFG.lot_size
            
            elif CFG.use_debit:
                res = run_exit_loop_bull_call(
                    df_leg1, df_leg2, strike1, strike2, expiry,
                    entry_ts, entry_net_prem, exit_cutoff, rpt
                )
                exit_p1 = res["exit_leg1_price"]
                exit_p2 = res["exit_leg2_price"]
                exit_net_prem = exit_p1 - exit_p2
                trade_pnl = (exit_net_prem - entry_net_prem) * CFG.lot_size   # debit: profit when spread widens
            
            elif CFG.use_naked:
                # need a spot stream for the naked exit loop — reuse df_leg1 as the stream
                res = run_exit_loop_naked(
                    df_leg1, entry_ts, e_leg1_prem, exit_cutoff, rpt
                )
                exit_p1       = res["exit_price"]
                exit_p2       = np.nan
                exit_net_prem = exit_p1
                trade_pnl     = res["trade_pnl"]

            exit_spot  = fetch_spot_price(res["exit_time"])
            exit_g1    = compute_greeks(exit_spot, strike1, right, res["exit_time"], expiry, exit_p1)
            exit_g2  = compute_greeks(exit_spot, strike2, right, res["exit_time"], expiry, exit_p2) \
                       if not CFG.use_naked else {}
            exit_indic = indic_snapshot(df_all, res["exit_time"].replace(second=0, microsecond=0))

            rpt.update(trade_pnl)
            if CFG.use_naked:
                total_pnl     = trade_pnl - (CFG.cost_per_trade/2)
            else:
                total_pnl     = trade_pnl - CFG.cost_per_trade
            pnl_lot       = trade_pnl / CFG.lot_size
            daily_trade_num += 1

            log_csv(
                current_date,        daily_trade_num,
                strike1,             strike2 if strike2 is not None else "NA",
                strategy,
                entry_ts,
                e_leg1_prem,         e_leg2_prem,        entry_net_prem,
                res["exit_time"],
                exit_p1,             exit_p2,             exit_net_prem,
                res["exit_type"],
                max_profit_lot,      max_loss_lot,
                res["initial_sl"],   res["final_trail_sl"], res["trade_max_dd"],
                pnl_lot,             CFG.lot_size,          total_pnl,
                rpt.cumulative_pnl,
                vix_open,
                entry_g1, entry_g2,
                exit_g1,  exit_g2,
                entry_indic, exit_indic,
            )

            log.info(
                f"  [DONE] side={side} exit={res['exit_time'].time()} "
                f"type={res['exit_type']} PnL=₹{total_pnl:.2f}  "
                f"CumPnL=₹{rpt.cumulative_pnl:.2f}  DD={rpt.max_drawdown:.2f}  "
                f"peak equity={rpt.peak_equity:.2f}"
            )

            if daily_trade_num >= 3:
                log.info("  [MAX TRADES] stopping day after 3 trades")
                break

            i = next(
                (j for j, ts in enumerate(scan_bars) if ts > res["exit_time"]),
                len(scan_bars),
            )

        rpt.record_eod(current_date)
        current_date += timedelta(days=1)

    rpt.summary()
    rpt.save_equity_curve()
    return rpt


if __name__ == "__main__":
    run_backtest()
