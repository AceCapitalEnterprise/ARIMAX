from breeze_connect import BreezeConnect
import csv, time, json, functools, logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    start_date: datetime = datetime(2025, 9, 1);  end_date:    datetime = datetime(2026, 5, 11)
    entry_start: str = "09:15:00";                exit_cutoff: str = "15:15:00"
    lot_size: int = 65;                           capital:     float = 22_000.0
    cost_per_trade: float = 60.0;                 risk_free_rate: float = 0.07
    dividend: float = 0.012;                      ema_short:   int = 5
    ema_long: int = 20;                           iv_min:      float = 0.05
    adx_lookback: int = 3;                        adx_period:  int = 14
    rv_window: int = 10;                          sl_pct_of_premium: float = 0.10
    strike_step: int = 50;                        opt_chunk_rows: int = 1000
    opt_lead_secs: int = 5
    expiry_json: str = "expiries_nifty.json";     csv_file: str = "trades_summary_25_26_demo.csv"
    fut_csv: str = "nifty_fut_1m_2023-2026.csv"
    vix_min: float = 13.0   
    vix_max: float = 20.0   
    trade_days: tuple = (2,)# 0=Mon … 4=Fri;  (0, 1, 2, 3, 4)
CFG = Config()
GREEK_COLS     = ["iv", "delta", "gamma", "vega", "theta"]
INDICATOR_COLS = ["ema_short", "ema_long", "adx", "rsi", "rv"]
GREEK_THRESHOLDS = {g: {"min": None, "max": None} for g in GREEK_COLS}

# ── Breeze ────────────────────────────────────────────────────────────────────
_breeze: Optional[BreezeConnect] = None
def get_breeze() -> BreezeConnect:
    global _breeze
    if _breeze is None:
        _breeze = BreezeConnect(api_key="=qw3v81645C94339h387K4461_520l05")
        _breeze.generate_session(api_secret="1h87H%27q23626t448M55J5605P532y5", session_token="55942496")
    return _breeze

# ── Expiry ────────────────────────────────────────────────────────────────────
with open(CFG.expiry_json) as f:
    _expiry_list: List[datetime] = sorted(datetime.strptime(d, "%Y-%m-%d") for d in json.load(f)["Nifty"])

def get_expiry_for_date(d: datetime) -> str:
    for e in _expiry_list:
        if e >= d: return e.strftime("%Y-%m-%dT00:00:00.000Z")
    return _expiry_list[-1].strftime("%Y-%m-%dT00:00:00.000Z")

def expiry_as_datetime(expiry: str) -> datetime:
    return datetime.strptime(expiry[:10], "%Y-%m-%d")

# ── Retry ─────────────────────────────────────────────────────────────────────
def with_retry(retries=3, delay=1.0):
    def dec(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for att in range(retries):
                try: return fn(*a, **kw)
                except Exception as e:
                    if att == retries - 1: raise
                    log.warning(f"[RETRY {att+1}] {fn.__name__}: {e}"); time.sleep(delay)
        return wrapper
    return dec

# ── Data fetchers ─────────────────────────────────────────────────────────────
@with_retry()
def _fetch_ohlcv(stock_code, exchange_code, product_type, interval,
                 from_dt, to_dt, expiry_date="", right="", strike_price=0) -> pd.DataFrame:
    kw = dict(interval=interval,
              from_date=from_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
              to_date=to_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
              stock_code=stock_code, exchange_code=exchange_code, product_type=product_type)
    if expiry_date:  kw["expiry_date"]  = expiry_date
    if right:        kw["right"]        = right
    if strike_price: kw["strike_price"] = strike_price
    resp = get_breeze().get_historical_data_v2(**kw)
    if not resp or not resp.get("Success"): raise ValueError(f"Empty: {stock_code} {interval}")
    df = pd.DataFrame(resp["Success"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]

@with_retry()
def fetch_spot(ts: datetime, window: int = 30) -> Optional[float]:
    try:
        df = _fetch_ohlcv("NIFTY", "NSE", "cash", "1second",
                          ts - timedelta(seconds=window), ts + timedelta(seconds=5))
        if df.empty: return None
        if ts in df.index: return float(df.loc[ts, "close"])
        idx = df.index.searchsorted(ts, side="right") - 1
        return float(df.iloc[idx]["close"]) if idx >= 0 else None
    except Exception as e:
        log.warning(f"  spot fetch error @ {ts.time()}: {e}"); return None

def load_fut_csv() -> pd.DataFrame:
    df = pd.read_csv(CFG.fut_csv, index_col="datetime", parse_dates=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]

# ── Indicators ────────────────────────────────────────────────────────────────
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
    df["adx"] = adx_df[f"ADX_{CFG.adx_period}"] if adx_df is not None else np.nan
    df["rsi"]     = ta.rsi(c, length=14)
    df["log_ret"] = np.log(c / c.shift(1))
    df["rv"]      = df["log_ret"].rolling(CFG.rv_window).std() * np.sqrt(375 * 252)
    return df.drop(columns=["log_ret"])

def indic_snapshot(df: pd.DataFrame, ts) -> dict:
    try:
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0: return {}
        row = df.iloc[idx]
        return {
            "ema_short": float(row[f"ema_{CFG.ema_short}"]) if pd.notna(row.get(f"ema_{CFG.ema_short}")) else None,
            "ema_long":  float(row[f"ema_{CFG.ema_long}"])  if pd.notna(row.get(f"ema_{CFG.ema_long}"))  else None,
            **{k: (float(row[k]) if pd.notna(row.get(k)) else None) for k in ["adx", "rsi", "rv"]},
        }
    except: return {}

# ── Entry signal ──────────────────────────────────────────────────────────────
def check_entry_signal(df: pd.DataFrame, candle_ts) -> bool:
    n = CFG.adx_lookback
    pos = df.index.searchsorted(candle_ts, side="right")
    if pos < n + 1: return False
    cur = df.iloc[pos - 1]
    c, es, el = float(cur["close"]), float(cur[f"ema_{CFG.ema_short}"]), float(cur[f"ema_{CFG.ema_long}"])
    if not (c > es > el): return False
    adx_vals = df.iloc[pos - n: pos]["adx"].values
    if any(np.isnan(adx_vals)): return False
    if not all(adx_vals[i] > adx_vals[i-1] for i in range(1, n)): return False
    log.info(f"  [SIGNAL] @ {candle_ts.time()} c={c:.2f} ema{CFG.ema_short}={es:.2f} "
             f"ema{CFG.ema_long}={el:.2f} ADX={adx_vals[-1]:.2f} (rising {n} bars)")
    return True

# ── Greeks ────────────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, q, sig):
    d1 = (np.log(S/K) + (r - q + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    return d1, d1 - sig*np.sqrt(T)

def bs_price(S, K, T, r, q, sig, opt):
    if T <= 0 or sig <= 0: return max(0., S-K) if opt=="call" else max(0., K-S)
    d1, d2 = _d1d2(S, K, T, r, q, sig)
    if opt == "call": return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def calc_iv(price, S, K, T, r, q, opt):
    try:
        if T <= 0 or price <= 0: return np.nan
        return brentq(lambda s: bs_price(S, K, T, r, q, s, opt) - price, 1e-4, 5.0, maxiter=50)
    except: return np.nan

def compute_greeks(spot, strike, opt_type, ts, expiry, opt_price) -> dict:
    expiry_dt = expiry_as_datetime(expiry).replace(hour=15, minute=30)
    T = (expiry_dt - ts).total_seconds() / (365*24*3600)
    r, q = CFG.risk_free_rate, CFG.dividend
    iv = calc_iv(opt_price, spot, strike, T, r, q, opt_type)
    if np.isnan(iv) or iv <= 0: return {g: np.nan for g in GREEK_COLS}
    d1, d2 = _d1d2(spot, strike, T, r, q, iv)
    Npd1  = norm.pdf(d1)
    gamma = np.exp(-q*T)*Npd1 / (spot*iv*np.sqrt(T))
    vega  = spot*np.exp(-q*T)*Npd1*np.sqrt(T) / 100
    if opt_type == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-np.exp(-q*T)*spot*Npd1*iv/(2*np.sqrt(T)) + q*spot*np.exp(-q*T)*norm.cdf(d1)
                 - r*strike*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        delta = np.exp(-q*T)*(norm.cdf(d1)-1)
        theta = (-np.exp(-q*T)*spot*Npd1*iv/(2*np.sqrt(T)) - q*spot*np.exp(-q*T)*norm.cdf(-d1)
                 + r*strike*np.exp(-r*T)*norm.cdf(-d2)) / 365
    if T < 1/365:
        intrinsic = max(0., spot-strike) if opt_type=="call" else max(0., strike-spot)
        theta = -min(abs(theta), max(0., opt_price-intrinsic))
    return {"iv": iv, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def greeks_pass(g: dict) -> bool:
    for name, bounds in GREEK_THRESHOLDS.items():
        v = g.get(name)
        if v is None or np.isnan(v): return False
        if bounds["min"] is not None and v < bounds["min"]: return False
        if bounds["max"] is not None and v > bounds["max"]: return False
    return True

# ── Lazy option stream ────────────────────────────────────────────────────────
class LazyOptionStream:
    def __init__(self, expiry, strike, right, entry_ts, hard_end, lead_secs=None, chunk_rows=None):
        self.expiry = expiry; self.strike = strike; self.right = right; self.hard_end = hard_end
        self.chunk_rows  = chunk_rows or CFG.opt_chunk_rows
        self._fetched_to = entry_ts - timedelta(seconds=(lead_secs or CFG.opt_lead_secs))
        self._df: Optional[pd.DataFrame] = None
        self._fetch_next_chunk()

    def _fetch_raw(self, from_dt, to_dt) -> pd.DataFrame:
        resp = get_breeze().get_historical_data_v2(
            interval="1second", from_date=from_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            to_date=to_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            stock_code="NIFTY", exchange_code="NFO", product_type="options",
            expiry_date=self.expiry, right=self.right, strike_price=self.strike)
        if not resp or not resp.get("Success"): raise ValueError(f"No data {self.right} {self.strike}")
        df = pd.DataFrame(resp["Success"])
        df["datetime"] = pd.to_datetime(df["datetime"]); df.set_index("datetime", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        for col in ["open", "high", "low", "close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _fetch_next_chunk(self) -> bool:
        if self._fetched_to >= self.hard_end: return False
        chunk_end = min(self._fetched_to + timedelta(seconds=self.chunk_rows), self.hard_end)
        try: chunk = self._fetch_raw(self._fetched_to, chunk_end)
        except Exception as e:
            log.warning(f"  chunk {self.right} {self.strike}: {e}"); self._fetched_to = chunk_end; return False
        self._fetched_to = chunk_end
        if chunk.empty: return False
        full_idx = pd.date_range(start=chunk.index[0], end=chunk.index[-1], freq="1s")
        chunk = chunk.reindex(full_idx).ffill()
        self._df = chunk if self._df is None else pd.concat([self._df, chunk[~chunk.index.isin(self._df.index)]])
        log.info(f"  [chunk] {self.right.upper()} {self.strike} +{len(chunk)} → {self._fetched_to:%H:%M:%S}")
        return len(chunk) > 0

    def price_at(self, ts: datetime, col: str = "close") -> Optional[float]:
        while ts > self._fetched_to and self._fetched_to < self.hard_end: self._fetch_next_chunk()
        if self._df is None: return None
        if ts in self._df.index:
            v = self._df.loc[ts, col]; return float(v) if pd.notna(v) else None
        pos = self._df.index.searchsorted(ts, side="right") - 1
        if pos >= 0:
            v = self._df.iloc[pos][col]; return float(v) if pd.notna(v) else None
        return None

    def iter_rows_from(self, start_ts: datetime):
        if self._df is None: return
        pos = self._df.index.searchsorted(start_ts, side="left")
        while True:
            while pos < len(self._df):
                ts = self._df.index[pos]; row = self._df.iloc[pos]; pos += 1
                if ts > self.hard_end: return
                yield ts, row
            if self._fetched_to >= self.hard_end: return
            old_len = len(self._df)
            if not self._fetch_next_chunk() or len(self._df) == old_len: return

# ── Exit loop ─────────────────────────────────────────────────────────────────
def run_exit_loop(stream, entry_ts, entry_premium, exit_cutoff, rpt) -> dict:
    sl_dist = entry_premium * CFG.sl_pct_of_premium
    trail_sl = initial_sl = entry_premium - sl_dist
    peak_price = entry_premium; max_profit = 0.
    last_ts = entry_ts; last_close = entry_premium
    exit_time = exit_price = exit_type = None
    trade_max_dd = 0.
    for ts, row in stream.iter_rows_from(entry_ts + timedelta(seconds=1)):
        c = float(row.get("close", entry_premium)); last_ts = ts; last_close = c
        if c > peak_price: peak_price = c; max_profit = peak_price - entry_premium;trail_sl = peak_price - sl_dist
        dd = (peak_price - c) / peak_price * 100
        trade_max_dd = max(trade_max_dd, dd)
        rpt.track_unrealized(c, entry_premium)
        if c <= trail_sl:
            exit_price, exit_time = c, ts
            exit_type = "Initial SL" if max_profit == 0. else "Trailing SL"
            log.info(f"  {exit_type} @ {ts.time()} sl={trail_sl:.2f} PnL=₹{(c-entry_premium)*CFG.lot_size:.2f}")
            break
        if ts >= exit_cutoff:
            exit_price, exit_time, exit_type = c, ts, "EOD"
            log.info(f"  EOD @ {ts.time()} px={c:.2f}"); break

    if exit_time is None: exit_price, exit_time, exit_type = last_close, last_ts, "EOD"
    return {"exit_time": exit_time, "exit_price": exit_price, "exit_type": exit_type,
            "max_profit": max_profit, "trade_max_dd": trade_max_dd, "trade_pnl": (exit_price - entry_premium) * CFG.lot_size,
            "initial_sl": initial_sl, "final_trail_sl": trail_sl}

# ── CSV ───────────────────────────────────────────────────────────────────────
_CSV_HEADER = [
    "Date", "Trade_Num_Today", "Strike", "Side", "Entry_Time", "Entry_Premium",
    "Exit_Time", "Exit_Premium", "Max_Profit", "Trade_Max_DD_Pct","PnL_Per_Lot", "Quantity",
    "Total_PnL", "Exit_Type", "Initial_SL", "Final_Trail_SL",
    "Cumulative_PnL", "Capital", "Return_Pct",
    *[f"entry_{g}" for g in GREEK_COLS], *[f"exit_{g}" for g in GREEK_COLS],
    *[f"entry_{c}" for c in INDICATOR_COLS], *[f"exit_{c}" for c in INDICATOR_COLS],
]
_fmt = lambda v: "NA" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

def _init_csv():
    try:
        with open(CFG.csv_file, "x", newline="") as f: csv.writer(f).writerow(_CSV_HEADER)
    except FileExistsError: pass

def log_csv(trade_date, trade_num,strike, side, entry_ts, entry_prem, entry_g,
            exit_ts, exit_prem, exit_g, max_profit, trade_max_dd,pnl_lot, qty,
            total_pnl, exit_type, isl, fsl, cum_pnl, ei, xi):
    eg = entry_g or {}; xg = exit_g or {}
    row = [trade_date.strftime("%Y-%m-%d"), trade_num,strike, side,
           entry_ts.strftime("%H:%M:%S"), _fmt(entry_prem),
           exit_ts.strftime("%H:%M:%S"), _fmt(exit_prem),
           _fmt(max_profit), _fmt(trade_max_dd),_fmt(pnl_lot), qty, _fmt(total_pnl), exit_type,
           _fmt(isl), _fmt(fsl), _fmt(cum_pnl),
           _fmt(CFG.capital + cum_pnl), _fmt(cum_pnl / CFG.capital * 100),
           *[_fmt(eg.get(g)) for g in GREEK_COLS], *[_fmt(xg.get(g)) for g in GREEK_COLS],
           *[_fmt(ei.get(c)) for c in INDICATOR_COLS], *[_fmt(xi.get(c)) for c in INDICATOR_COLS]]
    with open(CFG.csv_file, "a", newline="") as f: csv.writer(f).writerow(row)

# ── Report ────────────────────────────────────────────────────────────────────
@dataclass
class Report:
    cumulative_pnl: float = 0.
    peak_equity: float = 0.
    max_drawdown: float = 0.
    total: int = 0
    wins: int = 0
    losses: int = 0
    win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # (date, eod_equity)

    SECONDS_PER_YEAR: int = 252 *6.5*3600  # keep for reference, not used in EOD calc

    def record_eod(self, trade_date: datetime):
        """Call once at end of every trading day — trade or no trade."""
        eq = CFG.capital + self.cumulative_pnl
        self._track_equity(eq)
        self.equity_curve.append((trade_date.date(), eq))
    def _cagr(self) -> float:
        if len(self.equity_curve) < 2: return np.nan
        start_eq = self.equity_curve[0][1]
        end_eq   = self.equity_curve[-1][1]
        start_dt = pd.Timestamp(self.equity_curve[0][0])
        end_dt   = pd.Timestamp(self.equity_curve[-1][0])
        years    = (end_dt - start_dt).days / 365.25
        if years <= 0 or start_eq <= 0: return np.nan
        return (end_eq / start_eq) ** (1 / years) - 1
    def _track_equity(self, eq):
        if eq > self.peak_equity: self.peak_equity = eq
        if self.peak_equity > 0:
            dd = (self.peak_equity - eq) / self.peak_equity * 100
            if dd > self.max_drawdown: self.max_drawdown = dd

    def update(self, pnl):
        net = pnl - CFG.cost_per_trade
        self.cumulative_pnl += net
        self.total += 1
        if net >= 0: self.wins += 1;  self.win_pnls.append(net)
        else:        self.losses += 1; self.loss_pnls.append(net)
        # NOTE: _track_equity called in record_eod, not here

    def track_unrealized(self, price, entry_prem):
        # still used inside exit loop for intraday DD tracking only
        eq = CFG.capital + self.cumulative_pnl + (price - entry_prem) * CFG.lot_size
        self._track_equity(eq)

    def _sharpe_and_vol(self):
        if len(self.equity_curve) < 2:
            return np.nan, np.nan
        equities = np.array([e for _, e in self.equity_curve])
        daily_rets = np.diff(equities) / equities[:-1]   # day-over-day returns
        std  = np.std(daily_rets, ddof=1)
        if std == 0: return np.nan, np.nan
        vol    = std * np.sqrt(252)                        # annualised vol
        mean_r = np.mean(daily_rets)
        rf_daily = CFG.risk_free_rate / 252
        sharpe = (mean_r - rf_daily) / std * np.sqrt(252) # annualised Sharpe
        return sharpe, vol

    def summary(self):
        sharpe, vol = self._sharpe_and_vol()
        cagr = self._cagr()
        avg_win  = np.mean(self.win_pnls)  if self.win_pnls  else 0.
        avg_loss = np.mean(self.loss_pnls) if self.loss_pnls else 0.
        avg_pnl  = np.mean(self.win_pnls + self.loss_pnls) if self.total else 0.
        win_rate = self.wins / self.total * 100 if self.total else 0.
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
_vix_df = pd.read_csv("vix_daily_3yr.csv", index_col=0, parse_dates=True)
_vix_df.index = pd.to_datetime(_vix_df.index).normalize()

def get_opening_vix(trade_date: datetime) -> Optional[float]:
    key = pd.Timestamp(trade_date.date())
    if key in _vix_df.index:
        val = _vix_df.loc[key].iloc[0]  # first column = open VIX
        return float(val) if pd.notna(val) else None
    return None
# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest():
    rpt = Report(peak_equity=CFG.capital)
    _init_csv()
    log.info("Loading futures CSV and computing indicators …")
    df_all = compute_indicators(load_fut_csv())
    log.info(f"Indicators ready. Shape={df_all.shape}")

    current_date = CFG.start_date
    while current_date <= CFG.end_date:
        if current_date.weekday() >= 5: current_date += timedelta(days=1); continue
        # ── Day-of-week filter ────────────────────────────────────────────────
        if current_date.weekday() not in CFG.trade_days:
            log.info(f"[SKIP-DAY] {current_date.strftime('%A %Y-%m-%d')} not in trade_days")
            rpt.record_eod(current_date)
            current_date += timedelta(days=1)
            continue
        
        # # ── VIX filter ────────────────────────────────────────────────────────
        # vix_open = get_opening_vix(current_date)
        # if vix_open is None:
        #     log.warning(f"[SKIP-VIX] {current_date.date()} — VIX data missing")
        #     rpt.record_eod(current_date)
        #     current_date += timedelta(days=1)
        #     continue
        # if not (CFG.vix_min <= vix_open <= CFG.vix_max):
        #     log.info(f"[SKIP-VIX] {current_date.date()} VIX={vix_open:.2f} out of range [{CFG.vix_min}, {CFG.vix_max}]")
        #     rpt.record_eod(current_date)
        #     current_date += timedelta(days=1)
        #     continue
        # log.info(f"[VIX-OK] {current_date.date()} VIX={vix_open:.2f}")
        
        entry_start = datetime.combine(current_date, datetime.strptime(CFG.entry_start, "%H:%M:%S").time())
        exit_cutoff = datetime.combine(current_date, datetime.strptime(CFG.exit_cutoff,  "%H:%M:%S").time())
        expiry      = get_expiry_for_date(current_date)
        expiry_dt   = expiry_as_datetime(expiry).replace(hour=15, minute=30)

        df_day = df_all[df_all.index.date == current_date.date()]
        if df_day.empty:
            log.warning(f"[SKIP] {current_date.date()} — no data"); current_date += timedelta(days=1); continue

        log.info(f"\n{'='*52}\n{current_date.date()}  expiry={expiry[:10]}\n{'='*52}")
        scan_bars = [ts for ts in df_day.index if entry_start <= ts < exit_cutoff]
        daily_trade_num = 0
        i = 0
        while i < len(scan_bars):
            candle_ts = scan_bars[i]
            if not check_entry_signal(df_all, candle_ts): i += 1; continue

            entry_ts = candle_ts + timedelta(minutes=1)
            if entry_ts > exit_cutoff: i += 1; continue

            spot = fetch_spot(entry_ts)
            if spot is None: log.warning(f"  No spot @ {entry_ts.time()}"); i += 1; continue

            atm_strike = int(round(spot / CFG.strike_step)) * CFG.strike_step
            opt_type   = "call"

            try:
                stream = LazyOptionStream(expiry=expiry, strike=atm_strike, right=opt_type,
                                          entry_ts=entry_ts, hard_end=exit_cutoff)
            except Exception as e:
                log.error(f"  Stream init: {e}"); i += 1; continue

            entry_prem = stream.price_at(entry_ts)
            if not entry_prem or entry_prem <= 0:
                log.warning(f"  Bad entry premium {entry_prem}"); i += 1; continue

            entry_g = compute_greeks(spot, atm_strike, opt_type, entry_ts, expiry, entry_prem)
            iv = entry_g.get("iv", np.nan)
            # if np.isnan(iv) or iv < CFG.iv_min:
            #     log.info(f"  [SKIP-IV] IV={iv:.4f}"); i += 1; continue
            if not greeks_pass(entry_g):
                log.info(f"  [SKIP-GREEKS] {entry_g}"); i += 1; continue

            log.info(f"  ENTRY CALL K={atm_strike} px={entry_prem:.2f} IV={iv:.2%} Δ={entry_g.get('delta',0):.3f}")
            entry_indic = indic_snapshot(df_all, candle_ts)

            res = run_exit_loop(stream, entry_ts, entry_prem, exit_cutoff, rpt)

            exit_g = {}
            try:
                ex_spot = fetch_spot(res["exit_time"])
                if ex_spot is None:
                    idx2 = df_all.index.searchsorted(res["exit_time"], side="right") - 1
                    ex_spot = float(df_all.iloc[idx2]["close"]) if idx2 >= 0 else spot
                exit_g = compute_greeks(ex_spot, atm_strike, opt_type, res["exit_time"], expiry, res["exit_price"])
            except Exception as e: log.warning(f"  Exit greeks: {e}")

            exit_indic = indic_snapshot(df_all, res["exit_time"].replace(second=0, microsecond=0))
            rpt.update(res["trade_pnl"])
            pnl_lot   = res["trade_pnl"] / CFG.lot_size
            total_pnl = res["trade_pnl"] - CFG.cost_per_trade
            daily_trade_num += 1

            log_csv(current_date, daily_trade_num, atm_strike, opt_type, entry_ts, entry_prem, entry_g,
                    res["exit_time"], res["exit_price"], exit_g, res["max_profit"], res["trade_max_dd"],
                    pnl_lot, CFG.lot_size, total_pnl, res["exit_type"],
                    res["initial_sl"], res["final_trail_sl"], rpt.cumulative_pnl, entry_indic, exit_indic)

            log.info(f"  [DONE] exit={res['exit_time'].time()} PnL=₹{total_pnl:.2f}  CumPnL=₹{rpt.cumulative_pnl:.2f}")
            if daily_trade_num >= 3:                    # ← ADD
                log.info(f"  [MAX TRADES] 3 trades done for {current_date.date()}, stopping day")
                break    
            i = next((j for j, t in enumerate(scan_bars) if t > res["exit_time"]), len(scan_bars))
        rpt.record_eod(current_date)
        current_date += timedelta(days=1)
    
    rpt.summary()
    rpt.save_equity_curve()
    return rpt

if __name__ == "__main__":
    run_backtest()

