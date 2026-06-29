from breeze_connect import BreezeConnect
import csv, time, math
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import json
from scipy.stats import norm
from scipy.optimize import brentq

# --- BreezeConnect Setup mohit ---
breeze = BreezeConnect(api_key="=qw3v81645C94339h387K4461_520l05")
breeze.generate_session(api_secret="1h87H%27q23626t448M55J5605P532y5", session_token="56119530")

# # --- BreezeConnect SS ---
# breeze = BreezeConnect(api_key="9581N(586t6E2A047i2p5(730@yO5t74")
# breeze.generate_session(api_secret="9C3n144h3(5BA4f210074821g7V5+333", session_token="55419947")

# # --- BreezeConnect RS13 ---
# breeze = BreezeConnect(api_key="0Y44w$7Mt280%6668X837637^oq0311Y")
# breeze.generate_session(api_secret="A67598Fg3M471924ie8`08K23925a765", session_token="55701404")

# --- Config ---

start_date = datetime(2025, 11, 21)
end_date   = datetime(2026, 6, 14)
symbol          = "CNXBAN"
exchange        = "NSE"
option_exchange = "NFO"

entry_time_str = "10:15:00"
exit_time_str  = "15:00:00"

lot_size   = 60
partialqty = 30
ROUND_OFF  = 100

# Output files
csv_file = "Straddle_RAS_4+Delta_35_New_1.csv"

# Greeks / volatility config
RISK_FREE_RATE = 0.07
DIVIDEND       = 0.014
W_DELTA        = 0.2
W_THETA        = 0.3
W_VEGA         = 0.5

# ---------------------------------------------------------------------------
# Delta Config
# ---------------------------------------------------------------------------
DELTA_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# RAS Config  (function-based, no class)
# ---------------------------------------------------------------------------
RAS_1MIN_LOOKBACK   = 5          
RAS_1MIN_NORM       = 20         
RAS_1SEC_LOOKBACK   = 5 * 60     
RAS_1SEC_NORM       = 20 * 60    

RAS_WEIGHT_DELTA    = 0.2
RAS_WEIGHT_GAMMA    = 0.3
RAS_WEIGHT_VEGA     = 0.5

RAS_ENTRY_LIMIT     = 2.0        
RAS_KILL_SWITCH     = 4.0        

# ---------------------------------------------------------------------------
# Load weekly expiries
# ---------------------------------------------------------------------------
with open("expiries_nifty_Month.json", "r") as f:
    expiry_data = json.load(f)
expiry_list = [datetime.strptime(d, "%Y-%m-%d") for d in expiry_data["Nifty"]]
expiry_list.sort()

def get_expiry_for_date(trade_date):
    for expiry_date in expiry_list:
        if expiry_date >= trade_date:
            return expiry_date.strftime("%Y-%m-%dT00:00:00.000Z")
    return expiry_list[-1].strftime("%Y-%m-%dT00:00:00.000Z")

# ---------------------------------------------------------------------------
# ATM / entry filter
# ---------------------------------------------------------------------------
def calculate_atm(df_nifty, check_time):
    last_10 = df_nifty[df_nifty.index <= check_time].tail(15)
    if len(last_10) < 15:
        return None, None
    if (last_10.iloc[-15]['ATR'] > last_10.iloc[-5]['ATR']) and \
       (last_10.iloc[-15]['ADX'] > last_10.iloc[-5]['ADX']) and \
       (last_10.iloc[-15]['RV']  > last_10.iloc[-5]['RV']):
        return last_10.iloc[-1]['close'], last_10.iloc[-1]['RV']
    return None, None

def round_to_nearest_50(x):
    return int(round(x / 50.0)) * 50

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_1min_historical(symbol, from_date, to_date, exchange="NSE"):
    return breeze.get_historical_data_v2(
        interval="1minute",
        from_date=from_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        to_date=to_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        stock_code=symbol,
        exchange_code=exchange,
        product_type="cash"
    )

def get_monthly_historical():
    return breeze.get_historical_data_v2(
        interval="1day",
        from_date="2025-06-01T09:21:00.000Z",
        to_date="2026-05-30T09:21:00.000Z",
        stock_code="CNXBAN",
        exchange_code="NSE",
        product_type="cash"
    )

def generate_master_index(from_date, to_date):
    from_date = from_date.replace(microsecond=0)
    to_date   = to_date.replace(microsecond=0)
    return pd.date_range(start=from_date, end=to_date, freq="1s")

def get_option_df_complete(expiry, strike, right, from_date, to_date):
    step_seconds  = 1000
    current_start = from_date
    all_data      = []

    while current_start < to_date:
        current_end = min(current_start + timedelta(seconds=step_seconds), to_date)
        try:
            data = breeze.get_historical_data_v2(
                interval="1second",
                from_date=current_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                to_date=current_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                stock_code=symbol,
                exchange_code=option_exchange,
                product_type="options",
                expiry_date=expiry,
                right=right,
                strike_price=strike
            )
            if "Success" in data and data["Success"]:
                all_data.extend(data["Success"])
        except Exception as e:
            pass

        current_start = current_end
        time.sleep(0.2)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    master_index = generate_master_index(from_date, to_date)
    df = df.reindex(master_index).ffill()
    return df

def get_option_1min_historical(expiry, strike, right, from_date, to_date):
    try:
        data = breeze.get_historical_data_v2(
            interval="1minute",
            from_date=from_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            to_date=to_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            stock_code=symbol,
            exchange_code=option_exchange,
            product_type="options",
            expiry_date=expiry,
            right=right,
            strike_price=strike
        )
        if "Success" not in data or not data["Success"]:
            return None

        df = pd.DataFrame(data["Success"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        min_idx = pd.date_range(start=from_date.replace(second=0, microsecond=0),
                                end=to_date.replace(second=0, microsecond=0),
                                freq="1min")
        df = df.reindex(min_idx).ffill()
        return df

    except Exception:
        return None

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------
def merton_price(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0: return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def calculate_iv_merton(option_price, S, K, T, r, q, option_type):
    try:
        if T <= 0 or option_price <= 0: return 0.001
        objective_func = lambda sigma: merton_price(S, K, T, r, q, sigma, option_type) - option_price
        return brentq(objective_func, 1e-4, 5.0) 
    except ValueError:
        return np.nan

def merton_greeks(entry,S, K, T, sigma,r, q, option_type="call"):
    if T <= 0 or sigma <= 0 or np.isnan(sigma): return 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_prime_d1 = norm.pdf(d1)
    gamma = (np.exp(-q * T) * N_prime_d1) / (S * sigma * np.sqrt(T))
    vega = (S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T)) / 100
    vomma = vega * (d1 * d2) / sigma if sigma != 0 else 0
    if option_type == "call":
        delta = np.exp(-q * T) * N_d1
        theta = (-np.exp(-q * T) * S * N_prime_d1 * sigma / (2 * np.sqrt(T))) \
                + (q * S * np.exp(-q * T) * N_d1) \
                - (r * K * np.exp(-r * T) * norm.cdf(d2))
        theta= theta/365
    else:  
        delta = np.exp(-q * T) * (N_d1 - 1)
        theta = (-np.exp(-q * T) * S * N_prime_d1 * sigma / (2 * np.sqrt(T))) \
                - (q * S * np.exp(-q * T) * norm.cdf(-d1)) \
                + (r * K * np.exp(-r * T) * norm.cdf(-d2))
        theta= theta/365
        
    intrinsic= max(0, S - K)
    extrinsic = max(0, entry - intrinsic)

    # Cap daily Theta at the remaining extrinsic value on expiry day (< 1 Day)
    if T < (1 / 365):
        theta = -min(abs(theta), extrinsic)

    return delta, theta, vega, gamma, vomma


def calculate_all_greeks(entry,spot, strike, T, sigma, option_type):
    return merton_greeks(entry,spot, strike, T, sigma, RISK_FREE_RATE, DIVIDEND, option_type)


# Greek

def calculate_greeks_sl(entry, spot, strike, T, sigma, atr, option_type):
    delta, theta, vega, gamma, vomma = merton_greeks(
        entry,spot, strike, T, sigma, RISK_FREE_RATE, DIVIDEND, option_type
    )
    tag = "CE" if option_type == "call" else "PE"
    print(f"  Delta {tag}: {delta:.4f} | Theta {tag}: {theta:.4f} | Vega {tag}: {vega:.4f} |Gamma {tag}: {gamma:.4f} |Vomma {tag}: {vomma:.4f}")

    total_minutes = 369
    minutes_left  = (exit_time.hour * 60 + exit_time.minute) - \
                    (entry_time.hour * 60 + entry_time.minute)
    time_factor   = minutes_left / total_minutes if minutes_left > 0 else 0.01
    vol_factor    = atr / spot

    sl_offset = (W_DELTA * abs(delta) * atr +
                 W_THETA * abs(theta) * time_factor +
                 W_VEGA  * vega * vol_factor)
    sl = max(15,min(sl_offset * 6, 25))
    return sl, sl_offset, delta, theta, vega


def build_greeks_snapshot(spot, strike, tte, ce_price, pe_price):
    iv_ce = calculate_iv_merton(ce_price, spot, strike, tte, RISK_FREE_RATE, DIVIDEND, "call") 
    iv_pe = calculate_iv_merton(pe_price, spot, strike, tte, RISK_FREE_RATE, DIVIDEND, "put") 

    d_ce, t_ce, v_ce, g_ce, vomma_ce = calculate_all_greeks(ce_price, spot, strike, tte, iv_ce, "call")
    d_pe, t_pe, v_pe, g_pe, vomma_pe = calculate_all_greeks(pe_price, spot, strike, tte, iv_pe, "put")

    return {
        "net_delta": d_ce + d_pe,
        "net_gamma": g_ce + g_pe,
        "net_vega": v_ce + v_pe,
        "net_vomma": vomma_ce + vomma_pe,
        "d_ce": d_ce, "d_pe": d_pe,
        "g_ce": g_ce, "g_pe": g_pe,
        "v_ce": v_ce, "v_pe": v_pe,
        "vomma_ce": vomma_ce, "vomma_pe": vomma_pe,
        "iv_ce": iv_ce, "iv_pe": iv_pe,
    }


def compute_ras(greeks_history, lookback_bars, norm_window_bars):
    history = greeks_history[-norm_window_bars:]
    if len(history) < lookback_bars + 1: return 0.0

    arr_d = np.array([g["net_delta"] for g in history])
    arr_g = np.array([g["net_gamma"] for g in history])
    arr_v = np.array([g["net_vega"]  for g in history])

    delta_accel = abs(arr_d[-1] - arr_d[-lookback_bars - 1])
    gamma_accel = abs(arr_g[-1] - arr_g[-lookback_bars - 1])
    vega_accel  = abs(arr_v[-1] - arr_v[-lookback_bars - 1])

    std_d = np.std(arr_d) if np.std(arr_d) > 0 else 1e-6
    std_g = np.std(arr_g) if np.std(arr_g) > 0 else 1e-6
    std_v = np.std(arr_v) if np.std(arr_v) > 0 else 1e-6

    ras = (RAS_WEIGHT_DELTA * (delta_accel / std_d) +
           RAS_WEIGHT_GAMMA * (gamma_accel / std_g) +
           RAS_WEIGHT_VEGA  * (vega_accel  / std_v))
    return ras

def ras_regime_label(ras):
    if ras < 2.0: return "COMPRESSION"
    if ras < 4.0: return "TRANSITION"
    if ras < 6.0: return "EXPANSION"
    return "TAIL_RISK"

def build_preentry_greeks_history(expiry, strike, check_time, expiry_dt, df_nifty_1min, lookback_min=25):
    hist_start = check_time - timedelta(minutes=lookback_min)
    df_ce_1m = get_option_1min_historical(expiry, strike, "call", hist_start, check_time)
    df_pe_1m = get_option_1min_historical(expiry, strike, "put",  hist_start, check_time)

    if df_ce_1m is None or df_pe_1m is None: return []

    greeks_history = []
    bars = df_ce_1m.index.intersection(df_pe_1m.index)

    for ts in bars:
        ce_price = df_ce_1m.loc[ts, "close"]
        pe_price = df_pe_1m.loc[ts, "close"]
        spot_ts = df_nifty_1min.index.asof(ts)
        if pd.isnull(spot_ts): continue
        spot = float(df_nifty_1min.loc[spot_ts, "close"])

        tte = (expiry_dt - ts) / timedelta(days=1) / 365
        if tte <= 0: continue

        snap = build_greeks_snapshot(spot, strike, tte, ce_price, pe_price)
        greeks_history.append(snap)

    return greeks_history

# ---------------------------------------------------------------------------
# Premium-matching & GAP
# ---------------------------------------------------------------------------
def find_balanced_strike(expiry, atm, from_date, to_date, check_time):
    candidates = [atm - ROUND_OFF, atm, atm + ROUND_OFF]
    results    = []

    for strike in candidates:
        try:
            window_start = check_time - timedelta(seconds=10)
            dc = get_option_df_complete(expiry, strike, "call", window_start, check_time)
            time.sleep(0.3)
            dp = get_option_df_complete(expiry, strike, "put",  window_start, check_time)

            ce_ltp = dc["close"].dropna().iloc[-1] if dc is not None and not dc["close"].dropna().empty else None
            pe_ltp = dp["close"].dropna().iloc[-1] if dp is not None and not dp["close"].dropna().empty else None

            if ce_ltp and pe_ltp:
                diff      = abs(ce_ltp - pe_ltp)
                threshold = 0.20 * max(ce_ltp, pe_ltp)
                results.append((diff, strike, ce_ltp, pe_ltp, threshold))
        except Exception:
            pass

    valid = [(d, s, c, p) for (d, s, c, p, thr) in results if d <= thr]
    if not valid: return atm, None, None

    best = min(valid, key=lambda x: x[0])
    return best[1], best[2], best[3]

def check_gap_condition(df_nifty_daily, current_date):
    et     = current_date.strftime("%Y-%m-%d")
    last_3 = df_nifty_daily[df_nifty_daily.index <= et].tail(3)
    if len(last_3) < 2: return False
    gap = last_3.iloc[-1]['GAP']
    return abs(gap) <= 1.25

# ---------------------------------------------------------------------------
# CSV logging (Updated for Vomma)
# ---------------------------------------------------------------------------
def log_to_csv(file_name, date, atm_strike, expiry,
               entry_time, entry_call, entry_put, entry_premium, SG_ce, SG_pe,
               sl, exit_time, exit_call, exit_put, exit_premium,
               et, pb, pnl,
               entry_ras=None, exit_ras=None, kill_switch=False,
               entry_delta=None, exit_delta=None):
    try:
        with open(file_name, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Date", "ATM Strike", "Expiry Used",
                "Entry Time", "Entry Call", "Entry Put", "Entry Premium",
                "SL_CE_G", "SL_PE_G", "SL",
                "Exit Time", "Exit Call", "Exit Put", "Exit Premium",
                "Exit Type", "Partial Booking", "PnL",
                "Entry_RAS", "Exit_RAS", "RAS_KillSwitch",
                "Entry_Delta", "Exit_Delta" 
            ])
    except FileExistsError:
        pass

    def fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        return f"{v:.2f}" if v is not None else "NA"

    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            date.strftime("%Y-%m-%d"),
            atm_strike if atm_strike is not None else "NA",
            expiry     if expiry     is not None else "NA",
            entry_time.strftime("%H:%M:%S") if entry_time is not None else "NA",
            fmt(entry_call), fmt(entry_put), fmt(entry_premium),
            fmt(SG_ce), fmt(SG_pe), fmt(sl),
            exit_time.strftime("%H:%M:%S") if exit_time is not None else "NA",
            fmt(exit_call), fmt(exit_put), fmt(exit_premium), et,
            fmt(pb), fmt(pnl),
            f"{entry_ras:.4f}" if entry_ras is not None else "NA",
            f"{exit_ras:.4f}"  if exit_ras  is not None else "NA",
            "YES" if kill_switch else "NO",
            f"{entry_delta:.4f}" if entry_delta is not None else "NA", 
            f"{exit_delta:.4f}" if exit_delta is not None else "NA"   
        ])

# ---------------------------------------------------------------------------
# Pre-load daily CNXBAN data for GAP calculation
# ---------------------------------------------------------------------------
df_nifty_daily = get_monthly_historical()
df_nifty_daily = pd.DataFrame(df_nifty_daily["Success"])
df_nifty_daily['ATR']      = ta.atr(df_nifty_daily['high'], df_nifty_daily['low'], df_nifty_daily['close'])
df_nifty_daily['datetime'] = pd.to_datetime(df_nifty_daily['datetime'])
df_nifty_daily.set_index('datetime', inplace=True)
df_nifty_daily['10_Day_Avg']    = df_nifty_daily['ATR'].rolling(window=10).mean()
df_nifty_daily['Change_In_Atr'] = df_nifty_daily['ATR'] - df_nifty_daily['10_Day_Avg']
df_nifty_daily.dropna(subset=['Change_In_Atr'], inplace=True)
df_nifty_daily['GAP'] = ((df_nifty_daily['open'] - df_nifty_daily['close'].shift(1)) /
                          df_nifty_daily['close'].shift(1)) * 100

# ===========================================================================
# MAIN BACKTEST LOOP
# ===========================================================================
current_date = start_date

while current_date <= end_date:
    print(f"\n{'='*60}")
    print(f"  DATE : {current_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    try:
        Trade = AVG = Loss_Trade = Profit_Trade = Max_Loss = Max_Profit = 0
        expiry     = get_expiry_for_date(current_date)
        
        # ══════════════════════════════════════════════════════════

        entry_time = datetime.combine(current_date, datetime.strptime(entry_time_str, "%H:%M:%S").time())
        exit_time  = datetime.combine(current_date, datetime.strptime(exit_time_str,  "%H:%M:%S").time())

        nifty_data = get_1min_historical(symbol, entry_time.replace(hour=9, minute=0), exit_time)
        df_nifty   = pd.DataFrame(nifty_data["Success"])
        if df_nifty.empty: raise ValueError("Holiday or CNXBAN data not available")

        df_nifty['datetime'] = pd.to_datetime(df_nifty['datetime'])
        df_nifty.set_index('datetime', inplace=True)
        df_nifty['ADX']     = ta.adx(df_nifty['high'], df_nifty['low'], df_nifty['close'])['ADX_14']
        df_nifty['ATR']     = ta.atr(df_nifty['high'], df_nifty['low'], df_nifty['close'])
        df_nifty['log_ret'] = np.log(df_nifty['close'] / df_nifty['close'].shift(1))
        df_nifty['RV']      = df_nifty['log_ret'].rolling(window=10).std() * np.sqrt(252 * 390)

        df_nifty_1min = df_nifty.copy()
        daily_atr_condition = check_gap_condition(df_nifty_daily, current_date)
        current_time = entry_time

        while current_time <= exit_time:
            atm_price_raw, RV = calculate_atm(df_nifty, current_time)

            if atm_price_raw is not None and daily_atr_condition and RV is not None and Trade <= 7:
                atm_price      = float(atm_price_raw)
                atm_strike_raw = round_to_nearest_50(atm_price)

                atm_strike, ce_ltp_preview, pe_ltp_preview = find_balanced_strike(
                    expiry, atm_strike_raw,
                    current_time - timedelta(minutes=2), exit_time, current_time
                )
                
                expiry_dt = datetime.strptime(expiry[:10], "%Y-%m-%d") + timedelta(hours=15, minutes=30)

                preentry_greeks = build_preentry_greeks_history(
                    expiry, atm_strike, current_time, expiry_dt,
                    df_nifty_1min, lookback_min=25
                )

                entry_ras = compute_ras(preentry_greeks, RAS_1MIN_LOOKBACK, RAS_1MIN_NORM)

                if entry_ras > RAS_ENTRY_LIMIT:
                    current_time += timedelta(minutes=1)
                    continue

                df_ce = get_option_df_complete(expiry, atm_strike, "call", current_time - timedelta(minutes=2), exit_time)
                df_pe = get_option_df_complete(expiry, atm_strike, "put", current_time - timedelta(minutes=2), exit_time)

                if df_ce is None or df_pe is None:
                    current_time += timedelta(minutes=1)
                    continue

                df_ce['ATR'] = ta.atr(df_ce['high'], df_ce['low'], df_ce['close'], length=14)
                df_pe['ATR'] = ta.atr(df_pe['high'], df_pe['low'], df_pe['close'], length=14)

                master_index = generate_master_index(entry_time - timedelta(minutes=2), exit_time)
                df_nifty_sec = df_nifty.reindex(master_index).ffill()

                df_combined = pd.DataFrame({
                    "ce":   df_ce["close"],
                    "pe":   df_pe["close"],
                    "spot": df_nifty_sec["close"],
                })
                df_combined["total"] = df_combined["ce"] + df_combined["pe"]

                if current_time not in df_combined.index:
                    current_time += timedelta(seconds=1)
                    continue

                entry_row     = df_combined.loc[current_time]
                entry_premium = entry_row["total"]
                ce_entry      = entry_row["ce"]
                pe_entry      = entry_row["pe"]

                if entry_premium < 50:
                    current_time += timedelta(minutes=1)
                    continue

                entry_time_actual = current_time
                Entry_CE          = ce_entry
                Entry_PE          = pe_entry
                Trade            += 1

                print(f"ENTRY | CE: {ce_entry:.2f}  PE: {pe_entry:.2f}  Total: {entry_premium:.2f}")

                time_to_expiry = (expiry_dt - current_time) / timedelta(days=1) / 365
                atr_ce         = df_ce.loc[current_time]['ATR']
                atr_pe         = df_pe.loc[current_time]['ATR']

                volatility_ce = calculate_iv_merton(ce_entry, atm_price, atm_strike, time_to_expiry, RISK_FREE_RATE, DIVIDEND, 'call')
                volatility_pe = calculate_iv_merton(pe_entry, atm_price, atm_strike, time_to_expiry, RISK_FREE_RATE, DIVIDEND, 'put')

                sl_ce, Greek_CE, Delta_CE, Theta_CE, Vega_CE = calculate_greeks_sl(ce_entry, atm_price, atm_strike, time_to_expiry, volatility_ce or 0.15, atr_ce, "call")
                sl_pe, Greek_PE, Delta_PE, Theta_PE, Vega_PE = calculate_greeks_sl(pe_entry, atm_price, atm_strike, time_to_expiry, volatility_pe or 0.15, atr_pe, "put")

                base_stop_loss = -((sl_ce + sl_pe) / 2)
                SL             = -base_stop_loss
                trailing_sl    = base_stop_loss

                # ══════════════════════════════════════════════════════════
                # RECORD ENTRY Delta
                # ══════════════════════════════════════════════════════════
                entry_snap = build_greeks_snapshot(atm_price, atm_strike, time_to_expiry, ce_entry, pe_entry)
                entry_net_delta = entry_snap["net_delta"]
                print(f"  Entry Net Delta Baseline: {entry_net_delta:.4f}")
                # ══════════════════════════════════════════════════════════

                Quantity_C       = lot_size
                PNL_C            = 0
                PB_C             = 0
                max_profit       = 0.0
                exit_time_actual = None
                exit_premium     = None
                ras_kill_switch  = False
                live_ras         = 0.0
                live_net_delta   = 0.0 
                Exit_Type        = None
                base=0

                live_greeks_history = []
                for snap in preentry_greeks:
                    for _ in range(60):
                        live_greeks_history.append(snap)
                live_greeks_history = live_greeks_history[-RAS_1SEC_NORM:]

                current_time += timedelta(minutes=1)

                # ==========================================================
                # EXIT LOOP — tick-by-tick (1-second)
                # ==========================================================
                for ts, row in df_combined.loc[current_time:].iterrows():
                    current_total = row["total"]
                    Exit_CE       = row["ce"]
                    Exit_PE       = row["pe"]
                    spot          = row["spot"]
                    pnl           = entry_premium - current_total
                    tte_live      = (expiry_dt - ts) / timedelta(days=1) / 365

                    live_snap = build_greeks_snapshot(spot, atm_strike, tte_live, Exit_CE, Exit_PE)
                    live_greeks_history.append(live_snap)
                    if len(live_greeks_history) > RAS_1SEC_NORM: live_greeks_history.pop(0)

                    live_ras = compute_ras(live_greeks_history, RAS_1SEC_LOOKBACK, RAS_1SEC_NORM)
                    live_net_delta = abs(live_snap["net_delta"])

                    if pnl > max_profit:
                        max_profit  = pnl
                        trailing_sl = max_profit + base_stop_loss 
                        # print(f"TSL:{trailing_sl}")

                    # if (pnl > 15 and PB_C == 0) or (pnl > 30 and PB_C == 1):
                    if (pnl > 15 and PB_C == 0):
                        PNL_C      += pnl * partialqty
                        Quantity_C -= partialqty
                        PB_C       += 1
                        base = 1

                    # ── Delta Kill Switch ──────────────────────────────────
                    if live_net_delta > 0 and live_net_delta > DELTA_THRESHOLD:
                        exit_premium     = current_total
                        PNL_C           += (entry_premium - current_total) * Quantity_C
                        exit_time_actual = ts
                        Exit_Type        = "DELTA_SPIKE_KILL"
                        ras_kill_switch  = True
                        print(f"  [DELTA KILL] Delta Spiked  {live_net_delta:.4f} "
                              f"@ {ts.time()} | PnL=₹{PNL_C:.2f}")
                        break

                    # ── RAS Kill Switch ────────────────────────────────────
                    if live_ras > RAS_KILL_SWITCH:
                        exit_premium     = current_total
                        PNL_C           += (entry_premium - current_total) * Quantity_C
                        exit_time_actual = ts
                        Exit_Type        = "RAS_KILL_SWITCH"
                        ras_kill_switch  = True
                        print(f"  [RAS KILL] RAS={live_ras:.4f} Regime={ras_regime_label(live_ras)} "
                              f"@ {ts.time()} | PnL=₹{PNL_C:.2f}")
                        break

                    # ── Trailing SL ────────────────────────────────────────
                    if pnl <= trailing_sl:
                        exit_premium     = current_total
                        PNL_C           += (entry_premium - current_total) * Quantity_C
                        exit_time_actual = ts
                        Exit_Type        = "TSL"
                        print(f"  [TSL] Trailing SL HIT @ {ts.time()} | PnL=₹{PNL_C:.2f}")
                        break

                    # ── EOD exit ───────────────────────────────────────────
                    if ts >= exit_time:
                        exit_premium     = current_total
                        PNL_C           += (entry_premium - current_total) * Quantity_C
                        exit_time_actual = ts
                        Exit_Type        = "EOD"
                        print(f"  [EOD] EOD @ {ts.time()} | PnL=₹{PNL_C:.2f}")
                        break

                pnl = PNL_C
                if pnl > 0:
                    Profit_Trade += 1
                    Max_Profit    = max(Max_Profit, pnl)
                else:
                    Loss_Trade += 1
                    Max_Loss    = min(Max_Loss, pnl)

                AVG += pnl

                log_to_csv(csv_file, current_date, atm_strike, expiry,
                           entry_time_actual, Entry_CE, Entry_PE, entry_premium,
                           Greek_CE, Greek_PE, SL,
                           exit_time_actual, Exit_CE, Exit_PE, exit_premium,
                           Exit_Type, PB_C, pnl,
                           entry_ras=entry_ras,
                           exit_ras=live_ras,
                           kill_switch=ras_kill_switch,
                           entry_delta=entry_net_delta,     
                           exit_delta=live_net_delta)       

                if exit_time_actual is not None and exit_time_actual < exit_time:
                    current_time = exit_time_actual + timedelta(minutes=5)
                    continue
                else:
                    break

            else:
                current_time += timedelta(minutes=1)
                continue

        AVG = (AVG / Trade) if Trade > 0 else 0

    except Exception as e:
        log_to_csv(csv_file, current_date, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0, None, None, False, None, None)

    current_date += timedelta(days=1)
