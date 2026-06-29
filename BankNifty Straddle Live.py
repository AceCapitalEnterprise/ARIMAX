from breeze_connect import BreezeConnect
import hashlib
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
from datetime import datetime, date, timedelta, time as t
import csv
import time
import os
import requests
import threading
import json
import asyncio
import struct
import json
from bridgePy import connector
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import brentq
from math import log, sqrt, exp

# --- BreezeConnect SS ---
breeze = BreezeConnect(api_key="9581N(586t6E2A047i2p5(730@yO5t74")
breeze.generate_session(api_secret="9C3n144h3(5BA4f210074821g7V5+333", session_token="56117838")

# <--------------- Imp URL's --------------->

BASE_URL = "https://api.iiflcapital.com/v1"
URL = {
    "NSEEQ":   "https://api.iiflcapital.com/v1/contractfiles/NSEEQ.json",
    "INDICES": "https://api.iiflcapital.com/v1/contractfiles/INDICES.json",
    "NSEFO":   "https://api.iiflcapital.com/v1/contractfiles/NSEFO.json",
}

TIME_1 = t(10, 15)
TIME_2 = t(14, 59)
EXPIRY = "2026-06-30"          # change as needed
QTY    = 60
csv_file  = "IIFL_Straddle_RAS+Delta_Testing.csv"
temp_file = "open_position_IIFL_RD_Straddle_Trades_Testing.csv"
PROFIT_CUT_QTY = 30
ROUND_OFF      = 100
option_exchange = "NFO"
max_trades = 5
symbol     = "CNXBAN"

expiry  = datetime.strptime(EXPIRY, '%Y-%m-%d')
expiry1 = expiry.strftime('%Y-%m-%d')
expiry2 = expiry.strftime('%d-%b-%Y')

first_tick_received = threading.Event()
tick_data_lock      = threading.Lock()
tick_data           = {}
TOKENS              = {}
live_ltp            = {"CE": None, "PE": None}
entry_data          = None
sl_exit_time        = None
keep_receiving      = True

# ---------------------------------------------------------------------------
# [NEW] WebSocket-based live spot — populated by on_ticks, used in-trade
#       Replaces all get_live_spot() calls inside the exit loop
# ---------------------------------------------------------------------------
live_spot_ws: float | None = None

# ---------------------------------------------------------------------------
# Vomma and Delta Config
# ---------------------------------------------------------------------------
DELTA_THRESHOLD = 0.35
VOMMA_THRESHOLD = 2.5

# ---------------------------------------------------------------------------
# RAS Config
# ---------------------------------------------------------------------------
RAS_1MIN_LOOKBACK = 5
RAS_1MIN_NORM     = 20
RAS_1SEC_LOOKBACK = 5 * 60       
RAS_1SEC_NORM     = 20 * 60      

RAS_WEIGHT_DELTA = 0.2
RAS_WEIGHT_GAMMA = 0.3
RAS_WEIGHT_VEGA  = 0.5

RAS_ENTRY_LIMIT  = 2.0
RAS_KILL_SWITCH  = 4.0

# RAS runtime state
live_greeks_history   = []
live_ras              = 0.0
entry_ras             = 0.0
ras_kill_switch_fired = False

# ---------------------------------------------------------------------------
# Greek Config
# ---------------------------------------------------------------------------
RATE_LIMIT_DELAY = 3
RISK_FREE_RATE   = 0.07
DIVIDEND         = 0.014
W_DELTA          = 0.2
W_THETA          = 0.3
W_VEGA           = 0.5

# Live spot cache (kept for non-trade use such as ATR/ADX checks)
live_spot            = None
last_spot_fetch_time = None

USER_SESSION = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICIxVks4TEhlRnRvSmp6YWk1RmJlSGNPbDI3ekpGanBScTE2Vmt4eGJBZ0ZjIn0.eyJleHAiOjE3OTgyNTYyMzMsImlhdCI6MTc4MjcwNjIyNywianRpIjoib25ydHJ0Ojk3OWQ3N2YzLWVhNTItYjE3YS1lNGVjLTQxY2JlZTM3YTZhYiIsImlzcyI6Imh0dHBzOi8vMTAuMTI1LjY4LjE0NDo4MDgxL3JlYWxtcy9JSUZMIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImViMGQ0YzNmLTJmODAtNDY4ZC1iMTQwLWIxMjgyMmYwMGFkZSIsInR5cCI6IkJlYXJlciIsImF6cCI6IklJRkwiLCJzaWQiOiI0NDk5MjQxNi1lZGFmLTQ4MjEtYmRlOS05NDhmZDFiMmMyNGUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHA6Ly8xMC4xMjUuNjguMTQ0OjgwODAvIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLWlpZmwiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiSUlGTCI6eyJyb2xlcyI6WyJHVUVTVF9VU0VSIiwiQUNUSVZFX1VTRVIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIGVtYWlsIHByb2ZpbGUiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwib21zIjoiT05UMSIsInVjYyI6IkFDRUMyMDE2Iiwic29sYWNlX2dyb3VwIjoiU1VCU0NSSUJFUl9DTElFTlQiLCJuYW1lIjoiQUNFIENBUElUQUwgRU5URVJQUklTRSBOQSIsInByZWZlcnJlZF91c2VybmFtZSI6ImFjZWMyMDE2IiwiZ2l2ZW5fbmFtZSI6IkFDRSBDQVBJVEFMIEVOVEVSUFJJU0UiLCJmYW1pbHlfbmFtZSI6Ik5BIiwiZW1haWwiOiJhY2VjYXBpdGFsZW50QGdtYWlsLmNvbSJ9.KTYLGkLz7341xhAPFpCt1d18y0N-plnkwfgig6CIZPIU_sa4bTowlJM3T-Fzm4XBX-3m_GCx92A8CBYsFm_CNSVYg-xWZGiVi-2neD_SJitXWyYtszCGFjmyHdVhFZMIuWj__0AqsfO9IDjmp7UK5FiK1LK4wh6IGI5oM_a0Zylp1RQR6Hg0ez6g9NtRqL3cH8gPJRZ60zZV50MRvwAwHO4FTC_E1GsUWmQk2rqEhFxtkCdwCvO9BvgFRPp6AJQORCMP4MPGfLMQBZEZrtWJjPuJY3wxn1TOYar4h8GeUODSmy9s0qkB4RdK5fwDiXWXLQoX3HbB1Q9xK_6S8hC4Xg"
print(USER_SESSION)

headers = {
    "Authorization": f"Bearer {USER_SESSION}",
}


def get_key_from_params(strike, right):
    normalized_right = right.upper().replace('CALL', 'CE').replace('PUT', 'PE')
    return f"{strike}_{normalized_right}"



def on_ticks(ticks):
    global live_spot_ws

    if "strike_price" not in ticks:
        ltp = ticks.get("last") or ticks.get("ltp")
        if ltp is not None:
            live_spot_ws = float(ltp)
        else:
            print(f"Subscription message: {ticks}")
        return

    # ── Options tick ────────────────────────────────────────────────
    if 'right' not in ticks:
        print(f"Subscription message: {ticks}")
        return

    strike      = ticks['strike_price']
    right       = ticks['right']
    key         = get_key_from_params(strike, right)
    current_ltp = ticks.get("last")

    if current_ltp is None:
        return

    with tick_data_lock:
        tick_data[key] = ticks
        if key.endswith("CE"):
            live_ltp["CE"] = current_ltp
        elif key.endswith("PE"):
            live_ltp["PE"] = current_ltp

        if live_ltp["CE"] is not None and live_ltp["PE"] is not None:
            first_tick_received.set()

    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] {key} LTP: {current_ltp}")


# ═══════════════════════════════════════════════════════════════════
# [NEW] CNXBAN spot WebSocket subscription helpers
#   subscribe once at startup; keep alive across all trades;
#   unsubscribe only on full shutdown.
# ═══════════════════════════════════════════════════════════════════
def subscribe_nifty_spot_ws():
    """Subscribe to CNXBAN index feed so live_spot_ws is always current."""
    try:
        breeze.subscribe_feeds(
            exchange_code    = "NSE",
            stock_code       = "CNXBAN",
            product_type     = "index",
            get_market_depth = False,
            get_exchange_quotes = True,
        )
        print("[WS] Subscribed to CNXBAN spot index feed")
    except Exception as e:
        print(f"[ERROR] subscribe_nifty_spot_ws: {e}")


def unsubscribe_nifty_spot_ws():
    """Call only on full program shutdown."""
    try:
        breeze.unsubscribe_feeds(
            exchange_code    = "NSE",
            stock_code       = "CNXBAN",
            product_type     = "index",
            get_market_depth = False,
            get_exchange_quotes = True,
        )
        print("[WS] Unsubscribed CNXBAN spot index feed")
    except Exception as e:
        print(f"[ERROR] unsubscribe_nifty_spot_ws: {e}")


def initiate_ws(strike_price):
    try:
        exp    = str(expiry2)
        strike = str(strike_price)
        live_ltp["CE"] = None
        live_ltp["PE"] = None
        first_tick_received.clear()

        breeze.subscribe_feeds(
            exchange_code="NFO", stock_code="CNXBAN",
            expiry_date=exp, strike_price=strike, right="call",
            product_type="options", get_market_depth=False, get_exchange_quotes=True
        )
        breeze.subscribe_feeds(
            exchange_code="NFO", stock_code="CNXBAN",
            expiry_date=exp, strike_price=strike, right="put",
            product_type="options", get_market_depth=False, get_exchange_quotes=True
        )
        print(f"Subscribed to {strike_price}_CE and {strike_price}_PE")

        success = first_tick_received.wait(timeout=5)
        if not success:
            print("[WARNING] WebSocket subscription timed out. No initial ticks received.")

    except Exception as e:
        print(f"[ERROR] Failed to Initiate Web Socket: {e}")


def deactivate_ws(strike_price):
    try:
        exp    = str(expiry2)
        strike = str(strike_price)
        breeze.unsubscribe_feeds(
            exchange_code="NFO", stock_code="CNXBAN",
            expiry_date=exp, strike_price=strike, right="call",
            product_type="options", get_market_depth=False, get_exchange_quotes=True
        )
        breeze.unsubscribe_feeds(
            exchange_code="NFO", stock_code="CNXBAN",
            expiry_date=exp, strike_price=strike, right="put",
            product_type="options", get_market_depth=False, get_exchange_quotes=True
        )
        with tick_data_lock:
            tick_data.clear()
            live_ltp["CE"] = None
            live_ltp["PE"] = None
        print(f"Unsubscribed {strike_price} straddle")
    except Exception as e:
        print(f"[ERROR] Failed to Deactivate Web Socket: {e}")


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------
def merton_price(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def calculate_iv_merton(option_price, S, K, T, r, q, option_type):
    try:
        if T <= 0 or option_price <= 0:
            return 0.001
        objective_func = lambda sigma: merton_price(S, K, T, r, q, sigma, option_type) - option_price
        return brentq(objective_func, 1e-4, 5.0)
    except ValueError:
        return np.nan


def merton_greeks(entry, S, K, T, sigma, r, q, option_type="call"):
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1        = norm.cdf(d1)
    N_prime_d1  = norm.pdf(d1)
    gamma       = (np.exp(-q * T) * N_prime_d1) / (S * sigma * np.sqrt(T))
    vega        = (S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T)) / 100
    vomma       = vega * (d1 * d2) / sigma if sigma != 0 else 0
    if option_type == "call":
        delta = np.exp(-q * T) * N_d1
        theta = (-np.exp(-q * T) * S * N_prime_d1 * sigma / (2 * np.sqrt(T))) \
                + (q * S * np.exp(-q * T) * N_d1) \
                - (r * K * np.exp(-r * T) * norm.cdf(d2))
        theta = theta / 365
    else:
        delta = np.exp(-q * T) * (N_d1 - 1)
        theta = (-np.exp(-q * T) * S * N_prime_d1 * sigma / (2 * np.sqrt(T))) \
                - (q * S * np.exp(-q * T) * norm.cdf(-d1)) \
                + (r * K * np.exp(-r * T) * norm.cdf(-d2))
        theta = theta / 365

    intrinsic = max(0, S - K)
    extrinsic = max(0, entry - intrinsic)
    if T < (1 / 365):
        theta = -min(abs(theta), extrinsic)

    return delta, theta, vega, gamma, vomma


def calculate_all_greeks(entry, spot, strike, T, sigma, option_type):
    return merton_greeks(entry, spot, strike, T, sigma, RISK_FREE_RATE, DIVIDEND, option_type)


def calculate_greeks_sl(entry_time, exit_time, entry, spot, strike, T, sigma, atr, option_type):
    delta, theta, vega, gamma, vomma = merton_greeks(
        entry, spot, strike, T, sigma, RISK_FREE_RATE, DIVIDEND, option_type
    )
    tag = "CE" if option_type == "call" else "PE"
    print(f"  Delta {tag}: {delta:.4f} | Theta {tag}: {theta:.4f} | "
          f"Vega {tag}: {vega:.4f} | Gamma {tag}: {gamma:.4f} | Vomma {tag}: {vomma:.4f}")

    total_minutes = 369
    minutes_left  = (exit_time.hour * 60 + exit_time.minute) - \
                    (entry_time.hour * 60 + entry_time.minute)
    time_factor   = minutes_left / total_minutes if minutes_left > 0 else 0.01
    vol_factor    = atr / spot

    sl_offset = (W_DELTA * abs(delta) * atr +
                 W_THETA * abs(theta) * time_factor +
                 W_VEGA  * vega * vol_factor)
    sl = max(15, min(sl_offset * 6, 25))
    return sl


# ═══════════════════════════════════════════════════════════════════
# RAS (Regime Acceleration Score)
# ═══════════════════════════════════════════════════════════════════

def build_greeks_snapshot(spot, strike, tte, ce_price, pe_price):
    iv_ce = calculate_iv_merton(ce_price, spot, strike, tte, RISK_FREE_RATE, DIVIDEND, "call")
    iv_pe = calculate_iv_merton(pe_price, spot, strike, tte, RISK_FREE_RATE, DIVIDEND, "put")

    d_ce, t_ce, v_ce, g_ce, vomma_ce = calculate_all_greeks(ce_price, spot, strike, tte, iv_ce, "call")
    d_pe, t_pe, v_pe, g_pe, vomma_pe = calculate_all_greeks(pe_price, spot, strike, tte, iv_pe, "put")

    return {
        "net_delta": d_ce + d_pe,
        "net_gamma": g_ce + g_pe,
        "net_vega":  v_ce + v_pe,
        "net_vomma": vomma_ce + vomma_pe,
        "d_ce": d_ce, "d_pe": d_pe,
        "g_ce": g_ce, "g_pe": g_pe,
        "v_ce": v_ce, "v_pe": v_pe,
        "vomma_ce": vomma_ce, "vomma_pe": vomma_pe,
        "iv_ce": iv_ce, "iv_pe": iv_pe,
    }


def compute_ras(greeks_history, lookback_bars, norm_window_bars):
    history = greeks_history[-norm_window_bars:]
    if len(history) < lookback_bars + 1:
        return 0.0

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
    return "EXPANSION / TAIL_RISK"


def build_preentry_greeks_history(strike, check_time, expiry_dt, lookback_min=25):
    hist_start = check_time - timedelta(minutes=lookback_min)
    try:
        hist_ce = get_1_min_option_historical(expiry, strike, "call", hist_start, check_time)
        hist_pe = get_1_min_option_historical(expiry, strike, "put",  hist_start, check_time)

        if (not hist_ce or "Success" not in hist_ce or not hist_ce["Success"] or
                not hist_pe or "Success" not in hist_pe or not hist_pe["Success"]):
            print("[RAS Pre-entry] Option history unavailable — skipping RAS check")
            return []

        df_ce = pd.DataFrame(hist_ce["Success"])
        df_pe = pd.DataFrame(hist_pe["Success"])
        df_ce['datetime'] = pd.to_datetime(df_ce['datetime'])
        df_ce.set_index('datetime', inplace=True)
        df_pe['datetime'] = pd.to_datetime(df_pe['datetime'])
        df_pe.set_index('datetime', inplace=True)

        nifty_raw = get_1_min_historical(hist_start, check_time)
        if not nifty_raw or "Success" not in nifty_raw or not nifty_raw["Success"]:
            print("[RAS Pre-entry] Spot data unavailable — skipping RAS check")
            return []

        df_spot = pd.DataFrame(nifty_raw["Success"])
        df_spot['datetime'] = pd.to_datetime(df_spot['datetime'])
        df_spot.set_index('datetime', inplace=True)

        common_index   = df_ce.index.intersection(df_pe.index)
        greeks_history = []
        for ts in common_index:
            try:
                ce_price = float(df_ce.loc[ts, 'close'])
                pe_price = float(df_pe.loc[ts, 'close'])
                spot_ts  = df_spot.index.asof(ts)
                if pd.isnull(spot_ts):
                    continue
                spot_val = float(df_spot.loc[spot_ts, 'close'])
                tte = (expiry_dt - ts).total_seconds() / (365 * 24 * 60 * 60)
                if tte <= 0:
                    continue
                snap = build_greeks_snapshot(spot_val, strike, tte, ce_price, pe_price)
                greeks_history.append(snap)
            except Exception:
                continue

        print(f"[RAS Pre-entry] Built {len(greeks_history)} 1-min Greek bars")
        return greeks_history

    except Exception as e:
        print(f"[ERROR] build_preentry_greeks_history: {e}")
        return []



def build_1sec_greeks_seed(strike: int, now: datetime, expiry_dt: datetime,
                           lookback_sec: int = 120) -> list:
   
    hist_start = now - timedelta(seconds=lookback_sec)
    fmt        = "%Y-%m-%dT%H:%M:%S.000Z"

    try:
        # ── 1-sec CE ────────────────────────────────────────────────
        raw_ce = breeze.get_historical_data_v2(
            interval      = "1second",
            from_date     = hist_start.strftime(fmt),
            to_date       = now.strftime(fmt),
            stock_code    = "CNXBAN",
            exchange_code = "NFO",
            product_type  = "options",
            expiry_date   = expiry2,
            right         = "call",
            strike_price  = str(strike),
        )

        # ── 1-sec PE ────────────────────────────────────────────────
        raw_pe = breeze.get_historical_data_v2(
            interval      = "1second",
            from_date     = hist_start.strftime(fmt),
            to_date       = now.strftime(fmt),
            stock_code    = "CNXBAN",
            exchange_code = "NFO",
            product_type  = "options",
            expiry_date   = expiry2,
            right         = "put",
            strike_price  = str(strike),
        )

        # ── 1-sec CNXBAN spot (cash) ──────────────────────────────────
        raw_spot = breeze.get_historical_data_v2(
            interval      = "1second",
            from_date     = hist_start.strftime(fmt),
            to_date       = now.strftime(fmt),
            stock_code    = "CNXBAN",
            exchange_code = "NSE",
            product_type  = "cash",
        )

        # ── validate ────────────────────────────────────────────────
        def _ok(r):
            return r and "Success" in r and r["Success"]

        if not (_ok(raw_ce) and _ok(raw_pe) and _ok(raw_spot)):
            print("[1sec Seed] One or more feeds returned no data — "
                  "live_greeks_history will be empty; RAS stays 0.0 "
                  "until enough live ticks build up naturally")
            return []

        # ── DataFrames ──────────────────────────────────────────────
        def _df(raw):
            d = pd.DataFrame(raw["Success"])
            d["datetime"] = pd.to_datetime(d["datetime"])
            d.set_index("datetime", inplace=True)
            return d

        df_ce   = _df(raw_ce)
        df_pe   = _df(raw_pe)
        df_spot = _df(raw_spot)

        # ── build one snapshot per second ───────────────────────────
        common_idx     = df_ce.index.intersection(df_pe.index)
        greeks_history = []

        for ts in common_idx:
            try:
                ce_price = float(df_ce.loc[ts, "close"])
                pe_price = float(df_pe.loc[ts, "close"])

                spot_ts = df_spot.index.asof(ts)
                if pd.isnull(spot_ts):
                    continue
                spot_val = float(df_spot.loc[spot_ts, "close"])

                tte = (expiry_dt - ts).total_seconds() / (365 * 24 * 60 * 60)
                if tte <= 0:
                    continue

                snap = build_greeks_snapshot(spot_val, strike, tte, ce_price, pe_price)
                greeks_history.append(snap)

            except Exception:
                continue   # skip any malformed row silently

        print(f"[1sec Seed] Built {len(greeks_history)} 1-sec Greek bars "
              f"(requested {lookback_sec}s window)")
        return greeks_history

    except Exception as e:
        print(f"[ERROR] build_1sec_greeks_seed: {e}")
        return []


# kept for non-trade use (ATR/ADX checks etc.) — NOT used inside exit loop
def get_live_spot():
    global live_spot, last_spot_fetch_time
    now = datetime.now()
    if (live_spot is not None and
            last_spot_fetch_time is not None and
            (now - last_spot_fetch_time).seconds < 60):
        return live_spot
    try:
        data = get_1_min_historical(now - timedelta(minutes=3), now)
        if data and "Success" in data and data["Success"]:
            df               = pd.DataFrame(data["Success"])
            live_spot        = float(df.iloc[-1]['close'])
            last_spot_fetch_time = now
            return live_spot
    except Exception as e:
        print(f"[ERROR] get_live_spot: {e}")
    return live_spot


# ═══════════════════════════════════════════════════════════════════
# Position state persistence
# ═══════════════════════════════════════════════════════════════════

def save_entry_data_to_csv(entry_data):
    try:
        with open(temp_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(entry_data.keys())
            writer.writerow(entry_data.values())
    except Exception as e:
        print(f"Failed to Save Entry Data", exc_info=e)
        raise


def load_entry_data_from_csv():
    try:
        if not os.path.exists(temp_file):
            return None
        with open(temp_file, "r") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
            if not rows:
                return None
            row = rows[0]
            row["trade_rem"] = int(row["trade_rem"])
            row["time"]      = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S.%f")
            row["atm"]       = int(row["atm"])
            row["entry"]     = float(row["entry"])
            row["current"]   = float(row["current"])
            row["cut"]       = int(row["cut"])
            row["Qty_B"]     = int(row["Qty_B"])
            row["max_pnl"]   = float(row["max_pnl"])
            row["sl"]        = float(row["sl"])
            row["tsl"]       = float(row["tsl"])
            row["ce_token"]  = int(row["ce_token"])
            row["pe_token"]  = int(row["pe_token"])
            row["entry_ras"] = float(row["entry_ras"])
            row["pnl"]       = float(row["pnl"])
            return row
    except Exception as e:
        print(f"Failed to Load Entry Data", exc_info=e)
        raise


def clear_position_state():
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        print(f"Failed to Delete Entry Data", exc_info=e)
        raise


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════

def round_to_nearest_50(x):
    return int(round(x / float(ROUND_OFF))) * ROUND_OFF


def write_to_csv(data):
    hdrs = [
        'Date', 'Strike', 'Entry Time', 'Entry premium', 'SL',
        'Exit Time', 'Exit premium', 'Exit Type', 'Max PnL',
        'Partial Booking', 'PnL', 'Quantity', 'Total',
        'Entry_RAS', 'Exit_RAS', 'RAS_KillSwitch'
    ]
    try:
        try:
            with open(csv_file, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(hdrs)
                print("Created new CSV file")
        except FileExistsError:
            pass
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print(f"Trade data written to CSV: {data}")
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")


def get_instrument_id(exchange, symbol, expiry=None, strike=None, opt=None):
    try:
        data   = requests.get(URL[exchange]).json()
        symbol = symbol.upper()
        if exchange == "NSEEQ":
            return next((i["instrumentId"] for i in data
                         if i["underlyingInstrumentSymbol"] == symbol), None)
        if exchange == "INDICES":
            return next((i["instrumentId"] for i in data
                         if symbol in (i["underlyingInstrumentSymbol"],
                                       i["tradingSymbol"].replace(" INDEX", ""))), None)
        if exchange == "NSEFO":
            exp = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d-%b-%Y")
            return next((i["instrumentId"] for i in data
                         if i["underlyingInstrumentSymbol"] == symbol
                         and exp in i["expiry"]
                         and (opt is None or (i["optionType"] == opt and
                                              float(i["strikePrice"]) == float(strike)))), None)
    except Exception as e:
        print(f"[ERROR] Failed to Fetch Instrument ID: {e}")
        return None


def fetch_market_feed_scrip(id):
    try:
        url    = f"{BASE_URL}/marketdata/marketquotes"
        params = [{"instrumentId": id, "exchange": "NSEFO"}]
        response = requests.post(url, headers=headers, json=params)
        data     = response.json()
        if data["status"] == "Ok":
            return data["result"][0]["ltp"]
        print("Error:", data)
        return None
    except Exception as e:
        print(f"[ERROR] Fetching Live data: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Historical data fetchers
# ═══════════════════════════════════════════════════════════════════

def get_historical_candles(exchange, instrument_id, interval, from_date, to_date):
    try:
        url     = f"{BASE_URL}/marketdata/historicaldata"
        payload = {
            "exchange":     exchange,
            "instrumentId": instrument_id,
            "interval":     interval,
            "fromDate":     from_date,
            "toDate":       to_date,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["status"] != "Ok":
            raise Exception(data)
        return data["result"][0]["candles"]
    except Exception as e:
        print(f"[ERROR] Failed to Fetch Daily Historical Data: {e}")
        return None


def get_1_min_historical(from_date, to_date, exchange="NSE", interval="1minute"):
    try:
        return breeze.get_historical_data_v2(
            interval      = interval,
            from_date     = from_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            to_date       = to_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            stock_code    = "CNXBAN",
            exchange_code = exchange,
            product_type  = "cash"
        )
    except Exception as e:
        print(f"[ERROR] Failed to Fetch Historical Data: {e}")
        return None


def get_1_min_option_historical(expiry, strike, right, from_date, to_date):
    try:
        return breeze.get_historical_data_v2(
            interval      = "1minute",
            from_date     = from_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            to_date       = to_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            stock_code    = "CNXBAN",
            exchange_code = "NFO",
            product_type  = "options",
            expiry_date   = expiry,
            right         = right,
            strike_price  = strike
        )
    except Exception as e:
        print(f"[ERROR] Failed to Fetch Historical Option Data: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Market condition checks
# ═══════════════════════════════════════════════════════════════════

def check_GAP():
    try:
        now   = datetime.now()
        s     = (now - timedelta(days=10)).date()
        e     = now.date()
        start = s.strftime('%d-%b-%Y')
        end   = e.strftime('%d-%b-%Y')
        hist  = get_historical_candles("NSEEQ", 999920005, "1 day", start, end) 
        df    = pd.DataFrame(hist, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("datetime", inplace=True)
        if len(df) < 3:
            return None
        today    = df.iloc[-1]['Open']
        previous = df.iloc[-2]['Close']
        gap      = ((today - previous) / previous) * 100
        GAP_day  = abs(gap) <= 1.25
        print(f"[DEBUG] GAP: {gap:.5f} | Straddle Day => {GAP_day}")
        return GAP_day
    except Exception as e:
        print(f"[ERROR] GAP checking: {e}")
        return None


def check_atr_adx():
    try:
        date       = datetime.now()
        x          = t(14, 10)
        nifty_data = get_1_min_historical(
            (date - timedelta(days=4)).replace(hour=9, minute=15), date
        )
        df = pd.DataFrame(nifty_data["Success"])
        if df.empty:
            raise ValueError("CNXBAN data not available")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df["ADX"]     = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
        df["ATR"]     = ta.atr(df["high"], df["low"], df["close"])
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["RV"]      = df["log_ret"].rolling(window=10).std() * np.sqrt(252 * 390)
        df.dropna(inplace=True)
        if len(df) < 15:
            return None
        last   = df.iloc[-5]
        prev10 = df.iloc[-15]
        atr_down = prev10["ATR"] > last["ATR"]
        adx_down = prev10["ADX"] > last["ADX"]
        rv_down  = prev10["RV"]  > last["RV"]
        print(f"[DEBUG] ATR: {prev10['ATR']:.2f} > {last['ATR']:.2f} => {atr_down}")
        print(f"[DEBUG] ADX: {prev10['ADX']:.2f} > {last['ADX']:.2f} => {adx_down}")
        print(f"[DEBUG] RV:  {prev10['RV']:.2f}  > {last['RV']:.2f}  => {rv_down}")
        if (date.time() < x) and rv_down and atr_down and adx_down:
            print("Entry conditions matched")
            return last["close"]
        return None
    except Exception as e:
        print(f"[ERROR] ATR/ADX: {e}")
        return None


def check_nearest_ltp(atm, Cc, Cu, Cd, Pc, Pu, Pd):
    try:
        cc = fetch_market_feed_scrip(Cc)
        cu = fetch_market_feed_scrip(Cu)
        cd = fetch_market_feed_scrip(Cd)
        pc = fetch_market_feed_scrip(Pc)
        pu = fetch_market_feed_scrip(Pu)
        pd = fetch_market_feed_scrip(Pd)

        m1 = abs(cc - pc)
        m2 = abs(cd - pd)
        m3 = abs(cu - pu)
        m  = min(m1, m2, m3)
        print(f" CC: {cc:.2f} | PC: {pc:.2f} | Diff: {m1:.2f}")
        print(f" CD: {cd:.2f} | PD: {pd:.2f} | Diff: {m2:.2f}")
        print(f" CU: {cu:.2f} | PU: {pu:.2f} | Diff: {m3:.2f}")
        print(f" ATM: {m1:.2f} | Down: {m2:.2f} | Up: {m3:.2f} | Best: {m:.2f}")

        if m == m1 and m <= 0.2 * max(cc, pc):
            return Cc, Pc, atm
        elif m == m2 and m <= 0.2 * max(cd, pd):
            return Cd, Pd, atm - 50
        elif m == m3 and m <= 0.2 * max(cu, pu):
            return Cu, Pu, atm + 50
        print("Failed premium matching")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] Premium matching: {e}")
        return None, None, None


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════
print("[START] Waiting for signal...")

breeze.on_ticks = on_ticks

print("Connecting to Breeze WebSocket...")
breeze.ws_connect()
time.sleep(2)

# [NEW] Subscribe to CNXBAN spot index feed once — stays alive all session
subscribe_nifty_spot_ws()

GAP = check_GAP()

# ── Resume open position from disk ──────────────────────────────────
entry_data = load_entry_data_from_csv()
if entry_data:
    TOKENS["CE"]        = entry_data["ce_token"]
    TOKENS["PE"]        = entry_data["pe_token"]
    max_trades          = entry_data["trade_rem"]
    entry_ras           = entry_data["entry_ras"]
    atm_strike          = entry_data["atm"]
    # On resume live_greeks_history starts empty;
    # RAS returns 0.0 until enough live ticks accumulate (safe — no explosion).
    live_greeks_history = []
    print(f"[RESUME] Open position CE:{TOKENS['CE']} PE:{TOKENS['PE']}")
    initiate_ws(atm_strike)
    print("[RESUME] Loaded open position from CSV")

try:
    while True:
        now = datetime.now()
        live_net_delta   = 0.0 
        entry_net_delta =  0.0

        # ══════════════════════════════════════════════════════════
        # ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if (TIME_1 < now.time() < TIME_2 and GAP and max_trades > 0
                and entry_data is None and now.second == 0):

            # Cool-down after SL exit
            if sl_exit_time and (now - sl_exit_time).seconds < 300:
                time.sleep(1)
                continue

            print(f"Checking ATR/ADX at {now.time()}")
            spot = check_atr_adx()
            # spot = 58428

            if spot is not None:
                atm = round_to_nearest_50(spot)
                print(f"Spot: {spot:.2f} | ATM: {atm}")

                Cc = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm,               "CE")
                time.sleep(0.2)
                Cu = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm + ROUND_OFF,   "CE")
                time.sleep(0.2)
                Cd = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm - ROUND_OFF,   "CE")
                time.sleep(0.2)
                Pc = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm,               "PE")
                time.sleep(0.2)
                Pu = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm + ROUND_OFF,   "PE")
                time.sleep(0.2)
                Pd = get_instrument_id("NSEFO", "BANKNIFTY", EXPIRY, atm - ROUND_OFF,   "PE")

                C, P, atm = check_nearest_ltp(atm, Cc, Cu, Cd, Pc, Pu, Pd)
                TOKENS["CE"] = C
                TOKENS["PE"] = P

                if not TOKENS["CE"] or not TOKENS["PE"]:
                    print("[ABORT] Invalid instrument ID. Skipping.")
                    time.sleep(1)
                    continue

                expiry_dt        = expiry + timedelta(hours=15, minutes=30)
                preentry_history = build_preentry_greeks_history(
                    atm, now, expiry_dt, lookback_min=25
                )
                entry_ras = compute_ras(
                    preentry_history,
                    lookback_bars    = RAS_1MIN_LOOKBACK,
                    norm_window_bars = RAS_1MIN_NORM
                )
                print(f"[RAS Pre-entry] RAS = {entry_ras:.4f} "
                      f"| Regime = {ras_regime_label(entry_ras)}")

                if entry_ras > RAS_ENTRY_LIMIT:
                    print(f"[BLOCKED] Pre-entry RAS {entry_ras:.4f} > "
                          f"{RAS_ENTRY_LIMIT} — skipping entry")
                    time.sleep(1)
                    continue

                initiate_ws(atm)
                time.sleep(3)

                if live_ltp["CE"] and live_ltp["PE"]:
                    ce_entry      = live_ltp["CE"]
                    pe_entry      = live_ltp["PE"]
                    entry_premium = ce_entry + pe_entry

                    if entry_premium > 50:
                        entry_time = now
                        exit_time  = datetime.combine(now.date(), TIME_2)
                        tte = (expiry_dt - now).total_seconds() / (365 * 24 * 60 * 60)

                        # ATR / IV for Greek SL
                        start_hist = now - timedelta(minutes=30)

                        hist_ce = get_1_min_option_historical(expiry, atm, "call", start_hist, now)
                        df_ce   = pd.DataFrame(hist_ce["Success"])
                        if df_ce.empty:
                            raise ValueError("CE data not available")
                        df_ce['datetime'] = pd.to_datetime(df_ce['datetime'])
                        df_ce.set_index('datetime', inplace=True)
                        df_ce['ATR'] = ta.atr(df_ce['high'], df_ce['low'], df_ce['close'], length=14)
                        atr_ce = df_ce.iloc[-1]['ATR']
                        vol_ce = calculate_iv_merton(ce_entry, spot, atm, tte,
                                                     RISK_FREE_RATE, DIVIDEND, 'call')

                        hist_pe = get_1_min_option_historical(expiry, atm, "put", start_hist, now)
                        df_pe   = pd.DataFrame(hist_pe["Success"])
                        if df_pe.empty:
                            raise ValueError("PE data not available")
                        df_pe['datetime'] = pd.to_datetime(df_pe['datetime'])
                        df_pe.set_index('datetime', inplace=True)
                        df_pe['ATR'] = ta.atr(df_pe['high'], df_pe['low'], df_pe['close'], length=14)
                        atr_pe = df_pe.iloc[-1]['ATR']
                        vol_pe = calculate_iv_merton(pe_entry, spot, atm, tte,
                                                     RISK_FREE_RATE, DIVIDEND, 'put')

                        sl_ce = calculate_greeks_sl(
                            entry_time, exit_time, ce_entry, spot, atm, tte,
                            vol_ce or 0.15, atr_ce, "call"
                        )
                        sl_pe = calculate_greeks_sl(
                            entry_time, exit_time, pe_entry, spot, atm, tte,
                            vol_pe or 0.15, atr_pe, "put"
                        )

                        STOP_LOSS = -((sl_ce + sl_pe) / 2)
                        print(f"Stop Loss: {STOP_LOSS:.4f}")
                        CE_SP = live_ltp["CE"] - 0.5
                        PE_SP = live_ltp["PE"] - 0.5

                        time.sleep(1)

                        #
                        live_greeks_history = build_1sec_greeks_seed(
                            atm, now, expiry_dt, lookback_sec=120
                        )
                        # Trim to norm window (1200 bars = 20 min of 1-sec data)
                        max_len = RAS_1SEC_NORM
                        live_greeks_history = live_greeks_history[-max_len:]

                        live_ras              = entry_ras
                        ras_kill_switch_fired = False
                        live_spot             = spot

                        max_trades -= 1
                        entry_data = {
                            "trade_rem": max_trades,
                            "time":      entry_time,
                            "atm":       atm,
                            "entry":     entry_premium,
                            "current":   0,
                            "cut":       0,
                            "Qty_B":     0,
                            "max_pnl":   0,
                            "sl":        STOP_LOSS,
                            "tsl":       STOP_LOSS,
                            "ce_token":  TOKENS["CE"],
                            "pe_token":  TOKENS["PE"],
                            "entry_ras": entry_ras,
                            "pnl":       0
                        }
                        save_entry_data_to_csv(entry_data)
                        sl_exit_time = None

                        print(
                            f"[ENTRY] Time:{now.time()} | ATM:{atm} | "
                            f"Premium:{entry_premium:.2f} | Qty:{QTY} | "
                            f"SL:{STOP_LOSS:.4f} | RAS:{entry_ras:.4f}"
                        )
                    else:
                        print("Premium too low — skipping")

        # ══════════════════════════════════════════════════════════
        # TRAIL SL / LIVE RAS CHECK (in-trade)
        # ══════════════════════════════════════════════════════════
        

        elif entry_data and TIME_1 < now.time() < TIME_2:
            if live_ltp["CE"] and live_ltp["PE"]:
                current_premium = live_ltp["CE"] + live_ltp["PE"]
                pnl             = entry_data["entry"] - current_premium
                Exit_Type       = None
               

                entry_data["current"] = current_premium

                # Update trailing SL
                if pnl > entry_data["max_pnl"]:
                    entry_data["max_pnl"] = pnl
                    entry_data["tsl"]     = pnl + entry_data["sl"]

                # ── [UPDATED] Build 1-sec greeks snapshot using WS spot ──
                # live_spot_ws is populated by on_ticks() from the CNXBAN
                # index subscription — zero API calls needed.
                spot = live_spot_ws
                if spot is None:
                    print("[WARN] live_spot_ws not yet populated — "
                          "skipping RAS tick, retrying next second")
                    time.sleep(1)
                    continue

                atm_strike = entry_data["atm"]
                expiry_dt  = expiry + timedelta(hours=15, minutes=30)
                tte_live   = (expiry_dt - now).total_seconds() / (365 * 24 * 60 * 60)

                if tte_live > 0:
                    live_snap = build_greeks_snapshot(
                        spot, atm_strike, tte_live,
                        live_ltp["CE"], live_ltp["PE"]
                    )
                    live_greeks_history.append(live_snap)

                    # Bound history to norm window
                    if len(live_greeks_history) > RAS_1SEC_NORM:
                        live_greeks_history.pop(0)

                    # Recompute RAS every tick using 1-sec lookback / norm
                    live_ras = compute_ras(
                        live_greeks_history,
                        lookback_bars    = RAS_1SEC_LOOKBACK,
                        norm_window_bars = RAS_1SEC_NORM
                    )
                    live_net_delta = abs(live_snap["net_delta"])
                    print(f"[RAS Live] {live_ras:.4f} "
                          f"| {ras_regime_label(live_ras)}", end="\r")

                save_entry_data_to_csv(entry_data)

                # ── Partial profit booking ───────────────────────────────
                if pnl > 15 and entry_data["cut"] == 0:
                    book_prem = current_premium
                    CE_BP = live_ltp["CE"] + 0.5
                    PE_BP = live_ltp["PE"] + 0.5
                    partial_pnl         = (entry_data["entry"] - book_prem) * PROFIT_CUT_QTY
                    entry_data["pnl"]  += partial_pnl
                    entry_data["Qty_B"] += PROFIT_CUT_QTY
                    entry_data["cut"]  += 1
                    save_entry_data_to_csv(entry_data)
                    print(
                        f"\n[PARTIAL] Time:{now.strftime('%H:%M:%S')} | "
                        f"ExitPrem:{book_prem:.2f} | PnL:₹{partial_pnl:.2f}"
                    )

                # ── RAS Kill Switch ──────────────────────────────────────
                if live_ras > RAS_KILL_SWITCH and not ras_kill_switch_fired:
                    Q         = QTY - entry_data["Qty_B"]
                    CE_BP     = live_ltp["CE"] + 0.5
                    PE_BP     = live_ltp["PE"] + 0.5
                    exit_prem = current_premium
                    final_pnl = (((entry_data["entry"] - exit_prem) * Q) +
                                 entry_data["pnl"]) / QTY
                    Total     = final_pnl * QTY
                    Exit_Type = "RAS_KILL"
                    ras_kill_switch_fired = True

                    deactivate_ws(entry_data["atm"])
                    Ti = now.strftime("%H:%M:%S")
                    print(
                        f"\n[RAS KILL] Time:{Ti} | RAS:{live_ras:.4f} | "
                        f"ExitPrem:{exit_prem:.2f} | PnL:{final_pnl:.2f} | "
                        f"Total:₹{Total:.2f}"
                    )

                    e = entry_data["entry"]
                    write_to_csv([
                        entry_data["time"].strftime("%Y-%m-%d"),
                        entry_data["atm"],
                        entry_data["time"].strftime("%H:%M:%S"),
                        f"{e:.2f}",
                        entry_data["sl"],
                        Ti,
                        f"{exit_prem:.2f}",
                        Exit_Type,
                        entry_data["max_pnl"],
                        entry_data["cut"],
                        f"{final_pnl:.2f}",
                        QTY,
                        f"{Total:.2f}",
                        f"{entry_ras:.4f}",
                        f"{live_ras:.4f}",
                        "YES",
                    ])
                    entry_data          = None
                    live_greeks_history = []
                    clear_position_state()
                    TOKENS.clear()
                    sl_exit_time = now
                    time.sleep(1)
                    continue

                # ── Delta SL exit ─────────────────────────────────────
                if live_net_delta > DELTA_THRESHOLD:
                    Q         = QTY - entry_data["Qty_B"]
                    CE_BP     = live_ltp["CE"] + 0.5
                    PE_BP     = live_ltp["PE"] + 0.5
                    exit_prem = current_premium
                    final_pnl = (((entry_data["entry"] - exit_prem) * Q) +
                                 entry_data["pnl"]) / QTY
                    Total     = final_pnl * QTY
                    Exit_Type = "Delta Spike"

                    deactivate_ws(entry_data["atm"])
                    Ti = now.strftime("%H:%M:%S")
                    print(
                        f"\n[Delta Spike] Time:{Ti} | ExitPrem:{exit_prem:.2f} | "
                        f"PnL:{final_pnl:.2f} | Total:₹{Total:.2f} | "
                        f"DElta:{live_net_delta:.4f}"
                    )

                    e = entry_data["entry"]
                    write_to_csv([
                        entry_data["time"].strftime("%Y-%m-%d"),
                        entry_data["atm"],
                        entry_data["time"].strftime("%H:%M:%S"),
                        f"{e:.2f}",
                        entry_data["sl"],
                        Ti,
                        f"{exit_prem:.2f}",
                        Exit_Type,
                        entry_data["max_pnl"],
                        entry_data["cut"],
                        f"{final_pnl:.2f}",
                        QTY,
                        f"{Total:.2f}",
                        f"{entry_ras:.4f}",
                        f"{live_ras:.4f}",
                        "NO",
                    ])
                    entry_data          = None
                    live_greeks_history = []
                    clear_position_state()
                    TOKENS.clear()
                    sl_exit_time = now
                    continue

                # ── Trailing SL exit ─────────────────────────────────────
                if pnl <= entry_data["tsl"]:
                    Q         = QTY - entry_data["Qty_B"]
                    CE_BP     = live_ltp["CE"] + 0.5
                    PE_BP     = live_ltp["PE"] + 0.5
                    exit_prem = current_premium
                    final_pnl = (((entry_data["entry"] - exit_prem) * Q) +
                                 entry_data["pnl"]) / QTY
                    Total     = final_pnl * QTY
                    Exit_Type = "TSL"

                    deactivate_ws(entry_data["atm"])
                    Ti = now.strftime("%H:%M:%S")
                    print(
                        f"\n[SL HIT] Time:{Ti} | ExitPrem:{exit_prem:.2f} | "
                        f"PnL:{final_pnl:.2f} | Total:₹{Total:.2f} | "
                        f"RAS:{live_ras:.4f}"
                    )

                    e = entry_data["entry"]
                    write_to_csv([
                        entry_data["time"].strftime("%Y-%m-%d"),
                        entry_data["atm"],
                        entry_data["time"].strftime("%H:%M:%S"),
                        f"{e:.2f}",
                        entry_data["sl"],
                        Ti,
                        f"{exit_prem:.2f}",
                        Exit_Type,
                        entry_data["max_pnl"],
                        entry_data["cut"],
                        f"{final_pnl:.2f}",
                        QTY,
                        f"{Total:.2f}",
                        f"{entry_ras:.4f}",
                        f"{live_ras:.4f}",
                        "NO",
                    ])
                    entry_data          = None
                    live_greeks_history = []
                    clear_position_state()
                    TOKENS.clear()
                    sl_exit_time = now

        # ══════════════════════════════════════════════════════════
        # TIME-BASED EXIT
        # ══════════════════════════════════════════════════════════
        elif entry_data and now.time() >= TIME_2 and now.second == 0:
            if live_ltp["CE"] and live_ltp["PE"]:
                Q         = QTY - entry_data["Qty_B"]
                CE_BP     = live_ltp["CE"] + 0.5
                PE_BP     = live_ltp["PE"] + 0.5
                exit_prem = live_ltp["CE"] + live_ltp["PE"]
                final_pnl = (((entry_data["entry"] - exit_prem) * Q) +
                             entry_data["pnl"]) / QTY
                Total     = final_pnl * QTY
                Exit_Type = "EOD"

                deactivate_ws(entry_data["atm"])
                Ti = now.strftime("%H:%M:%S")
                print(
                    f"\n[TIME EXIT] Time:{Ti} | ExitPrem:{exit_prem:.2f} | "
                    f"PnL:{final_pnl:.2f} | Total:₹{Total:.2f} | "
                    f"RAS:{live_ras:.4f}"
                )
                e = entry_data["entry"]
                write_to_csv([
                    entry_data["time"].strftime("%Y-%m-%d"),
                    entry_data["atm"],
                    entry_data["time"].strftime("%H:%M:%S"),
                    f"{e:.2f}",
                    entry_data["sl"],
                    Ti,
                    f"{exit_prem:.2f}",
                    Exit_Type,
                    entry_data["max_pnl"],
                    entry_data["cut"],
                    f"{final_pnl:.2f}",
                    QTY,
                    f"{Total:.2f}",
                    f"{entry_ras:.4f}",
                    f"{live_ras:.4f}",
                    "NO",
                ])
                entry_data          = None
                live_greeks_history = []
                clear_position_state()
                TOKENS.clear()
                sl_exit_time = None

            print("[INFO] Market close time reached. Exiting strategy.")
            break

        time.sleep(1)

except KeyboardInterrupt:
    if entry_data:
        print("\nStopping feed...")
        deactivate_ws(entry_data["atm"])
    unsubscribe_nifty_spot_ws()    # [NEW] clean up spot subscription
    print("\nStrategy Paused")
    
