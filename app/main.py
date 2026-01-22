import os
import time
from dotenv import load_dotenv
load_dotenv("/app/.env")
DEBUG_HEARTBEAT = os.getenv("DEBUG_HEARTBEAT", "0") == "1"
if DEBUG_HEARTBEAT:
    import faulthandler, signal, threading
    faulthandler.register(signal.SIGUSR1)
    def _hb():
        while True:
            time.sleep(60)
            print(f"[HB] alive ts={int(time.time())}")
    threading.Thread(target=_hb, daemon=True).start()

import yaml
import requests
import ccxt
import json

STATE_FILE = "/app/state.json"

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        # SEED_GUARDS_FROM_ENV: fill missing guard params from ENV (do NOT override existing)
        try:
            import os
            def _env_num(key, cast, default=None):
                v = os.getenv(key)
                if v is None or str(v).strip()=="":
                    return default
                try:
                    return cast(v)
                except Exception:
                    return default

            if state.get("max_trades_per_day") is None:
                state["max_trades_per_day"] = _env_num("MAX_TRADES_PER_DAY", int, None)
            if state.get("daily_max_loss_pct") is None:
                state["daily_max_loss_pct"] = _env_num("DAILY_MAX_LOSS_PCT", float, None)
            if state.get("max_consec_losses") is None:
                state["max_consec_losses"] = _env_num("MAX_CONSEC_LOSSES", int, None)
            if state.get("cooldown_loss_min") is None:
                state["cooldown_loss_min"] = _env_num("COOLDOWN_LOSS_MIN", int, None)
            if state.get("cooldown_win_min") is None:
                state["cooldown_win_min"] = _env_num("COOLDOWN_WIN_MIN", int, None)
        except Exception:
            pass
        return state
    except Exception:
        return {}

def save_state(state: dict):
    try:
        # Preserve externally updated cooldown_until (merge with file; keep max per symbol)
        try:
            existing = load_state() or {}
            cd_file = existing.get("cooldown_until") or {}
            cd_mem  = state.get("cooldown_until") or {}
            if isinstance(cd_file, dict) and isinstance(cd_mem, dict):
                merged = dict(cd_file)
                for k, v in cd_mem.items():
                    try:
                        v = float(v)
                        prev = float(merged.get(k, 0) or 0)
                        if v > prev:
                            merged[k] = v
                    except Exception:
                        pass
                state["cooldown_until"] = merged
        except Exception:
            pass

        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def now_ts() -> int:
    return int(time.time())


def daily_loss_guard_tick(state: dict, ex, server_name: str, symbols: list[str]) -> None:
    # PER-SYMBOL HALT until tomorrow if per-symbol daily drawdown exceeds MAX_DAILY_LOSS_PCT_PER_SYMBOL
    try:
        if ex is None:
            return

        max_dd = float(env("MAX_DAILY_LOSS_PCT_PER_SYMBOL", "0.08"))
        if max_dd <= 0:
            return

        # baseline config: % of free balance, split equally across symbols, clamped
        base_pct = float(env("DAILY_GUARD_BASE_PCT", "0.30"))
        base_min = float(env("DAILY_GUARD_BASE_MIN_USD", "20"))
        base_max = float(env("DAILY_GUARD_BASE_MAX_USD", "2000"))

        day_key = time.strftime("%Y%m%d")

        dgs = state.get("daily_guard_symbols", {})
        if not isinstance(dgs, dict):
            dgs = {}

        # account free balance
        asset, free = equity_preferred(ex)
        remaining_usdt = float(free)
        # DEBUG CASCADE (AUTO_DRYRUN only)
        if env('AUTO_DRYRUN','0') == '1' and not locals().get('_dbg_cascade_printed'):
            _dbg_cascade_printed = True
            p_ = float(env('CASCADING_PCT', env('CASCADE_PCT','0.10')) or 0.10)
            r0 = float(remaining_usdt)
            a1 = r0*p_; r1 = max(0.0, r0-a1)
            a2 = r1*p_; r2 = max(0.0, r1-a2)
            a3 = r2*p_
            msg = 'DEBUG CASCADE | bal={:.2f} p={:.1f}% | a1={:.2f} rem1={:.2f} | a2={:.2f} rem2={:.2f} | a3={:.2f}'.format(float(remaining_usdt), p_*100.0, a1, r1, a2, r2, a3)
            send_telegram(msg)
        free = float(free or 0.0)
        n = max(1, len(symbols or []))

        base_total = max(0.0, free * base_pct)
        base_per = base_total / n
        if base_per < base_min:
            base_per = base_min
        if base_per > base_max:
            base_per = base_max

        def realized_pnl_today_symbol(sym: str) -> float:
            try:
                return float(realized_pnl_today(ex, sym))
            except Exception:
                return 0.0

        def unrealized_symbol(sym: str) -> float:
            try:
                p = get_position(ex, sym)
                if not p:
                    return 0.0
                u = p.get("unrealizedProfit", p.get("unRealizedProfit", 0.0))
                return float(u or 0.0)
            except Exception:
                return 0.0

        for sym in (symbols or []):
            eq0 = 0.0  # safety init
            eq0 = 0.0  # safety init
            g = dgs.get(sym, {})
            if not isinstance(g, dict):
                g = {}

            # init baseline daily OR fix legacy baseline<=0
            needs_init = (g.get("day") != day_key) or ("equity_start" not in g) or (float(g.get("equity_start", 0.0) or 0.0) <= 0.0)
            if needs_init:
                eq0 = float(base_per)
                g = {'day': day_key, 'equity_start': eq0, 'min_equity': eq0, 'halted': False}
                dgs[sym] = g
                state['daily_guard_symbols'] = dgs
                save_state(state)
                send_telegram(f"RISK | Symbol baseline set | {sym} day={day_key} eq_start={eq0:.4f} ({base_pct:.0%} balance split/clamp) | host={server_name}")

            # IMPORTANT: always read eq0 from g AFTER init
            eq0 = float(g.get("equity_start", 0.0) or 0.0)
            if eq0 <= 0.0:
                continue

            pnl = realized_pnl_today_symbol(sym) + unrealized_symbol(sym)
            eq = eq0 + float(pnl)

            # track min equity
            g["min_equity"] = float(min(float(g.get("min_equity", eq)), float(eq)))
            dgs[sym] = g
            state["daily_guard_symbols"] = dgs
            save_state(state)

            # drawdown relative to baseline magnitude
            denom = abs(eq0) if abs(eq0) > 1e-9 else 1.0
            dd = (eq0 - eq) / denom

            if dd >= max_dd and not bool(g.get("halted", False)):
                intel_set_symbol(state, sym, "HALT", 24*60, f"max_daily_loss_symbol dd={dd:.2%} pnl={pnl:.4f} base={eq0:.2f}")
                g["halted"] = True
                dgs[sym] = g
                state["daily_guard_symbols"] = dgs

                save_state(state)
                send_telegram(f"RISK | MAX DAILY LOSS HIT -> HALT {sym} 24h | dd={dd:.2%} | eq={eq:.2f}/{eq0:.2f} {asset} | host={server_name}")
    except Exception as e:
        send_telegram(f"RISK | Daily guard (per-symbol) failed | {e}")


def intel_get(state: dict) -> dict:
    # Structure:
    # state["intel"] = {
    #   "global": {"mode": "NORMAL|CAUTION|HALT", "until": 0, "reason": ""},
    #   "symbols": {"BTCUSDT": {"mode":"NORMAL|CAUTION","until":0,"reason":""}, ...},
    #   "last_update": 0
    # }
    intel = state.get("intel")
    if not isinstance(intel, dict):
        intel = {}
        state["intel"] = intel

    g = intel.get("global")
    if not isinstance(g, dict):
        g = {"mode": "NORMAL", "until": 0, "reason": ""}
        intel["global"] = g

    sy = intel.get("symbols")
    if not isinstance(sy, dict):
        sy = {}
        intel["symbols"] = sy

    if "last_update" not in intel:
        intel["last_update"] = 0

    if "last_notified" not in intel:
        intel["last_notified"] = {}  # per-symbol last notified mode

    return intel


def intel_ensure_symbols(state: dict, symbols: list[str]):
    intel = intel_get(state)
    sy = intel.get('symbols', {})
    if not isinstance(sy, dict):
        sy = {}
        intel['symbols'] = sy
    ln = intel.get('last_notified', {})
    if not isinstance(ln, dict):
        ln = {}
        intel['last_notified'] = ln
    lt = intel.get('last_notify_ts', {})
    if not isinstance(lt, dict):
        lt = {}
        intel['last_notify_ts'] = lt
    changed = False
    for s in symbols:
        if s not in sy:
            sy[s] = {'mode':'NORMAL','until':0,'reason':''}
            changed = True
        if s not in ln:
            ln[s] = 'NORMAL'
            changed = True
        if s not in lt:
            lt[s] = 0
            changed = True
    if changed:
        intel['last_update'] = now_ts()
    save_state(state)


def intel_set_global(state: dict, mode: str, minutes: int, reason: str):
    intel = intel_get(state)
    until = now_ts() + int(minutes * 60)
    intel["global"] = {"mode": mode, "until": until, "reason": reason}
    intel["last_update"] = now_ts()
    save_state(state)

def intel_set_symbol(state: dict, sym: str, mode: str, minutes: int, reason: str):
    intel = intel_get(state)
    until = now_ts() + int(minutes * 60)
    sy = intel.get("symbols", {})
    if not isinstance(sy, dict):
        sy = {}
        intel["symbols"] = sy
    sy[sym] = {"mode": mode, "until": until, "reason": reason}
    intel["last_update"] = now_ts()
    save_state(state)


def intel_notify_transition(state: dict, sym: str, new_mode: str, reason: str, until: int):
    intel = intel_get(state)
    ln = intel.get("last_notified", {})
    lt = intel.get("last_notify_ts", {})

    prev = ln.get(sym, "")
    last_ts = int(lt.get(sym, 0) or 0)
    gap = int(env("MIN_ALERT_GAP_SEC", "900"))

    # anti-spam: alert only if mode changed AND min gap passed
    if prev == new_mode:
        return
    if (now_ts() - last_ts) < gap:
        return

    ln[sym] = new_mode
    lt[sym] = now_ts()
    intel["last_notified"] = ln
    intel["last_notify_ts"] = lt
    save_state(state)

    if new_mode == "CAUTION":
        send_telegram(f"INTEL | ENTER CAUTION | {sym} until={until} reason={reason}")
    elif new_mode == "NORMAL":
        send_telegram(f"INTEL | EXIT to NORMAL | {sym}")

def intel_is_halt(state: dict) -> tuple[bool, str, int]:
    intel = intel_get(state)
    g = intel.get("global", {})
    mode = str(g.get("mode", "NORMAL"))
    until = int(g.get("until", 0) or 0)
    reason = str(g.get("reason", "") or "")

    if mode == "HALT" and until > now_ts():
        return True, reason, until
    return False, "", 0


def intel_symbol_mode(state: dict, sym: str) -> tuple[str, str, int]:
    # returns mode, reason, until
    intel = intel_get(state)
    s = intel.get("symbols", {}).get(sym, {})
    mode = str(s.get("mode", "NORMAL"))
    until = int(s.get("until", 0) or 0)
    reason = str(s.get("reason", "") or "")
    if mode in ("CAUTION", "HALT") and until > now_ts():
        return mode, reason, until
    return "NORMAL", "", 0
def pre_trade_guard(state: dict, sym: str) -> tuple[bool, float, str]:
    """
    Returns:
      allow_trade (bool),
      risk_multiplier (float),
      note (str)
    """
    # Global HALT
    halted, reason, until = intel_is_halt(state)
    if halted:
        return False, 0.0, f"HALT global until={until} reason={reason}"

    # Symbol CAUTION/HALT (v1: only CAUTION used here)
    mode, s_reason, s_until = intel_symbol_mode(state, sym)
    if mode == "HALT":
        return False, 0.0, f"HALT {sym} until={s_until} reason={s_reason}"
    if mode == "CAUTION":
        return True, float(env("CAUTION_RISK_MULT", "0.33")), f"CAUTION {sym} until={s_until} reason={s_reason}"

    return True, 1.0, "NORMAL"



def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


def send_telegram(text: str) -> None:
    # dedupe + rate limit (basic)
    try:
        print("[TG] " + str(text), flush=True)
    except Exception:
        pass
    global _TG_LAST_SENT_TEXT, _TG_LAST_SENT_TS
    try:
        _TG_LAST_SENT_TEXT
    except NameError:
        _TG_LAST_SENT_TEXT = None
        _TG_LAST_SENT_TS = 0.0
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # avoid sending same text repeatedly; and limit burst
    now = time.time()
    min_gap = float(env("TG_MIN_GAP_SEC", "2"))
    dedupe_window = float(env("TG_DEDUPE_WINDOW_SEC", "120"))
    if _TG_LAST_SENT_TEXT == text and (now - _TG_LAST_SENT_TS) < dedupe_window:
        return
    if (now - _TG_LAST_SENT_TS) < min_gap:
        return
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        if r.status_code != 200:
            print(f"[TELEGRAM] status={r.status_code} text={r.text[:200]}")
        else:
            _TG_LAST_SENT_TEXT = text
            _TG_LAST_SENT_TS = now
    except Exception as e:
        print(f"[TELEGRAM] error={e}")


def tg_get_updates(offset: int | None):
    token = env("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"timeout": 5}
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, headers={"Connection":"close"}, timeout=(5, 10))
        data = r.json()
        return data.get("result", [])
    except Exception as e:
        print(f"[TG] getUpdates error={e}")
        return []


def load_config():
    with open("/app/config/config.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def make_exchange():
    api_key = env("BINANCE_API_KEY")
    api_secret = env("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET")

    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    try:
        ex.timeout = int(env("CCXT_TIMEOUT_MS", "15000"))
    except Exception:
        pass
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] load_markets failed: {e}")
    return ex


def set_leverage(ex, symbol, lev: int):
    try:
        ex.set_leverage(lev, symbol)
    except Exception as e:
        print(f"[WARN] set_leverage failed: {e}")

def get_funding_rate(ex, symbol: str) -> float | None:
    """
    Returnează funding rate (aprox) pentru symbol.
    Dacă nu poate, returnează None.
    """
    try:
        # Binance futures endpoint: /fapi/v1/premiumIndex?symbol=BTCUSDT
        sym = symbol.replace("/", "")
        r = ex.fapiPublicGetPremiumIndex({"symbol": sym})
        fr = r.get("lastFundingRate")
        return float(fr) if fr is not None else None
    except Exception:
        return None




def equity_preferred(ex) -> tuple[str, float]:
    """Returneaza ("USD", availableBalance) din UM Futures account (v2)."""
    acc = ex.fapiPrivateV2GetAccount()
    bal = float(acc.get("availableBalance", 0.0) or 0.0)
    return "USD", bal

def cascade_alloc_usdt(free_usdt: float, step: int, pct: float = 0.10) -> float:
    """10% cascade allocation.
    step=0 => free*0.10
    step=1 => (free - step0)*0.10 (equivalent to free*(1-p)^1*p)
    etc.
    """
    try:
        free = float(free_usdt)
        p = float(pct)
        if free <= 0 or p <= 0:
            return 0.0
        if step < 0:
            step = 0
        remain = free * ((1.0 - p) ** step)
        alloc = remain * p
        return max(0.0, float(alloc))
    except Exception:
        return 0.0


def open_alloc_map(state: dict) -> dict:
    m = state.get("open_alloc_usdt") or {}
    if not isinstance(m, dict):
        m = {}
        state["open_alloc_usdt"] = m
    return m

def reserved_open_usdt(state: dict) -> float:
    m = open_alloc_map(state)
    tot = 0.0
    for v in m.values():
        try:
            tot += float(v or 0.0)
        except Exception:
            pass
    return float(tot)

def remaining_free_usdt(free: float, state: dict) -> float:
    rem = float(free or 0.0) - reserved_open_usdt(state)
    return max(0.0, rem)

def reserve_open_alloc(state: dict, sym: str, usdt: float):
    m = open_alloc_map(state)
    m[str(sym)] = float(usdt or 0.0)
    state["open_alloc_usdt"] = m

def release_open_alloc(state: dict, sym: str):
    m = open_alloc_map(state)
    m.pop(str(sym), None)
    state["open_alloc_usdt"] = m


from decimal import Decimal, ROUND_DOWN

def round_step(qty: float, step: float) -> float:
    try:
        st = Decimal(str(step))
        if st <= 0:
            return float(qty)
        q = Decimal(str(qty))
        return float((q / st).to_integral_value(rounding=ROUND_DOWN) * st)
    except Exception:
        return float(qty)

def symbol_filters(ex, sym: str):
    pair = sym if "/" in sym else sym.replace("USDT", "") + "/USDT"
    m = ex.market(pair)
    info = (m.get("info") or {})
    flt = info.get("filters") or []
    step = None
    min_qty = None
    min_notional = None
    for f in flt:
        if f.get("filterType") == "LOT_SIZE":
            try:
                step = float(f.get("stepSize") or 0) or None
            except Exception:
                step = None
            try:
                min_qty = float(f.get("minQty") or 0) or None
            except Exception:
                min_qty = None
        if f.get("filterType") == "MIN_NOTIONAL":
            mn = f.get("notional") or f.get("minNotional")
            try:
                min_notional = float(mn or 0) or None
            except Exception:
                min_notional = None
    return step, min_qty, min_notional

def normalize_qty_for_market(ex, sym: str, qty: float, px: float):
    """Return (qty_norm, ok, note)."""
    try:
        step, min_qty, min_notional = symbol_filters(ex, sym)
        q = float(qty)
        notes=[]
        if step:
            q = round_step(q, step)
        if min_qty and q < float(min_qty):
            notes.append(f"minQty={min_qty}")
        if min_notional and (float(px) * q) < float(min_notional):
            notes.append(f"minNotional={min_notional}")
        if q <= 0:
            notes.append("qty<=0")
        ok = (len(notes)==0)
        return q, ok, ("/".join(notes) if notes else "ok")
    except Exception as e:
        return float(qty), False, "VETO:filters_unavailable"

def market_price(ex, symbol) -> float:
    t = ex.fetch_ticker(symbol)
    return float(t["last"])


def place_market(ex, symbol, side: str, amount: float):
    """Market order with amount precision + min notional guard."""
    # ensure markets
    try:
        ex.load_markets()
    except Exception:
        pass

    # get last price (best-effort)
    px = 0.0
    try:
        t = ex.fetch_ticker(symbol)
        px = float(t.get("last") or t.get("close") or 0.0)
    except Exception:
        px = 0.0

    amt = float(amount)

    # amount precision
    try:
        amt = float(ex.amount_to_precision(symbol, amt))
    except Exception:
        amt = float(amt)

    # min notional / min cost
    try:
        m = ex.market(symbol)
        min_cost = ((m.get("limits") or {}).get("cost") or {}).get("min")
        if px > 0 and min_cost and (px * amt) < float(min_cost):
            need = float(min_cost) / float(px)
            try:
                need = float(ex.amount_to_precision(symbol, need))
            except Exception:
                pass
            # bump by one precision increment if rounding went under
            if (px * need) < float(min_cost):
                prec = (m.get("precision") or {}).get("amount")
                if isinstance(prec, int):
                    need = need + (10 ** (-prec))
                    try:
                        need = float(ex.amount_to_precision(symbol, need))
                    except Exception:
                        pass
            amt = max(amt, need)
    except Exception:
        pass

    return ex.create_order(symbol, "market", side, amt)

def get_position(ex, symbol: str):
    sym = symbol.replace("/", "")
    pr = ex.fapiPrivateV2GetPositionRisk({"symbol": sym})
    if not pr:
        return None
    return pr[0]

def realized_pnl_since(ex, sym: str, start_ms: int, end_ms: int) -> float:
    """Sum realized pnl (and fees) for sym between start_ms and end_ms using fapi income history."""
    try:
        symbol = sym.replace("/", "")
        params = {"symbol": symbol, "startTime": int(start_ms), "endTime": int(end_ms), "limit": 1000}
        rows = ex.fapiPrivateGetIncome(params)
        pnl = 0.0
        fee = 0.0
        for r in rows or []:
            t = str(r.get("incomeType", "") or "")
            inc = float(r.get("income", 0) or 0)
            if t == "REALIZED_PNL":
                pnl += inc
            elif t in ("COMMISSION",):
                fee += inc
        # return net pnl (pnl + fee) because fee is negative
        return float(pnl + fee)
    except Exception:
        return 0.0


def atr_simple(ex, symbol_usdt: str, tf: str = "15m", period: int = 14) -> float | None:
    """ATR simplu (SMA TR) pe UM Futures. Returnează ATR în unități de preț sau None."""
    try:
        base = symbol_usdt.replace("USDT", "")
        pair = f"{base}/USDT" if "/" not in symbol_usdt else symbol_usdt
        o = ex.fetch_ohlcv(pair, timeframe=tf, limit=period + 2)
        if not o or len(o) < period + 1:
            return None
        # ohlcv: [ts, open, high, low, close, vol]
        trs = []
        prev_close = None
        for c in o[-(period+1):]:
            h = float(c[2]); l = float(c[3]); cl = float(c[4])
            if prev_close is None:
                tr = h - l
            else:
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = cl
        if len(trs) < period:
            return None
        # SMA TR
        atr = sum(trs[-period:]) / float(period)
        return float(atr) if atr > 0 else None
    except Exception:
        return None


def sltp_hybrid_triggers(ex, symbol: str, side_open: str, entry: float, sl_pct: float, tp_pct: float) -> tuple[float, float, str]:
    """Returnează (sl_trigger, tp_trigger, mode_used)."""
    # pct input is in PERCENT (0.30 = 0.30%)
    slp = float(sl_pct) / 100.0
    tpp = float(tp_pct) / 100.0

    mode = str(env("SLTP_MODE", "PCT")).upper()  # PCT | ATR | HYBRID
    atr_tf = str(env("ATR_TF", "15m"))
    atr_period = int(env("ATR_PERIOD", "14"))
    atr_mult_sl = float(env("ATR_MULT_SL", "2.0"))
    atr_mult_tp = float(env("ATR_MULT_TP", "3.0"))

    used = "PCT"

    atr = None
    if mode in ("ATR", "HYBRID"):
        atr = atr_simple(ex, symbol, atr_tf, atr_period)

    # compute distance
    if mode == "ATR" and atr is None:
        # forced ATR but unavailable -> fallback to pct anyway (safety)
        mode_eff = "PCT"
    elif mode == "ATR" and atr is not None:
        mode_eff = "ATR"
    elif mode == "HYBRID" and atr is not None:
        mode_eff = "ATR"
    else:
        mode_eff = "PCT"

    if mode_eff == "ATR":
        sl_dist = float(atr) * atr_mult_sl
        tp_dist = float(atr) * atr_mult_tp
        used = f"ATR(tf={atr_tf},n={atr_period},sl={atr_mult_sl}x,tp={atr_mult_tp}x)"
    else:
        sl_dist = float(entry) * slp
        tp_dist = float(entry) * tpp
        used = "PCT"

    if side_open == "buy":
        sl_trig = float(entry) - sl_dist
        tp_trig = float(entry) + tp_dist
    else:
        sl_trig = float(entry) + sl_dist
        tp_trig = float(entry) - tp_dist

    return sl_trig, tp_trig, used


def place_sl_tp(ex, symbol: str, side_open: str, qty: float, entry: float, sl_pct: float, tp_pct: float):
    """
    Plaseaza SL/TP ca *Algo conditional orders* (Binance USDS-M Futures: POST /fapi/v1/algoOrder).
    side_open: 'buy' (LONG) or 'sell' (SHORT)
    sl_pct / tp_pct in PERCENT (ex: 0.30 = 0.30%)
    HYBRID: ATR (optional) + fallback to %.
    """
    sl_trig, tp_trig, used = sltp_hybrid_triggers(ex, symbol, side_open, entry, sl_pct, tp_pct)

    if side_open == "buy":
        close_side = "SELL"
    else:
        close_side = "BUY"
    # precision-safe using exchange market rules
    try:
        sl_trig = float(ex.price_to_precision(symbol, sl_trig))
        tp_trig = float(ex.price_to_precision(symbol, tp_trig))
        qty = float(ex.amount_to_precision(symbol, qty))
    except Exception:
        # fallback
        sl_trig = float(sl_trig)
        tp_trig = float(tp_trig)
        qty = float(qty)

    sym = symbol.replace("/", "")

    def algo_cond(order_type: str, trigger: float):
        params = {
            "algoType": "CONDITIONAL",
            "symbol": sym,
            "side": close_side,
            "type": order_type,               # STOP_MARKET / TAKE_PROFIT_MARKET
            "triggerPrice": ex.price_to_precision(symbol, trigger),
            "workingType": "MARK_PRICE",
            "closePosition": "true",
            "priceProtect": "false",
        }
        # ccxt generic request to fapiPrivate
        return ex.request("algoOrder", "fapiPrivate", "POST", params)

    # SL + TP
    # SL + TP (idempotent-ish): if already exists, ignore (-4130)
    try:
        algo_cond("STOP_MARKET", sl_trig)
    except Exception as e:
        if "-4130" not in str(e):
            raise
    try:
        algo_cond("TAKE_PROFIT_MARKET", tp_trig)
    except Exception as e:
        if "-4130" not in str(e):
            raise

    return sl_trig, tp_trig, used



def algo_open_orders(ex, symbol: str = None):
    """
    Binance USDS-M Futures: GET /fapi/v1/openAlgoOrders
    ccxt: ex.request("openAlgoOrders","fapiPrivate","GET", params)
    """
    params = {}
    if symbol:
        params["symbol"] = symbol.replace("/", "")
    try:
        return ex.request("openAlgoOrders", "fapiPrivate", "GET", params) or []
    except Exception:
        # fallback: some ccxt builds expose it as fapiPrivateGetAlgoOpenOrders
        try:
            fn = getattr(ex, "fapiPrivateGetAlgoOpenOrders", None)
            if fn:
                return (fn(params) or [])
        except Exception:
            pass
        return []

def algo_has_sltp(ex, symbol: str) -> bool:
    """
    True if we see BOTH STOP_MARKET and TAKE_PROFIT_MARKET CONDITIONAL algo orders for this symbol.
    NOTE: Binance openAlgoOrders returns keys: algoType, orderType (not 'type').
    """
    oo = algo_open_orders(ex, symbol)
    need = {"STOP_MARKET": False, "TAKE_PROFIT_MARKET": False}
    sym = symbol.replace("/", "")
    for o in (oo or []):
        try:
            if str(o.get("symbol","")) != sym:
                continue
            at = str(o.get("algoType","") or "")
            if at and at != "CONDITIONAL":
                continue
            t = str(o.get("orderType") or o.get("type") or "")
            if t in need:
                need[t] = True
        except Exception:
            continue
    return all(need.values())


def sltp_guard_tick(state: dict, ex, sym: str, sl_pct: float, tp_pct: float, last_sl_tp: dict, guard_last_ts: dict) -> None:
    # Re-apply SL/TP if position exists and last SL/TP is missing/stale (ccxt cannot list algo orders reliably here)
    now = time.time()
    every = int(env("SLTP_GUARD_EVERY_SEC", "30"))
    cd = int(env("SLTP_GUARD_COOLDOWN_SEC", "20"))
    lt = float(guard_last_ts.get(sym, 0) or 0)
    if (now - lt) < every:
        return
    guard_last_ts[sym] = now

    try:
        p = get_position(ex, sym)
        amt = float(p.get("positionAmt", 0.0)) if p else 0.0
        entry = float(p.get("entryPrice", 0.0)) if p else 0.0
        if amt == 0.0 or entry == 0.0:
            # if symbol is flat, prune stale marker
            try:
                so = state.get('sltp_ok', {})
                if isinstance(so, dict) and sym in so:
                    so.pop(sym, None)
                state['sltp_ok'] = so
                save_state(state)
            except Exception:
                pass
            return

        # ENSURE MODE (ccxt listing for conditional/algo orders is unreliable):
        # if we haven't verified recently, just attempt to (re)place SL/TP (idempotent via -4130 ignore in place_sl_tp)
        ok_ttl = int(env("SLTP_OK_TTL_SEC", "600"))  # 10 min default
        so = state.get("sltp_ok", {}) or {}
        if not isinstance(so, dict):
            so = {}
        last_ok = int(so.get(sym, 0) or 0)
        if last_ok and (int(time.time()) - last_ok) < ok_ttl:
            return

        side_open = "buy" if amt > 0 else "sell"
        qty = abs(amt)

        sl, tp, used = place_sl_tp(ex, sym, side_open, qty, entry, sl_pct, tp_pct)

        so[sym] = int(time.time())
        state["sltp_ok"] = so
        save_state(state)
        send_telegram(f"SAFETY | SLTP GUARD VERIFIED | {sym} entry={entry} SL={sl} TP={tp} | mode={used}")

    except Exception as e:
        send_telegram(f"SAFETY | SLTP GUARD FAILED | {sym} | {e}")
def ema(values, period: int):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def trend_ema(ex, symbol: str, tf: str = "1h") -> str:
    # returns BULL / BEAR / FLAT
    o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=80)
    closes = [c[4] for c in o if c and len(c) >= 5]
    if len(closes) < 60:
        return "FLAT"
    e20 = ema(closes[-60:], 20)
    e50 = ema(closes[-60:], 50)
    if e20 is None or e50 is None:
        return "FLAT"
    if e20 > e50:
        return "BULL"
    if e20 < e50:
        return "BEAR"
    return "FLAT"



# =========================
# ANTI-CHOP helpers (v1)
# =========================

def adx_simple(ex, symbol: str, tf: str = "1h", period: int = 14):
    """Return ADX (0..100) or None. Wilder smoothing, classic ADX."""
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=max(200, period * 6))
        if not o or len(o) < (period * 3 + 5):
            return None

        # ohlcv: [ts, open, high, low, close, vol]
        highs  = [float(c[2]) for c in o if c and len(c) >= 6]
        lows   = [float(c[3]) for c in o if c and len(c) >= 6]
        closes = [float(c[4]) for c in o if c and len(c) >= 6]
        n = min(len(highs), len(lows), len(closes))
        highs, lows, closes = highs[-n:], lows[-n:], closes[-n:]
        if n < (period * 3 + 5):
            return None

        tr = []
        pdm = []
        ndm = []
        for i in range(1, n):
            up = highs[i] - highs[i-1]
            dn = lows[i-1] - lows[i]
            pdm.append(up if (up > dn and up > 0) else 0.0)
            ndm.append(dn if (dn > up and dn > 0) else 0.0)

            tr_i = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            tr.append(float(tr_i))

        if len(tr) < (period + 5):
            return None

        # Wilder smoothing: RMA
        def rma(values, length):
            out=[]
            s=sum(values[:length])
            out.append(s)
            for v in values[length:]:
                s = s - (s/length) + v
                out.append(s)
            return out

        tr_s  = rma(tr, period)
        pdm_s = rma(pdm, period)
        ndm_s = rma(ndm, period)

        # Align lengths
        L = min(len(tr_s), len(pdm_s), len(ndm_s))
        tr_s, pdm_s, ndm_s = tr_s[-L:], pdm_s[-L:], ndm_s[-L:]

        pdi = []
        mdi = []
        dx  = []
        for t, p, m_ in zip(tr_s, pdm_s, ndm_s):
            if t <= 0:
                pdi.append(0.0); mdi.append(0.0); dx.append(0.0); continue
            pdi_v = 100.0 * (p / t)
            mdi_v = 100.0 * (m_ / t)
            # ensure non-negative
            if pdi_v < 0: pdi_v = 0.0
            if mdi_v < 0: mdi_v = 0.0
            denom = (pdi_v + mdi_v)
            dx_v = 0.0 if denom == 0 else (100.0 * abs(pdi_v - mdi_v) / denom)
            # clamp to [0,100]
            if dx_v < 0: dx_v = 0.0
            if dx_v > 100: dx_v = 100.0
            pdi.append(pdi_v); mdi.append(mdi_v); dx.append(dx_v)

        if len(dx) < (period + 2):
            return None

        adx_s = rma(dx, period)
        adx = adx_s[-1] / period  # because our rma uses "sum-style" smoothing
        # clamp
        if adx < 0: adx = 0.0
        if adx > 100: adx = 100.0
        return float(adx)
    except Exception:
        return None

def bb_width(ex, symbol: str, tf: str = "15m", period: int = 20, mult: float = 2.0):
    """Return Bollinger Band width normalized by mid ( (upper-lower)/mid )."""
    try:
        import math
        o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=period + 60)
        if not o or len(o) < period + 5:
            return None
        closes = [c[4] for c in o if c and len(c) >= 5]
        if len(closes) < period + 5:
            return None

        def sma(vals):
            return sum(vals)/len(vals) if vals else None

        def stdev(vals):
            if not vals:
                return None
            m = sum(vals)/len(vals)
            var = sum((x-m)*(x-m) for x in vals)/len(vals)
            return math.sqrt(var)

        window = closes[-period:]
        mid = sma(window)
        sd = stdev(window)
        if mid is None or sd is None or mid == 0:
            return None
        upper = mid + mult * sd
        lower = mid - mult * sd
        width = (upper - lower) / mid
        return float(width)
    except Exception:
        return None

def ema_slope(ex, symbol: str, tf: str = "1h", period: int = 20, lookback: int = 5):
    """Return EMA slope as pct over lookback candles (ema_now - ema_prev) / ema_prev."""
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=max(80, period + lookback + 30))
        if not o or len(o) < (period + lookback + 5):
            return None
        closes = [c[4] for c in o if c and len(c) >= 5]
        if len(closes) < (period + lookback + 5):
            return None

        # reuse ema(values, period) helper already defined above
        # compute EMA series by sliding window (lightweight enough at our limits)
        def ema_series(vals, period):
            if len(vals) < period:
                return []
            k = 2 / (period + 1)
            e = vals[0]
            out = [e]
            for v in vals[1:]:
                e = v * k + e * (1 - k)
                out.append(e)
            return out

        # use last ~60 closes to stabilize
        tail = closes[-(60 + lookback):]
        es = ema_series(tail, period)
        if len(es) < (lookback + 2):
            return None
        ema_now = float(es[-1])
        ema_prev = float(es[-1 - lookback])
        if ema_prev == 0:
            return None
        slope_pct = (ema_now - ema_prev) / ema_prev
        return float(slope_pct)
    except Exception:
        return None

def anti_chop_check(ex, symbol: str):
    """
    Returns: (allow:bool, reason:str, metrics:dict)

    Veto logic:
    - ADX(1H) too low => chop/range
    - BBWidth(15m) too low => compression
    - EMA slope(1H) too low => no direction
    """
    try:
        adx_min = float(env("CHOP_ADX_MIN", "18"))
        bbw_min = float(env("CHOP_BBWIDTH_MIN", "0.012"))
        slope_min = float(env("CHOP_EMA_SLOPE_MIN", "0.0005"))

        adx = adx_simple(ex, symbol.replace("USDT","/USDT") if "/" not in symbol else symbol, "1h", 14)
        bbw = bb_width(ex, symbol.replace("USDT","/USDT") if "/" not in symbol else symbol, "15m", 20, 2.0)
        slp = ema_slope(ex, symbol.replace("USDT","/USDT") if "/" not in symbol else symbol, "1h", 20, 5)

        metrics = {
            "adx_1h": None if adx is None else float(adx),
            "bbw_15m": None if bbw is None else float(bbw),
            "slope_1h": None if slp is None else float(slp),
            "thr_adx": adx_min,
            "thr_bbw": bbw_min,
            "thr_slope": slope_min,
        }

        # If indicators unavailable -> allow (do NOT block trading due to data hiccup)
        if adx is None or bbw is None or slp is None:
            return True, "OK(indicators_unavailable)", metrics

        # gates
        if float(adx) < adx_min:
            return False, f"CHOP_VETO adx_1h<{adx_min}", metrics
        if float(bbw) < bbw_min:
            return False, f"CHOP_VETO bbw_15m<{bbw_min}", metrics
        if abs(float(slp)) < slope_min:
            return False, f"CHOP_VETO ema_slope_1h<{slope_min}", metrics

        return True, "OK", metrics
    except Exception as e:
        # fail-open
        return True, f"OK(err:{e})", {}
def count_open_conditional(ex, sym_usdt: str) -> int:
    """UM Futures conditional (SL/TP) orders as shown in Binance UI (openAlgoOrders).
    Returns count of CONDITIONAL algo orders that are STOP_MARKET or TAKE_PROFIT_MARKET for given symbol.
    """
    sym = (sym_usdt or '').replace('/','')
    try:
        r = ex.request('openAlgoOrders', 'fapiPrivate', 'GET', {'symbol': sym})
        if not isinstance(r, list):
            r = ex.request('openAlgoOrders', 'fapiPrivate', 'GET', {})
        if not isinstance(r, list):
            return 0
        n = 0
        for o in (r or []):
            if str(o.get('symbol','')) != sym:
                continue
            at = str(o.get('algoType','') or '')
            if at and at != 'CONDITIONAL':
                continue
            ot = str(o.get('orderType') or o.get('type') or '')
            if ot in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
                n += 1
        return n
    except Exception:
        try:
            r = ex.request('openAlgoOrders', 'fapiPrivate', 'GET', {})
            n = 0
            for o in (r or []):
                if str(o.get('symbol','')) != sym:
                    continue
                at = str(o.get('algoType','') or '')
                if at and at != 'CONDITIONAL':
                    continue
                ot = str(o.get('orderType') or o.get('type') or '')
                if ot in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
                    n += 1
            return n
        except Exception:
            return -1
def pullback_entry_ok(ex, sym_usdt: str, side: str):
    """
    Swing entry: pullback to EMA on 15m + confirmation candle close in trend direction.
    Works with ccxt fetch_ohlcv.

    LONG:
      - last closed candle ([-2]) low touches/breaches EMA within tolerance
      - last closed candle closes ABOVE EMA
      - optional: candle body direction (close >= open)

    SHORT:
      - last closed candle high touches/breaches EMA within tolerance
      - last closed candle closes BELOW EMA
      - optional: candle body direction (close <= open)

    Returns: (ok:bool, reason:str, metrics:dict)
    """
    try:
        if side not in ("buy","sell"):
            return False, "PULLBACK_BAD_SIDE", {}

        tf = env("PULLBACK_TF", "15m")
        period = int(env("PULLBACK_EMA_PERIOD", "20"))
        tol = float(env("PULLBACK_TOL_PCT", "0.002"))   # 0.2%
        require_body = env("PULLBACK_REQUIRE_BODY", "1") == "1"

        # normalize symbol for ccxt (BTCUSDT -> BTC/USDT)
        pair = sym_usdt.replace("USDT","/USDT") if "/" not in sym_usdt else sym_usdt

        # Need enough candles to stabilize EMA
        lim = max(80, period + 30)
        o = ex.fetch_ohlcv(pair, timeframe=tf, limit=lim)
        if not o or len(o) < (period + 5):
            return True, "PULLBACK_OK(indicators_unavailable)", {"tf": tf}

        # last CLOSED candle
        c = o[-2]
        _ts, opn, high, low, close, vol = c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])

        closes = [float(x[4]) for x in o if x and len(x) >= 5]
        e = ema(closes[-(period+30):], period)
        if e is None or e == 0:
            return True, "PULLBACK_OK(indicators_unavailable)", {"tf": tf}

        ema15 = float(e)
        # touch logic with tolerance band
        up_band = ema15 * (1.0 + tol)
        dn_band = ema15 * (1.0 - tol)

        if side == "buy":
            touched = (low <= up_band) and (low >= dn_band)  # wick touches EMA band
            confirmed = (close > ema15)
            body_ok = (close >= opn) if require_body else True
        else:
            touched = (high >= dn_band) and (high <= up_band)
            confirmed = (close < ema15)
            body_ok = (close <= opn) if require_body else True

        ok = bool(touched and confirmed and body_ok)
        metrics = {
            "tf": tf,
            "ema15": ema15,
            "tol": tol,
            "o": opn, "h": high, "l": low, "c": close,
            "touched": int(touched),
            "confirmed": int(confirmed),
            "body_ok": int(body_ok),
        }
        if ok:
            return True, "PULLBACK_OK", metrics
        return False, "PULLBACK_WAIT", metrics

    except Exception as e:
        # fail-open (do not block on errors)
        return True, f"PULLBACK_OK(err:{e})", {}

def main():
    cfg = load_config()
    mode = (cfg.get("mode") or "SAFE").upper()
    hb = int(cfg.get("logic", {}).get("heartbeat_sec", 60))

    # risk config (percent)
    risk = cfg.get("risk", {}) or {}
    sl_pct = float(risk.get("sl_pct", 0.30))
    tp_pct = float(risk.get("tp_pct", 0.10))

    symbols = [s.strip().upper() for s in env("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
    symbol = symbols[-1]  # simbolul implicit pentru comenzile manuale (/buy /sell /close)
    lev = int(env("LEVERAGE", "3"))

    server_name = env("SERVER_NAME", env("HOSTNAME", os.uname().nodename))
    live_enabled = env("LIVE_ENABLED", "NO").upper() == "YES"
    live_max_usdt = float(env("LIVE_MAX_USDT", "0"))  # 0 => AUTO (cap = available/remaining)
    state = load_state()
    intel_ensure_symbols(state, symbols)
    # init state file if missing/empty, using env defaults
    if not isinstance(state, dict) or not state:
        state = {
            "auto_enabled": False,
            "live_enabled": live_enabled,
        }
        save_state(state)

    auto_enabled = bool(state.get("auto_enabled", False))
    live_enabled = bool(state.get("live_enabled", live_enabled))

    print(f"[START] host={server_name} mode={mode} symbols={symbols} lev={lev} live_enabled={live_enabled} max_usdt={live_max_usdt}")
    send_telegram(f"BOT STARTED | host={server_name} | mode={mode} | symbols={symbols} | lev={lev} | LIVE_ENABLED={live_enabled}")
    send_telegram("Comenzi: /status | /analyze | /buy | /sell | /close | /live_on | /live_off | /auto_on | /auto_off | /health")

    ex = None
    if mode == "LIVE":
        ex = make_exchange()
        for s in symbols:
            set_leverage(ex, s, lev)

    # Telegram offset persisted (prevents replay after restart)
    offset = int(state.get("tg_offset", 0) or 0) or None
    n = 0
    # AUTO state per symbol
    auto_last_trade_ts = {s: 0 for s in symbols}
    auto_trades_today = {s: 0 for s in symbols}
    loss_streak = {s: 0 for s in symbols}      # consecutive losses per symbol
    cooldown_until = state.get('cooldown_until', {}) or {}  # sym -> unix seconds
    auto_blocked = {s: False for s in symbols} # block symbol until tomorrow after 2 losses
    cascade_step = {s: 0 for s in symbols}     # Modul 2: 10% cascade per position (0..)

    # GLOBAL risk (Modul 3)
    trades_today_global = int(state.get("trades_today_global", 0) or 0)
    last_sl_tp = {s: None for s in symbols}    # (side_open, sl, tp)
    last_pos_amt = {s: 0.0 for s in symbols}
    trade_open_ms = {s: 0 for s in symbols}  # track position open time (ms) for realized pnl calc
    guard_last_ts = {s: 0.0 for s in symbols}  # SL/TP guardian tick timer per symbol
    auto_day = time.strftime("%Y%m%d")

    # INIT: seed last_pos_amt + trade_open_ms from existing positions (after restart)
    if mode == "LIVE" and ex is not None:
        try:
            for s in symbols:
                p0 = get_position(ex, s)
                a0 = float(p0.get("positionAmt", 0.0)) if p0 else 0.0
                if a0 != 0.0:
                    last_pos_amt[s] = a0
                    # best-effort open time from exchange updateTime
                    ut = int(p0.get("updateTime", 0) or 0)
                    trade_open_ms[s] = ut if ut > 0 else int(time.time()*1000)
            send_telegram(f"INIT | seeded positions after restart | open={[(k, float(last_pos_amt[k])) for k in symbols if float(last_pos_amt[k])!=0.0]}" )
            # Boot prune: keep sltp_ok only for open positions
            open_syms = {k for k in symbols if float(last_pos_amt.get(k, 0.0) or 0.0) != 0.0}
            so = state.get('sltp_ok', {})
            if isinstance(so, dict):
                for k in list(so.keys()):
                    if k not in open_syms:
                        so.pop(k, None)
            state['sltp_ok'] = so
            save_state(state)
        except Exception as e:
            send_telegram(f"INIT | seed positions failed | {e}")

    auto_last_info_ts = {s: 0 for s in symbols}
    auto_last_health_ts = 0  # global auto health timer
    AUTO_INFO_EVERY_SEC = int(env("AUTO_INFO_EVERY_SEC", "600"))
    AUTO_HEALTH_EVERY_SEC = int(env("AUTO_HEALTH_EVERY_SEC", "3600"))
    # Only send AUTO WAIT/HOLD/VETO info for these symbols (comma-separated). Example: "SOLUSDT"
    AUTO_INFO_SYMBOLS = env("AUTO_INFO_SYMBOLS", "SOLUSDT")
    info_symbols = {x.strip().upper() for x in AUTO_INFO_SYMBOLS.split(",") if x.strip()}

    AUTO_COOLDOWN_SEC = 15 * 60
    AUTO_MAX_TRADES_PER_DAY = int(env('MAX_TRADES_PER_DAY', 4))


    # INTEL auto-check timers
    last_funding_check_ts = 0
    FUNDING_CHECK_EVERY_SEC = int(env("FUNDING_CHECK_EVERY_SEC", "60"))

    while True:
        n += 1

        # RISK: daily max loss guard (global HALT)
        if mode == "LIVE" and ex is not None and live_enabled:
            daily_loss_guard_tick(state, ex, server_name, symbols)

        # INTEL: funding extreme guard (auto CAUTION)
        if mode == "LIVE" and ex is not None and live_enabled:
            if (time.time() - last_funding_check_ts) >= FUNDING_CHECK_EVERY_SEC:
                last_funding_check_ts = time.time()
                try:
                    pos_thr = float(env("FUNDING_EXTREME_POS", "0.01"))
                    neg_thr = float(env("FUNDING_EXTREME_NEG", "-0.01"))
                    cau_min = int(env("FUNDING_CAUTION_MINUTES", "45"))
                    for sym in symbols:
                        fr = get_funding_rate(ex, sym.replace("USDT", "") + "/USDT" if "/" not in sym else sym)
                        ov = env(f"FUNDING_OVERRIDE_{sym}", "")
                        if ov:
                            try: fr = float(ov)
                            except: pass
                        if fr is None:
                            continue
                        if fr >= pos_thr or fr <= neg_thr:
                            reason = f"funding_extreme {fr:.6f}"
                            # anti-spam: notify only when CAUTION is not already active
                            prev_mode, _, prev_until = intel_symbol_mode(state, sym)
                            already = (prev_mode == "CAUTION" and prev_until > now_ts())
                            if not already:
                                intel_set_symbol(state, sym, "CAUTION", cau_min, reason)
                                # notify transition (enter CAUTION once)
                                _, r, u = intel_symbol_mode(state, sym)
                                intel_notify_transition(state, sym, "CAUTION", r, u)
                            else:
                                # refresh CAUTION silently
                                intel_set_symbol(state, sym, "CAUTION", cau_min, reason)
                        else:
                            # if previously in CAUTION and funding normalized -> EXIT to NORMAL
                            prev_mode, _, prev_until = intel_symbol_mode(state, sym)
                            if prev_mode == "CAUTION" and prev_until > now_ts():
                                intel_set_symbol(state, sym, "NORMAL", 0, "")
                                intel_notify_transition(state, sym, "NORMAL", "", 0)
                except Exception as e:
                    send_telegram(f"INTEL AUTO FAILED | {e}")
        # SAFETY LOOP (runs even when AUTO is OFF): ensure SL/TP exists for any open (seeded) positions
        if mode == "LIVE" and ex is not None and live_enabled:
            try:
                for _sym in symbols:
                    _p = get_position(ex, _sym)
                    _amt = float(_p.get("positionAmt", 0.0)) if _p else 0.0
                    if _amt != 0.0:
                        sltp_guard_tick(state, ex, _sym, sl_pct, tp_pct, last_sl_tp, guard_last_ts)
            except Exception as _e:
                send_telegram(f"SAFETY | SLTP GUARD LOOP FAILED | {_e}")

# AUTO ENGINE (simple v1): trade only when LIVE_ENABLED + AUTO_ENABLED and no open position
        # Limits: cooldown + max trades/day
        day_now = time.strftime("%Y%m%d")
        if day_now != auto_day:
            auto_day = day_now
            for s in symbols:
                auto_trades_today[s] = 0
                loss_streak[s] = 0
                auto_blocked[s] = False
                last_sl_tp[s] = None
                last_pos_amt[s] = 0.0

        if mode == "LIVE" and ex is not None and live_enabled and auto_enabled:


            try:
                # market regime filter (SOL-only)
                sol_1h = trend_ema(ex, "SOL/USDT", "1h")
                btc_1h = trend_ema(ex, "BTC/USDT", "1h")
                eth_1h = trend_ema(ex, "ETH/USDT", "1h")

                if sol_1h == "BULL":
                    market_bias = "LONG"
                elif sol_1h == "BEAR":
                    market_bias = "SHORT"
                else:
                    market_bias = "NO TRADE"
                # Global max-trades gate disabled (per-symbol only)

                # AUTO HEALTH (global, low spam) - without legacy daily_guard global
                if (time.time() - auto_last_health_ts) >= AUTO_HEALTH_EVERY_SEC:
                    try:
                        intel = intel_get(state)
                        g = intel.get("global", {})
                        gmode = g.get("mode","NORMAL")
                        guntil = int(g.get("until",0) or 0)
                        sltp = state.get("sltp_ok", {})
                        sol_ok = int(sltp.get("SOLUSDT",0) or 0)
                        should_send = ((gmode != "NORMAL") or (sol_ok == 0))
                        if should_send:
                            send_telegram(
                                f"AUTO HEALTH | intel={gmode} until={guntil} | sltp_ok SOL={sol_ok}"
                            )
                    except Exception as e:
                        send_telegram(f"AUTO HEALTH FAILED | {e}")
                    auto_last_health_ts = time.time()

                # CASCADE (global per AUTO cycle): free0 read once; each opened position uses % of remaining
                try:
                    _a, _free = equity_preferred(ex)
                    cascade_free0 = float(_free or 0.0)
                except Exception:
                    cascade_free0 = 0.0
                cascade_used = 0.0
                cascade_used = reserved_open_usdt(state)
                cascade_open_idx = 0

                for sym in symbols:
                    if auto_blocked.get(sym, False):
                        continue
                    # AUTO COOLDOWN (state.cooldown_until)
                    try:
                        cd = (load_state().get("cooldown_until") or {})
                        until = float(cd.get(sym, 0) or 0)
                        now = time.time()
                        if until and now < until:
                            left = int(until - now)
                            if (now - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC:
                                send_telegram(f"AUTO COOLDOWN | {sym} | left={left}s")
                                auto_last_info_ts[sym] = now
                            continue
                    except Exception:
                        pass

                    if auto_trades_today[sym] >= AUTO_MAX_TRADES_PER_DAY:
                        continue
                    if (time.time() - auto_last_trade_ts[sym]) < AUTO_COOLDOWN_SEC:
                        continue

                    p = get_position(ex, sym)
                    amt = float(p.get("positionAmt", 0.0)) if p else 0.0
                    # SLTP conditional guard (AUTO): if pos open but conditional orders != 2 => CAUTION 60m
                    try:
                        if amt != 0.0:
                            n_cond = count_open_conditional(ex, sym)
                            if n_cond >= 0 and n_cond != 2:
                                intel_set_symbol(state, sym, 'CAUTION', 60, f'sltp_guard cond_open={n_cond}')
                                send_telegram(f"INTEL | CAUTION {sym} 60 min | sltp_guard cond_open={n_cond}")
                                continue
                    except Exception:
                        pass

                    # SAFETY: ensure SL/TP exists (guardian)
                    # (moved to SAFETY LOOP) sltp_guard_tick(state, ex, sym, sl_pct, tp_pct, last_sl_tp, guard_last_ts)

                    # detect position close -> update loss streak / block symbol
                    prev_amt = float(last_pos_amt.get(sym, 0.0))

                    # detect position open -> set open timestamp for realized pnl calc
                    if prev_amt == 0.0 and amt != 0.0:
                        trade_open_ms[sym] = int(time.time() * 1000)
                    if prev_amt != 0.0 and amt == 0.0:
                        info = last_sl_tp.get(sym)
                        if info:
                            side_open, sl, tp = info
                            now_ms = int(time.time() * 1000)
                            start_ms = int(trade_open_ms.get(sym, 0) or (now_ms - 6*60*60*1000))
                            pnl = realized_pnl_since(ex, sym, start_ms, now_ms)
                            px_now = market_price(ex, sym)
                            is_loss = pnl < 0

                            if is_loss:
                                loss_streak[sym] = loss_streak.get(sym, 0) + 1
                                # global loss streak disabled
                                send_telegram(f"AUTO RESULT | {sym} LOSS | loss_streak={loss_streak[sym]}/2 | pnl={pnl:.4f} | px={px_now}")
                                # cooldown after LOSS (escalating 30/{int(env('MAX_TRADES_PER_DAY',4))}/120 min) persisted in state.cooldown_until
                                try:
                                    ls = int(loss_streak.get(sym, 0) or 0)
                                    cd1 = int(env('COOLDOWN_AFTER_LOSS_MIN', '30'))
                                    cd2 = int(env('COOLDOWN_AFTER_LOSS_MIN_2', '60'))
                                    cd3 = int(env('COOLDOWN_AFTER_LOSS_MIN_3', '120'))
                                    cd_min = cd1 if ls <= 1 else (cd2 if ls == 2 else cd3)
                                    cdmap = (state.get('cooldown_until') or {})
                                    cdmap[sym] = time.time() + (cd_min * 60)
                                    state['cooldown_until'] = cdmap
                                    save_state(state)
                                    send_telegram(f"AUTO COOLDOWN SET | {sym} | loss_streak={ls} | cd_min={cd_min}")
                                except Exception as e:
                                    send_telegram(f"AUTO COOLDOWN SET FAILED | {sym} | {e}")

                                if loss_streak[sym] >= 2:
                                    # keep trading enabled tomorrow; escalation handled by cooldown (cd_min uses _2/_3)
                                    send_telegram(f"AUTO COOLDOWN ESCALATE | {sym} | loss_streak={loss_streak[sym]} (no tomorrow-block)")
                            else:
                                loss_streak[sym] = 0
                                # global loss streak reset disabled
                                send_telegram(f"AUTO RESULT | {sym} WIN | loss_streak reset | pnl={pnl:.4f} | px={px_now}")
                                # cooldown after WIN (default 10 min) persisted in state.cooldown_until
                                try:
                                    cdw = int(env('COOLDOWN_AFTER_WIN_MIN', '10'))
                                    if cdw > 0:
                                        cdmap = (state.get('cooldown_until') or {})
                                        cdmap[sym] = time.time() + (cdw * 60)
                                        state['cooldown_until'] = cdmap
                                    save_state(state)
                                    send_telegram(f"AUTO COOLDOWN SET | {sym} | reason=WIN | cd_min={cdw}")
                                except Exception as e:
                                    send_telegram(f"AUTO COOLDOWN SET FAILED | {sym} | reason=WIN | {e}")

                        last_sl_tp[sym] = None
                        trade_open_ms[sym] = 0  # reset open ts after close
                        cascade_step[sym] = 0   # reset cascade after position close
                        release_open_alloc(state, sym)
                        save_state(state)

                    last_pos_amt[sym] = amt

                    # if position already open, still send info (no spam) then skip
                    if amt != 0.0:
                        if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                            # per-symbol 1h bias for HOLD msg (use precomputed 1h trends)
                            try:
                                _t1h = sol_1h if sym == "SOLUSDT" else (btc_1h if sym == "BTCUSDT" else (eth_1h if sym == "ETHUSDT" else "NA"))
                                _bias = "LONG" if _t1h == "BULL" else ("SHORT" if _t1h == "BEAR" else "NO TRADE")
                            except Exception:
                                _t1h = "NA"
                                _bias = "NO TRADE"
                            send_telegram(
                                f"AUTO HOLD | {sym} | pos_open=YES | bias_1h={_bias} t1h={_t1h} | trades_today={auto_trades_today[sym]}/{AUTO_MAX_TRADES_PER_DAY} | loss_streak={loss_streak[sym]}/2"
                            )
                            auto_last_info_ts[sym] = time.time()
                        continue

                    base = sym.replace("USDT", "")
                    # market regime filter (PER-SYMBOL)
                    try:
                        if sym == "SOLUSDT":
                            _t1h = sol_1h
                        elif sym == "BTCUSDT":
                            _t1h = btc_1h
                        elif sym == "ETHUSDT":
                            _t1h = eth_1h
                        else:
                            _t1h = trend_ema(ex, f"{base}/USDT", "1h")

                        if _t1h == "BULL":
                            market_bias = "LONG"
                        elif _t1h == "BEAR":
                            market_bias = "SHORT"
                        else:
                            market_bias = "NO TRADE"
                    except Exception:
                        market_bias = "NO TRADE"
                    trend_15 = trend_ema(ex, f"{base}/USDT", "15m")

                    # ENTRY MODE: AGGRESSIVE (1H bias drives entries; 15m used only for info)
                    if market_bias == "LONG":
                        side = "buy"
                    elif market_bias == "SHORT":
                        side = "sell"
                    else:
                        side = None

                    # AUTO WAIT (market_bias info) - trimite doar cand NU avem side (NO TRADE), ca sa nu blocheze VETO-urile
                    if (not side) and (((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols)):
                        send_telegram(
                            f"AUTO WAIT | {sym} | market_bias={market_bias} | SOL1H={sol_1h} BTC1H={btc_1h} ETH1H={eth_1h} | "
                            f"{base}15={trend_15} | trades_today={auto_trades_today[sym]}/{AUTO_MAX_TRADES_PER_DAY} | loss_streak={loss_streak[sym]}/2"
                        )
                        auto_last_info_ts[sym] = time.time()

                    if not side:
                        continue

                    # ANTI-HEDGE GATE (v1): avoid mixed long/short across symbols
                    try:
                        # Determine current open direction across symbols (LIVE positions)
                        open_dir = None  # "buy" or "sell"
                        mixed = False
                        for _s in symbols:
                            _p = get_position(ex, _s)
                            _amt = float(_p.get("positionAmt", 0.0)) if _p else 0.0
                            if _amt == 0.0:
                                continue
                            _d = "buy" if _amt > 0 else "sell"
                            if open_dir is None:
                                open_dir = _d
                            elif _d != open_dir:
                                mixed = True
                                break

                        if mixed:
                            # already mixed -> do not add exposure (stabilize)
                            if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                send_telegram(f"AUTO VETO | {sym} | ANTI_HEDGE mixed_open_dirs -> block new entries")
                                auto_last_info_ts[sym] = time.time()
                            continue

                        if (open_dir is not None) and (side != open_dir):
                            if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                send_telegram(f"AUTO VETO | {sym} | ANTI_HEDGE open_dir={open_dir} wanted={side} -> block")
                                auto_last_info_ts[sym] = time.time()
                            continue
                    except Exception:
                        # fail-open: do not block trading on anti-hedge errors
                        pass
                    # ANTI-CHOP VETO (v1): block entries in range/compression/no-direction
                    try:
                        chop_on = env("ANTI_CHOP_ENABLED", "1") == "1"
                        if chop_on:
                            allow_chop, chop_reason, chop_m = anti_chop_check(ex, sym)
                            if not allow_chop:
                                # DEBUG CASCADE on CHOP_VETO (low-spam)
                                if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                    send_telegram(f"DEBUG CASCADE(VETO) | {sym} | reserved={reserved_open_usdt(state):.2f} open_idx={len(open_alloc_map(state))}")
                                    auto_last_info_ts[sym] = time.time()
                                if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                    adx = chop_m.get("adx_1h")
                                    bbw = chop_m.get("bbw_15m")
                                    slp = chop_m.get("slope_1h")
                                    send_telegram(
                                        f"AUTO VETO | {sym} | {chop_reason} | adx1h={adx} bbw15m={bbw} slope1h={slp}"
                                    )
                                    auto_last_info_ts[sym] = time.time()
                                continue
                    except Exception as _e:
                        # fail-open: do not block trading on anti-chop errors
                        pass

                    # PULLBACK VETO (swing): require pullback to EMA + 15m confirmation (avoid mid-range entries)
                    try:
                        pb_on = env("PULLBACK_ENTRY_ENABLED", "1") == "1"
                        if pb_on:
                            ok_pb, pb_reason, pb_m = pullback_entry_ok(ex, sym, side)
                            if not ok_pb:
                                if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                    send_telegram(
                                        f"AUTO WAIT | {sym} | {pb_reason} | tf={pb_m.get('tf')} ema15={pb_m.get('ema15')} "
                                        f"o={pb_m.get('o')} h={pb_m.get('h')} l={pb_m.get('l')} c={pb_m.get('c')} "
                                        f"touched={pb_m.get('touched')} confirmed={pb_m.get('confirmed')} body_ok={pb_m.get('body_ok')}"
                                    )
                                    auto_last_info_ts[sym] = time.time()
                                continue
                    except Exception:
                        # fail-open
                        pass
                    # Intel pre-trade guard (global/symbol)
                    allow, risk_mult, guard_note = pre_trade_guard(state, sym)
                    if not allow:
                            # optional: low-spam notice
                            if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                if sym in info_symbols:
                                    send_telegram(f"AUTO VETO | {sym} | {guard_note}")
                                auto_last_info_ts[sym] = time.time()
                            continue

                    asset, bal = equity_preferred(ex)
                    px = market_price(ex, sym)

                    # VOL VETO (v2): block entry if ATR% too high (extreme volatility)
                    try:
                        vol_tf = env("VOL_ATR_TF", env("ATR_TF", "15m"))
                        vol_n = int(env("VOL_ATR_PERIOD", env("ATR_PERIOD", "14")))
                        vol_veto = float(env("VOL_VETO_ATR_PCT", "0.015"))
                        atr_sym = sym.replace("USDT", "") + "/USDT" if "/" not in sym else sym
                        atr = atr_simple(ex, atr_sym, vol_tf, vol_n)
                        if atr is not None and px:
                            atr_pct = float(atr) / float(px)
                            if atr_pct >= vol_veto:
                                # low-spam notice
                                if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                                    send_telegram(f"AUTO VETO | {sym} | VOL_VETO atr_pct={atr_pct:.4f} thr={vol_veto:.4f} tf={vol_tf} n={vol_n}")
                                    auto_last_info_ts[sym] = time.time()
                                continue
                    except Exception as _e:
                        # do not break trading on volatility-check errors; just report once in logs
                        send_telegram(f"VOL CHECK FAILED | {sym} | {_e}")

                    # CASCADE allocation (global, strict): 10% of remaining free after each opened position
                    alloc_pct = float(env("CASCADE_PCT", env("CASCADING_PCT", "0.10")))
                    cap_env = float(env("LIVE_MAX_USDT", "0"))  # 0 => AUTO cap (use remaining_cycle)
                    # derive cascade inputs from reserved open allocations
                    # free = equity_preferred() free balance
                    cascade_free0_local = float(cascade_free0 or 0.0)
                    cascade_used = reserved_open_usdt(state)
                    cascade_open_idx = len(open_alloc_map(state))

                    remaining_cycle = max(0.0, float(cascade_free0_local) - float(cascade_used))
                    live_max_usdt = (cap_env if cap_env > 0 else remaining_cycle)
                    step = int(cascade_open_idx)

                    alloc_usd_raw = float(cascade_alloc_usdt(float(cascade_free0_local), step, float(alloc_pct))) * float(risk_mult)
                    alloc_usd = float(alloc_usd_raw)

                    # cap by LIVE_MAX_USDT (if >0); if <=0 => AUTO cap=remaining_cycle
                    if live_max_usdt > 0:
                        alloc_usd = min(float(live_max_usdt), float(alloc_usd))

                    # never exceed remaining_cycle
                    if alloc_usd > remaining_cycle:
                        alloc_usd = remaining_cycle
                    # DEBUG CASCADE (low-spam)
                    if ((time.time() - auto_last_info_ts[sym]) >= AUTO_INFO_EVERY_SEC) and (sym in info_symbols):
                        send_telegram(f"DEBUG CASCADE | {sym} | free={cascade_free0_local:.2f} reserved={cascade_used:.2f} open_idx={cascade_open_idx} step={step} pct={alloc_pct:.2f} raw={alloc_usd_raw:.2f} final={alloc_usd:.2f} remaining={remaining_cycle:.2f} cap={live_max_usdt:.2f}")
                        auto_last_info_ts[sym] = time.time()
                    notional = alloc_usd * lev
                    qty = (0.0 if px<=0 else (notional / px))

                    max_notional = float(env("MAX_NOTIONAL_USD", "1000"))
                    if notional > max_notional:
                        qty = max_notional / px
                        notional = max_notional

                    qty, ok_qty, qty_note = normalize_qty_for_market(ex, sym, qty, px)
                    if not ok_qty:
                        send_telegram(f"AUTO VETO | {sym} | qty_invalid={qty_note} | qty={qty} | px~{px} | alloc={alloc_usd:.2f} {asset}")
                        continue

                    send_telegram(
                        f"AUTO PLAN {side.upper()} | {sym} | guard={guard_note} | qty={qty} | alloc={alloc_usd:.2f} {asset} step={step} pct={alloc_pct*100:.1f}% mult={risk_mult:.2f} notional~{notional:.2f} | price~{px} | "
                        f"market_bias={market_bias} | {base}15={trend_15}"
                    )

                    if (env("DRY_RUN", env("AUTO_DRYRUN","0")) == "1") or (env("AUTO_DRYRUN","0") == "1"):
                        time.sleep(2.1)
                        send_telegram(f"AUTO DRYRUN | {sym} | would_place_market side={side} qty={qty}")
                        continue

                    # HARD FAILSAFE: require explicit arming for LIVE execution
                    if env("LIVE_ARM","NO") != "YES":
                        send_telegram(f"AUTO VETO | {sym} | LIVE_ARM!=YES -> blocked (set LIVE_ARM=YES to trade)")
                        continue

                    res = place_market(ex, sym, side, qty)
                    send_telegram(f"AUTO ORDER OK | {sym} id={res.get('id','?')}")
                    # reserve open allocation for cascading (10% of remaining free)
                    try:
                        reserve_open_alloc(state, sym, float(alloc_usd))
                        save_state(state)
                    except Exception:
                        pass

                    # update cascade counters only after successful order
                    try:
                        cascade_used = float(cascade_used) + float(alloc_usd)
                        cascade_open_idx = int(cascade_open_idx) + 1
                    except Exception:
                        pass

                    time.sleep(1)
                    p2 = get_position(ex, sym)
                    amt2 = float(p2.get("positionAmt", 0.0)) if p2 else 0.0
                    entry = float(p2.get("entryPrice", 0.0)) if p2 else 0.0

                    if amt2 != 0.0 and entry != 0.0:
                        sl, tp, used = place_sl_tp(ex, sym, "buy" if amt2 > 0 else "sell", abs(amt2), entry, sl_pct, tp_pct)
                        send_telegram(f"AUTO SL/TP SET | {sym} entry={entry} | SL={sl} | TP={tp} | mode={used}")
                        last_sl_tp[sym] = ("buy" if amt2 > 0 else "sell", sl, tp)
                    else:
                        send_telegram(f"AUTO SL/TP SKIP | {sym} no position detected after entry")

                    auto_last_trade_ts[sym] = time.time()
                    auto_trades_today[sym] += 1
                    cascade_step[sym] = cascade_step.get(sym,0) + 1
                    trades_today_global += 1
                    state["trades_today_global"] = trades_today_global
                    save_state(state)

            except Exception as e:
                if "__AUTO_SKIP__" in str(e):
                    pass
                else:
                    send_telegram(f"AUTO FAILED | {e}")


        # heartbeat every ~5 minutes
        if n % max(1, (300 // hb)) == 0:
            send_telegram(f"HEARTBEAT | host={server_name} | mode={mode} | LIVE_ENABLED={live_enabled}")

        updates = tg_get_updates(offset)
        for upd in updates:

            offset = upd["update_id"] + 1
            state["tg_offset"] = offset
            save_state(state)
            msg = upd.get("message") or {}
            # ACL: accept commands only from configured TELEGRAM_CHAT_ID
            allowed = str(env("TELEGRAM_CHAT_ID", "")).strip()
            from_id = str((msg.get("chat") or {}).get("id") or "").strip()
            if allowed and from_id and from_id != allowed:
                continue
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            cmd = text.split()[0].lower()

            if cmd == "/live_on":
                live_enabled = True
                state["live_enabled"] = True
                save_state(state)
                send_telegram(f"LIVE_ENABLED set to YES (persisted) | host={server_name}")
                continue

            if cmd == "/auto_on":
                auto_enabled = True
                state["auto_enabled"] = True
                save_state(state)
                send_telegram(f"AUTO_ENABLED set to YES (persisted) | host={server_name}")
                continue

            if cmd == "/live_off":
                live_enabled = False
                state["live_enabled"] = False
                save_state(state)
                send_telegram(f"LIVE_ENABLED set to NO (persisted) | host={server_name}")
                continue

            if cmd == "/auto_off":
                auto_enabled = False
                state["auto_enabled"] = False
                save_state(state)
                send_telegram(f"AUTO_ENABLED set to NO (persisted) | host={server_name}")
                continue

            if cmd == "/intel":
                intel = intel_get(state)
                g = intel.get("global", {})
                mode_g = g.get("mode", "NORMAL")
                until_g = int(g.get("until", 0) or 0)
                reason_g = g.get("reason", "")

                parts = [f"INTEL | global={mode_g} until={until_g} reason={reason_g}"]

                # Funding snapshot (only if LIVE + exchange available)
                if mode == "LIVE" and ex is not None:
                    fs = []
                    for s in symbols:
                        try:
                            ov = env(f"FUNDING_OVERRIDE_{s}", "")
                            if ov:
                                fr = float(ov)
                                fs.append(f"{s}:funding={fr:.6f}(OVR)")
                            else:
                                fr = get_funding_rate(ex, s.replace("USDT", "") + "/USDT" if "/" not in s else s)
                                if fr is None:
                                    fs.append(f"{s}:funding=NA")
                                else:
                                    fs.append(f"{s}:funding={float(fr):.6f}")
                        except Exception:
                            fs.append(f"{s}:funding=ERR")
                    parts.append("FUNDING | " + " ".join(fs))

                sy = intel.get("symbols", {}) or {}
                # show only active symbol modes
                for sym_k, v in sy.items():
                    try:
                        m = str(v.get("mode", "NORMAL"))
                        u = int(v.get("until", 0) or 0)
                        r = str(v.get("reason", "") or "")
                        if m in ("CAUTION", "HALT") and u > now_ts():
                            parts.append(f"{sym_k}={m} until={u} reason={r}")
                    except Exception:
                        pass

                send_telegram(" | ".join(parts))
                continue

            if cmd == "/resume":
                intel_set_global(state, "NORMAL", 0, "")
                send_telegram("INTEL | global NORMAL (resume)")
                continue

            if cmd == "/halt":
                parts = text.split(maxsplit=2)
                minutes = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 60
                reason = parts[2] if len(parts) >= 3 else "manual halt"
                intel_set_global(state, "HALT", minutes, reason)
                send_telegram(f"INTEL | HALT set for {minutes} min | {reason}")
                continue

            if cmd == "/caution":
                parts = text.split(maxsplit=3)
                if len(parts) < 2:
                    send_telegram("Usage: /caution SYMBOL [minutes] [reason]")
                    continue
                sym_in = parts[1].upper()
                minutes = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 45
                reason = parts[3] if len(parts) >= 4 else "manual caution"
                intel_set_symbol(state, sym_in, "CAUTION", minutes, reason)
                send_telegram(f"INTEL | CAUTION {sym_in} for {minutes} min | {reason}")
                continue


            if cmd == "/health":
                if mode != "LIVE" or ex is None:
                    send_telegram(f"HEALTH | host={server_name} | mode={mode} | LIVE_ENABLED={live_enabled} | AUTO_ENABLED={auto_enabled}")
                    continue
                try:
                    asset, bal = equity_preferred(ex)
                    sol_t = trend_ema(ex, "SOL/USDT", "1h")
                    btc_t = trend_ema(ex, "BTC/USDT", "1h")
                    eth_t = trend_ema(ex, "ETH/USDT", "1h")
                    if sol_t == "BULL" and btc_t == "BULL" and eth_t == "BULL":
                        bias = "LONG"
                    elif sol_t == "BEAR" and btc_t == "BEAR" and eth_t == "BEAR":
                        bias = "SHORT"
                    else:
                        bias = "NO TRADE"
                    intel = intel_get(state)
                    g = intel.get("global", {})
                    gmode = g.get("mode", "NORMAL")
                    guntil = int(g.get("until", 0) or 0)
                    greason = g.get("reason", "")

                    dg = state.get("daily_guard", {})
                    eq0 = float(dg.get("equity_start", 0.0) or 0.0)
                    eqmin = float(dg.get("min_equity", 0.0) or 0.0)
                    halted = bool(dg.get("halted", False))
                    dd = (0.0 if eq0 <= 0 else max(0.0, (eq0 - eqmin) / eq0))
                    maxdd = float(env("MAX_DAILY_LOSS_PCT", "0.05"))

                    sltp = state.get("sltp_ok", {})
                    sltp_sol = int(sltp.get("SOLUSDT", 0) or 0)
                    sltp_eth = int(sltp.get("ETHUSDT", 0) or 0)

                    opens = []
                    for symx in ["SOLUSDT", "ETHUSDT", "BTCUSDT"]:
                        try:
                            px = get_position(ex, symx)
                            ax = float(px.get("positionAmt", 0.0)) if px else 0.0
                            if ax != 0.0:
                                opens.append(f"{symx}={ax:.3f}")
                        except Exception:
                            pass
                    open_txt = ",".join(opens) if opens else "none"
                    # Cooldown left per symbol (state.cooldown_until)
                    try:
                        cd = ((load_state().get("cooldown_until")) or {})
                        now = time.time()
                        cd_parts = []
                        for symx in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
                            until = float(cd.get(symx, 0) or 0)
                            left = int(until - now) if until and until > now else 0
                            cd_parts.append(f"{symx}={left}s")
                        cooldown_txt = " ".join(cd_parts)
                    except Exception:
                        cooldown_txt = "ERR"

                    # Conditional (SL/TP) orders (UM Futures openAlgoOrders)
                    try:
                        c_parts = []
                        for symx in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
                            n = count_open_conditional(ex, symx)
                            c_parts.append(f"{symx}={n}" if n >= 0 else f"{symx}=ERR")
                        cond_txt = " ".join(c_parts)
                    except Exception:
                        cond_txt = "ERR"

                    # SL/TP guard based on conditional orders count (expect 2 when position open)
                    try:
                        need = []
                        for symx in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
                            pxg = get_position(ex, symx)
                            axg = float(pxg.get("positionAmt", 0.0)) if pxg else 0.0
                            ng = count_open_conditional(ex, symx)
                            if axg != 0.0 and ng != 2:
                                need.append(f"{symx}:pos={axg:.3f},cond={ng}")
                        sltp_guard_txt = ("BAD " + ";".join(need)) if need else "OK"
                    except Exception:
                        sltp_guard_txt = "ERR"


                    # Daily guard per-symbol summary (state.daily_guard_symbols)
                    try:
                        max_dd_sym = float(env("MAX_DAILY_LOSS_PCT_PER_SYMBOL", "0.08"))
                        dgs = state.get("daily_guard_symbols") or {}
                        parts = []
                        for symx in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
                            g0 = dgs.get(symx) or {}
                            eq0s = float(g0.get("equity_start", 0.0) or 0.0)
                            mins = float(g0.get("min_equity", eq0s) or eq0s)
                            halted_s = bool(g0.get("halted", False))
                            dd_s = (0.0 if eq0s <= 0 else max(0.0, (eq0s - mins) / abs(eq0s)))
                            parts.append(f"{symx}:{dd_s*100:.2f}%/{max_dd_sym*100:.2f}%{'H' if halted_s else ''}")
                        daily_sym_txt = " ".join(parts)
                    except Exception:
                        daily_sym_txt = "ERR"

                    # Telemetry: funding + volatility snapshot (read-only)
                    try:
                        pos_thr = float(env("FUNDING_EXTREME_POS", "0.01"))
                        neg_thr = float(env("FUNDING_EXTREME_NEG", "-0.01"))
                        vol_tf = env("VOL_ATR_TF", env("ATR_TF", "15m"))
                        vol_n = int(env("VOL_ATR_PERIOD", env("ATR_PERIOD", "14")))
                        vol_veto = float(env("VOL_VETO_ATR_PCT", "0.015"))

                        f_parts = []
                        v_parts = []
                        for symx in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
                            # funding
                            try:
                                fr = get_funding_rate(ex, symx.replace("USDT","") + "/USDT")
                                if fr is None:
                                    f_parts.append(f"{symx}=NA")
                                else:
                                    f_parts.append(f"{symx}={float(fr):.6f}")
                            except Exception:
                                f_parts.append(f"{symx}=ERR")

                            # vol (ATR%)
                            try:
                                px0 = market_price(ex, symx)
                                atr0 = atr_simple(ex, symx.replace("USDT","") + "/USDT", vol_tf, vol_n)
                                if atr0 is None or not px0:
                                    v_parts.append(f"{symx}=NA")
                                else:
                                    atr_pct = float(atr0)/float(px0)
                                    v_parts.append(f"{symx}={atr_pct:.4f}")
                            except Exception:
                                v_parts.append(f"{symx}=ERR")

                        funding_txt = " ".join(f_parts)
                        vol_txt = " ".join(v_parts)
                    except Exception:
                        funding_txt = "ERR"
                        vol_txt = "ERR"

                    msg = (
                        f"HEALTH | host={server_name} | LIVE={live_enabled} AUTO={auto_enabled} | bal={bal:.2f} {asset} | "
                        f"BIAS={bias} | INTEL global={gmode} until={guntil} reason={greason} | "
                                                f"SLTP_OK SOL={sltp_sol} ETH={sltp_eth} | SLTP_GUARD={sltp_guard_txt} | COND_OPEN={cond_txt} | COOLDOWN_LEFT={cooldown_txt} | OPEN_POS {open_txt} | DAILY_SYM {daily_sym_txt} | FUNDING_NOW {funding_txt} (thr {neg_thr:.4f}..{pos_thr:.4f}) | VOL_ATR_PCT {vol_txt} (veto {vol_veto:.4f} tf={vol_tf} n={vol_n})"
                    )
                    send_telegram(msg)
                except Exception as e:
                    send_telegram(f"HEALTH FAILED | {e}")
                continue

            if cmd == "/status":
                if mode != "LIVE" or ex is None:
                    send_telegram(f"STATUS | host={server_name} | mode={mode}")
                    continue
                asset, bal = equity_preferred(ex)
                px = market_price(ex, symbol)
                send_telegram(f"STATUS | host={server_name} | mode={mode} | LIVE_ENABLED={live_enabled} | AUTO_ENABLED={auto_enabled} | bal={bal:.2f} {asset} | price={px} | SL={sl_pct:.2f}% TP={tp_pct:.2f}%")
                continue

            if cmd == "/analyze":
                if mode != "LIVE" or ex is None:
                    send_telegram("ANALYZE only in LIVE mode.")
                    continue
                sol_t = trend_ema(ex, "SOL/USDT", "1h")
                btc_t = trend_ema(ex, "BTC/USDT", "1h")
                eth_t = trend_ema(ex, "ETH/USDT", "1h")

                if sol_t == "BULL" and btc_t == "BULL" and eth_t == "BULL":
                    bias = "LONG"
                elif sol_t == "BEAR" and btc_t == "BEAR" and eth_t == "BEAR":
                    bias = "SHORT"
                else:
                    bias = "NO TRADE"

                send_telegram(f"ANALYZE 1H EMA20/50 | SOL={sol_t} BTC={btc_t} ETH={eth_t} => BIAS={bias}")
                continue

            if cmd in ("/buy", "/sell"):
                if mode != "LIVE" or ex is None:
                    send_telegram("Not in LIVE mode.")
                    continue
                if not live_enabled:
                    send_telegram("LIVE disabled. Use /live_on first.")
                    continue

                side = "buy" if cmd == "/buy" else "sell"
                px = market_price(ex, symbol)
                qty = 10.0

                send_telegram(f"EXEC {side.upper()} | {symbol} qty={qty} SOL price~{px}")
                try:
                    res = place_market(ex, symbol, side, qty)
                    send_telegram(f"ORDER OK | id={res.get('id','?')}")

                    # get entry + real qty from position, then place SL/TP
                    time.sleep(1)
                    p = get_position(ex, symbol)
                    if p:
                        amt = float(p.get("positionAmt", 0.0))
                        entry = float(p.get("entryPrice", 0.0))
                        if amt != 0.0 and entry != 0.0:
                            sl, tp, used = place_sl_tp(ex, symbol, "buy" if amt > 0 else "sell", abs(amt), entry, sl_pct, tp_pct)
                            send_telegram(f"SL/TP SET | entry={entry} | SL={sl} | TP={tp} | mode={used}")
                        else:
                            send_telegram("SL/TP SKIP | no position detected after entry")
                    else:
                        send_telegram("SL/TP SKIP | positionRisk unavailable")
                except Exception as e:
                    send_telegram(f"ORDER FAILED | {e}")
                continue

            if cmd == "/close":
                if mode != "LIVE" or ex is None:
                    send_telegram("Not in LIVE mode.")
                    continue
                if not live_enabled:
                    send_telegram("LIVE disabled. Use /live_on first.")
                    continue
                try:
                    p = get_position(ex, symbol)
                    amt = float(p.get("positionAmt", 0.0)) if p else 0.0
                    if amt == 0.0:
                        send_telegram(f"CLOSE | no open position for {symbol}")
                        continue
                    close_side = "sell" if amt > 0 else "buy"
                    qty = float(f"{abs(amt):.3f}")
                    r = place_market(ex, symbol, close_side, qty)
                    send_telegram(f"CLOSE OK | qty={qty} side={close_side.upper()} id={r.get('id','?')}")
                except Exception as e:
                    send_telegram(f"CLOSE FAILED | {e}")
                continue

        time.sleep(hb)


if __name__ == "__main__":
    main()
