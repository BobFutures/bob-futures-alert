import os
import time
import yaml
import requests
from dotenv import load_dotenv
import ccxt

load_dotenv("/app/.env")


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


def send_telegram(text: str) -> None:
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        print(f"[TELEGRAM] status={r.status_code}")
    except Exception as e:
        print(f"[TELEGRAM] error={e}")


def tg_get_updates(offset: int | None):
    token = env("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"timeout": 20}
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=30)
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
    return ex


def set_leverage(ex, symbol, lev: int):
    try:
        ex.set_leverage(lev, symbol)
    except Exception as e:
        print(f"[WARN] set_leverage failed: {e}")


def equity_preferred(ex) -> tuple[str, float]:
    """
    Pentru Multi-Assets, 'total' poate fi in USDC, iar BNFCR poate fi negativ.
    Prioritate: USDC -> USDT -> BNFCR -> BTC
    """
    bal = ex.fetch_balance()
    total = bal.get("total", {}) or {}
    for a in ("USDC", "USDT", "BNFCR", "BTC"):
        v = float(total.get(a, 0.0) or 0.0)
        if v != 0.0:
            return a, v
    return "USDT", float(total.get("USDT", 0.0) or 0.0)


def market_price(ex, symbol) -> float:
    t = ex.fetch_ticker(symbol)
    return float(t["last"])


def place_market(ex, symbol, side: str, amount: float):
    return ex.create_order(symbol, "market", side, amount)


def get_position(ex, symbol: str):
    sym = symbol.replace("/", "")
    pr = ex.fapiPrivateV2GetPositionRisk({"symbol": sym})
    if not pr:
        return None
    return pr[0]



def place_sl_tp(ex, symbol: str, side_open: str, qty: float, entry: float, sl_pct: float, tp_pct: float):
    """
    Plaseaza SL/TP ca *Algo conditional orders* (Binance USDS-M Futures: POST /fapi/v1/algoOrder).
    side_open: 'buy' (LONG) or 'sell' (SHORT)
    sl_pct / tp_pct in PERCENT (ex: 0.30 = 0.30%)
    """
    slp = sl_pct / 100.0
    tpp = tp_pct / 100.0

    if side_open == "buy":
        sl_trig = entry * (1.0 - slp)
        tp_trig = entry * (1.0 + tpp)
        close_side = "SELL"
    else:
        sl_trig = entry * (1.0 + slp)
        tp_trig = entry * (1.0 - tpp)
        close_side = "BUY"

    sl_trig = float(f"{sl_trig:.3f}")
    tp_trig = float(f"{tp_trig:.3f}")
    qty = float(f"{qty:.3f}")

    sym = symbol.replace("/", "")

    def algo_cond(order_type: str, trigger: float):
        params = {
            "algoType": "CONDITIONAL",
            "symbol": sym,
            "side": close_side,
            "type": order_type,               # STOP_MARKET / TAKE_PROFIT_MARKET
            "triggerPrice": float(f"{trigger:.3f}"),
            "workingType": "MARK_PRICE",
            "closePosition": "true",
            "priceProtect": "false",
        }
        # ccxt generic request to fapiPrivate
        return ex.request("algoOrder", "fapiPrivate", "POST", params)

    # SL + TP
    algo_cond("STOP_MARKET", sl_trig)
    algo_cond("TAKE_PROFIT_MARKET", tp_trig)

    return sl_trig, tp_trig
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


def main():
    cfg = load_config()
    mode = (cfg.get("mode") or "SAFE").upper()
    hb = int(cfg.get("logic", {}).get("heartbeat_sec", 60))

    # risk config (percent)
    risk = cfg.get("risk", {}) or {}
    sl_pct = float(risk.get("sl_pct", 0.30))
    tp_pct = float(risk.get("tp_pct", 0.10))

    symbol = env("SYMBOL", "SOLUSDT")
    lev = int(env("LEVERAGE", "3"))

    server_name = env("SERVER_NAME", env("HOSTNAME", os.uname().nodename))
    live_enabled = env("LIVE_ENABLED", "NO").upper() == "YES"
    live_max_usdt = float(env("LIVE_MAX_USDT", "20"))

    print(f"[START] host={server_name} mode={mode} symbol={symbol} lev={lev} live_enabled={live_enabled} max_usdt={live_max_usdt}")
    send_telegram(f"BOT STARTED | host={server_name} | mode={mode} | {symbol} lev={lev} | LIVE_ENABLED={live_enabled}")
    send_telegram("Comenzi: /status | /analyze | /buy | /sell | /close | /live_on | /live_off")

    ex = None
    if mode == "LIVE":
        ex = make_exchange()
        set_leverage(ex, symbol, lev)

    offset = None
    n = 0

    while True:
        n += 1

        # heartbeat every ~5 minutes
        if n % max(1, (300 // hb)) == 0:
            send_telegram(f"HEARTBEAT | host={server_name} | mode={mode} | LIVE_ENABLED={live_enabled}")

        updates = tg_get_updates(offset)
        for upd in updates:
            offset = upd["update_id"] + 1
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            cmd = text.split()[0].lower()

            if cmd == "/live_on":
                live_enabled = True
                send_telegram(f"LIVE_ENABLED set to YES (in-memory) | host={server_name}")
                continue

            if cmd == "/live_off":
                live_enabled = False
                send_telegram(f"LIVE_ENABLED set to NO (in-memory) | host={server_name}")
                continue

            if cmd == "/status":
                if mode != "LIVE" or ex is None:
                    send_telegram(f"STATUS | host={server_name} | mode={mode}")
                    continue
                asset, bal = equity_preferred(ex)
                px = market_price(ex, symbol)
                send_telegram(f"STATUS | host={server_name} | mode={mode} | bal={bal:.2f} {asset} | price={px} | SL={sl_pct:.2f}% TP={tp_pct:.2f}%")
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
                qty = live_max_usdt / px
                qty = float(f"{qty:.3f}")

                send_telegram(f"EXEC {side.upper()} | {symbol} qty={qty} (max_usdt={live_max_usdt}) price~{px}")
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
                            sl, tp = place_sl_tp(ex, symbol, "buy" if amt > 0 else "sell", abs(amt), entry, sl_pct, tp_pct)
                            send_telegram(f"SL/TP SET | entry={entry} | SL={sl} | TP={tp}")
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
