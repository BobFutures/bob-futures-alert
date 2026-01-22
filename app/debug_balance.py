import os, json
import ccxt

STATE_FILE = "/app/state.json"

def env_first(*names, default=None):
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def reserved_open_usdt(state: dict) -> float:
    m = state.get("open_alloc_usdt") or {}
    try:
        return float(sum(float(v or 0.0) for v in m.values()))
    except Exception:
        return 0.0

def make_exchange():
    api_key = env_first("BINANCE_API_KEY","BINANCE_KEY","API_KEY","KEY")
    api_sec = env_first("BINANCE_API_SECRET","BINANCE_SECRET","API_SECRET","SECRET")
    if not api_key or not api_sec:
        raise SystemExit("Missing Binance API keys in env (BINANCE_API_KEY / BINANCE_API_SECRET).")
    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_sec,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        },
    })
    return ex

def main():
    st = load_state()
    ex = make_exchange()

    # TRUTH SOURCE: FAPI v2 account (Multi-Assets Margin safe)
    acc = ex.fapiPrivateV2GetAccount({})
    free = float(acc.get('totalWalletBalance') or 0.0)
    total = float(acc.get('totalWalletBalance') or 0.0)
    mam = acc.get('multiAssetsMargin')
    print('multiAssetsMargin=', mam)
    casc = float(env_first("CASCADING_PCT", default="0.10"))
    max_usdt = env_first("MAX_USDT", default="")
    symbols = env_first("SYMBOLS", default="")

    res = reserved_open_usdt(st)
    free_for_new = free - res
    next_alloc = max(0.0, free_for_new) * casc

    print("=== DEBUG BALANCE / CASCADE TRUTH ===")
    print("SYMBOLS=", symbols)
    print("MAX_USDT=", max_usdt)
    print("CASCADING_PCT=", casc)
    print("USDT.free =", free)
    print("USDT.total=", total)
    print("reserved_open_usdt(state) =", res)
    print("free_for_new = free - reserved =", free_for_new)
    print("next_alloc = CASCADING_PCT * free_for_new =", next_alloc)
    print("open_alloc_usdt map =", st.get("open_alloc_usdt"))

if __name__ == "__main__":
    main()
