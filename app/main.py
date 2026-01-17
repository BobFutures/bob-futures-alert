import os
import time
import yaml
import requests
from dotenv import load_dotenv

load_dotenv("/app/.env")

def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v

def send_telegram(text: str) -> None:
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        print(f"[TELEGRAM] status={r.status_code}")
    except Exception as e:
        print(f"[TELEGRAM] error={e}")

def load_config():
    with open("/app/config/config.yaml", "r") as f:
        return yaml.safe_load(f) or {}

def main():
    cfg = load_config()
    mode = (cfg.get("mode") or "SAFE").upper()
    timeframe = cfg.get("timeframe", "1m")
    hb = int(cfg.get("logic", {}).get("heartbeat_sec", 60))

    symbol = env("SYMBOL", "SOLUSDT")
    leverage = env("LEVERAGE", "3")

    host = env("SERVER_NAME", env("HOSTNAME", os.uname().nodename))

    print(f"[START] host={host} mode={mode} symbol={symbol} timeframe={timeframe} lev={leverage}")
    send_telegram(f"BOT STARTED | host={host} | mode={mode} | {symbol} tf={timeframe} lev={leverage}")

    n = 0
    while True:
        n += 1
        print(f"[HEARTBEAT] n={n} host={host} mode={mode}")
        if n % max(1, (300 // hb)) == 0:  # aproximativ la 5 minute
            send_telegram(f"BOT HEARTBEAT | host={host} | mode={mode} | n={n}")
        time.sleep(hb)

if __name__ == "__main__":
    main()
