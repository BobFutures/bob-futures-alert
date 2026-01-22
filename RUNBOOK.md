# BOB-FUTURES-BOT-ULTIMATE — RUNBOOK (Single Source of Truth)

## Scope (obiectiv final)
Bot complet, stabil, observabil, cu risc controlat și edge confirmat (forward test).
- Multi-symbol: BTCUSDT + ETHUSDT + SOLUSDT
- Money mgmt: cascading allocation 10% din free balance rămas (per new position)
- Guards: SLTP guard, daily guard, funding extreme guard, cooldown WIN/LOSS, Intel NORMAL/CAUTION/HALT
- Ops: restart policy, log rotation, dedupe Telegram, daily summary
- Validare: KPI + forward test 14 zile

---

# ✅ DONE (gata + confirmat)

## 1) Core runtime / LIVE
- Multi-symbol LIVE: BTCUSDT + ETHUSDT + SOLUSDT
- state.json: live_enabled=True, auto_enabled=True
- HARD FAILSAFE: LIVE_ARM=YES necesar pentru ordine reale

## 2) Money management
- Cascading allocation reală (din balanța Futures / availableBalance):
  - .env: CASCADING_PCT=0.10 (alias CASCADE_PCT=0.10)
  - remaining_usdt se calculează din equity_preferred(ex) și scade după fiecare entry (AUTO ORDER OK)
- Bug fix: unbound remaining_usdt (inițializat înainte de calc alloc)

## 3) Guards / protecții
- Cooldown după WIN: COOLDOWN_AFTER_WIN_MIN=3
- Cooldown după LOSS (escaladare): 30/60/120 min (COOLDOWN_AFTER_LOSS_MIN / _2 / _3)
- Funding extreme guard:
  - FUNDING_EXTREME_POS=0.003
  - FUNDING_EXTREME_NEG=-0.003
  - FUNDING_CHECK_EVERY_SEC=60
  - FUNDING_CAUTION_MINUTES=45
- SL/TP: se setează după entry + SLTP_GUARD loop
- Conditional orders check: COND_OPEN trebuie să fie 2 per symbol
- Daily guard: bug bal/remaining_usdt fixat (nu mai dă erori)

## 4) Intel layer (v1)
- Intel global + per symbol în state.json
- Moduri: NORMAL / CAUTION / HALT
- Notificări Telegram la transition

## 5) Telegram observability
- Comenzi: /status /analyze /buy /sell /close /live_on /live_off /auto_on /auto_off /health
- HEALTH include: INTEL, SLTP_OK, SLTP_GUARD, COND_OPEN, COOLDOWN_LEFT, OPEN_POS, DAILY_SYM, FUNDING_NOW, VOL_ATR_PCT

## 6) Backup
- Backup folder creat (exemplu): /opt/futures-bot/BACKUP_YYYYMMDD_HHMMSS (app/, .env, docker-compose.yml, RUNBOOK.md, state.json)

---

# 🟡 TODO (ca să devină bot care FACE BANI)

## A) Profit Engine (obligatoriu)
1) ANTI-CHOP module (range filter) + VETO în AUTO engine
   - ADX(1H) gate (ex: ADX<18 => NO_TRADE)
   - BBWidth(15m) gate (compresie range)
   - EMA slope(1H) gate (direcție clară)
   ✅ DONE (ANTI-CHOP v1 implementat + legat în AUTO engine)
   - Indicators:
     - ADX(1H): adx_simple()
     - BBWidth(15m): bb_width() (normalized width)
     - EMA slope(1H): ema_slope() (pct over lookback)
   - AUTO VETO:
     - injectat după `if not side: continue` și înainte de `pre_trade_guard()`
     - mesaj TG: `AUTO VETO | {sym} | CHOP_VETO... | adx1h=... bbw15m=... slope1h=...`
   - ENV knobs (.env):
     - ANTI_CHOP_ENABLED=1
     - CHOP_ADX_MIN=18
     - CHOP_BBWIDTH_MIN=0.012
     - CHOP_EMA_SLOPE_MIN=0.0005

2) Entry upgrade (swing, nu scalp)
   - Pullback entry + 15m confirmation (evită intrări în mijloc de range)
3) Exit upgrade (calitate profit)
   - TP în 2-3 trepte + BE/trailing după TP1

## B) Execution quality
4) Spread/Slippage guard (VETO dacă spread peste prag)
5) Session filter (trading doar în ore cu volum)

## C) Risk / exposure
6) Cap global exposure (max poziții simultan / max notional total)
7) Dynamic leverage (reduce lev când VOL_ATR_PCT e mare)

## D) Validare (fără asta e „noroc”)
8) Daily Summary automat (KPI)
   - trades, winrate, avg RR, expectancy, fees, net pnl, maxDD
9) Forward test 14 zile + praguri GO/NO-GO (EV/trade>0, DD sub limită)

## E) Ops hardening
10) Anti-spam HOLD/WAIT (AUTO_INFO_EVERY_SEC + AUTO_INFO_SYMBOLS)
11) log rotation
12) restart policy + watchdog
13) dedupe Telegram (anti duplicate)

---

# 🎯 NEXT (mâine)
Implementăm: **Entry upgrade (swing, nu scalp)** — Pullback entry + 15m confirmation (evită intrări în mijloc de range).

---

## ✅ Status curent (2026-01-22) — checkpoint

### LIVE / Runtime
- Bot LIVE multisymbol: BTCUSDT + ETHUSDT + SOLUSDT
- Telegram OK: BOT STARTED + Comenzi + AUTO HEALTH + AUTO VETO/HOLD

### Cascading allocation 10% (CONFIRMAT)
- Cascading allocation global: 10% din free rămas per poziție nouă
- Reserved open allocation se ține în `state.json`:
  - `open_alloc_usdt` map per symbol
  - `reserved_open_usdt(state)` calculează suma
  - `open_alloc_map(state)` returnează map
- DEBUG CASCADE(VETO) confirmă corect reserved/open_idx în timp real
- În acest moment:
  - open_alloc_map = {'ETHUSDT': 125.12, 'SOLUSDT': 103.48}
  - reserved_open_usdt ≈ 228.60
  - open_idx = 2/3 (lipsește BTCUSDT)

### Fix-uri critice
- Eliminat error: `AUTO FAILED | name 'free' is not defined`
- `cascade_used` derivat din state: `reserved_open_usdt(state)`
- Anti-chop / pullback / vol veto funcționează

### Poziții deschise (confirmate din Binance positionRisk v2)
- SOLUSDT SHORT amt=-2.34 entry=127.87
- ETHUSDT SHORT amt=-0.102 entry=2932.26

### SL/TP Conditional Orders (FIXED)
- Binance UI "Conditional Orders" = **FAPI openAlgoOrders** (nu openOrders)
- Verificare corectă via ccxt: `ex.request("openAlgoOrders","fapiPrivate","GET", {"symbol":"SOLUSDT"})`
- `openOrders` poate fi 0 chiar dacă există SL/TP în UI
- Chei răspuns corecte: `orderType` + `algoType` + `algoStatus` (nu `type`/`status`)
- `count_open_conditional()` numără strict doar CONDITIONAL + {STOP_MARKET, TAKE_PROFIT_MARKET}
- `algo_has_sltp()` fixat pe `orderType` (fallback `type`)
- SLTP guard: dacă detectează deja cele 2 ordine, doar refresh TTL și NU replasează (nu mai mută triggerele ATR)
- Confirmare LIVE: SOLUSDT=2, ETHUSDT=2, BTCUSDT=0 în openAlgoOrders; fără spam "SAFETY | SLTP GUARD VERIFIED" după 70s
---

## ✅ Target 15–25%/lună — PARAMS ACTIVE (2026-01-22)

### Risk / sizing
- RISK_PCT=0.008 (0.8% per trade) + SYNC_RISK_PCT_FROM_ENV on start (state.json forced)

### Guards
- DAILY_MAX_LOSS_PCT=2.0
- MAX_CONSEC_LOSSES=2
- MAX_TRADES_PER_DAY=4

### Cooldown
- COOLDOWN_LOSS_MIN=90
- COOLDOWN_WIN_MIN=45

### Anti-chop / Vol-veto
- EMA_SPREAD_MIN_PCT=0.15
- ADX_MIN=18
- VOL_VETO_ATR_PCT_MAX=1.2

---

## ✅ FINAL CHECKPOINT (2026-01-23) — Stability + 15–25%/lună params

### LIVE / Runtime
- Multisymbol LIVE: BTCUSDT + ETHUSDT + SOLUSDT
- Leverage: 3x
- AUTO_ENABLED=1, LIVE_ENABLED=YES
- Cascading allocation: 10% din free rămas per poziție nouă (reserved map în state.json)

### Risk / Target 15–25%/lună (active)
- RISK_PCT=0.008 (0.8% / trade) + SYNC_RISK_PCT_FROM_ENV (forțează state.json la start)
- DAILY_MAX_LOSS_PCT=2.0
- MAX_CONSEC_LOSSES=2
- MAX_TRADES_PER_DAY=4
- COOLDOWN_LOSS_MIN=90
- COOLDOWN_WIN_MIN=45
- EMA_SPREAD_MIN_PCT=0.15
- ADX_MIN=18
- VOL_VETO_ATR_PCT_MAX=1.2

### Fixes / Notes
- CCXT positionRisk: folosește v2 (v1 dă 404).
- trades_today afișează corect /4 (AUTO_MAX_TRADES_PER_DAY citește MAX_TRADES_PER_DAY).

### Known issue (de rezolvat ulterior)
- Vizualizare SL/TP via API: Binance UI folosește Conditional Orders; CCXT endpoints pentru acestea cer permisiuni PAPI (posibil neactivate/whitelist IP).

### ✅ AUTO CAP (balanța disponibilă)
- `LIVE_MAX_USDT=0` => cap automat = `remaining_cycle = availableBalance - reserved_open_usdt(state)`
- Practic: MAX USDT = balanța disponibilă; se aplică doar cascading 10%.

---

## ✅ CHECKPOINT FINAL — Balance + Cascade (Single Source of Truth)

### Futures Balance (Multi-Assets Margin = True)
**Problemă:** `ccxt.fetch_balance({'type':'future'})` poate raporta `USDT=0` în Multi-Assets Margin și NU e sursa adevărului.

**Sursa adevăr (Binance Futures / FAPI v2):**
- `acc = ex.fapiPrivateV2GetAccount({})`
- `totalWalletBalance` = **balanța totală** (crește automat când adaugi bani)
- `availableBalance` = **balanța disponibilă** (disponibil de folosit acum)

### Cascading allocation 10% (global, strict)
- `CASCADING_PCT=0.10`
- `reserved_open_usdt(state)` = sum(open_alloc_usdt map)
- `remaining_cycle = totalWalletBalance - reserved_open_usdt(state)`
- `next_alloc = CASCADING_PCT * remaining_cycle`

### Cap (MAX_USDT / LIVE_MAX_USDT) — AUTO
- Setare: `LIVE_MAX_USDT=0` (AUTO cap)
- În AUTO: `cap = remaining_cycle` (adică **balanța totală rămasă** după reserved)
- Dacă `LIVE_MAX_USDT > 0`: cap fix pe per-trade allocation.

### Verificare rapidă
```bash
docker exec -it futures-bot sh -lc "python app/debug_balance.py"

