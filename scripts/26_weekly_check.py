"""주봉 전략(워뇨띠가 파는 모양의 주봉 뒤에 1주일 롱 — 워뇨띠와 반대 방향)을 BTC·ETH 밖의 코인으로 검증한다.

전략과 값은 scripts/25_tf_strategies.py 가 저장한 models/aoa_tf_1w.json 그대로다. 코인은
  ① 워뇨띠가 거래한 나머지 코인 9개
  ② 바이낸스 USDT 선물 알트 중 2021-06 이전에 상장된 것에서 난수(seed)로 뽑은 20개
구간은 2022-01 ~ 현재(주봉 약 250개).
사용: python scripts/26_weekly_check.py [seed=31]
"""
import importlib.util
import sys

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.features_k import to_higher
from aoabot.policy import BUY, ROOT, SELL, load_bundle

spec = importlib.util.spec_from_file_location("tfs", "scripts/25_tf_strategies.py")
tfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfs)

pd.set_option("display.width", 250)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 31
HIS = ["XRP", "LTC", "BCH", "DOT", "DOGE", "LINK", "BNB", "ADA", "TRX"]
b = load_bundle(ROOT / "models" / "aoa_tf_1w.json")
plan = {int(k): tuple(v) for k, v in b["meta"]["plan"].items()}
thr = {BUY: b["thr_buy"], SELL: b["thr_sell"]}
size = {k[0]: v for k, v in b["size"].items()}
print("전략:", {("매수 신호" if k == BUY else "매도 신호"): f"{'롱' if d > 0 else '숏'} {h}주 보유, 크기 {size[k]:.2f}배" for k, (d, h) in plan.items()})

ex = ccxt.binanceusdm({"enableRateLimit": True})
markets = ex.load_markets()


def daily(coin):
    rows, since, now = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000), int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    while since < now:
        x = ex.fetch_ohlcv(f"{coin}/USDT:USDT", "1d", since=since, limit=1000)
        if not x:
            break
        rows += x
        since = x[-1][0] + 86_400_000
    c = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    c.index = pd.to_datetime(c.t, unit="ms") + pd.Timedelta(days=1)
    return c[c.index <= pd.Timestamp.now()].drop(columns="t")


alts = sorted(m["base"] for s, m in markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")
              and m["base"] not in ["BTC", "ETH"] + HIS)
rng = np.random.default_rng(SEED)
rng.shuffle(alts)
groups, rnd = {"워뇨띠가 거래한 코인": HIS, "무작위 알트": []}, []
for a in alts:
    if len(rnd) == 20:
        break
    first = ex.fetch_ohlcv(f"{a}/USDT:USDT", "1d", since=0, limit=1)
    if first and pd.Timestamp(first[0][0], unit="ms") < pd.Timestamp("2021-06-01"):
        rnd.append(a)
groups["무작위 알트"] = rnd

rows = []
for gname, coins in groups.items():
    for coin in coins:
        k = to_higher(daily(coin), "W")
        for pname, (s, e) in {"2022~2023": ("2022-01-01", "2023-12-31"), "2024~": ("2024-01-01", None), "전체 2022~": ("2022-01-01", None)}.items():
            r = tfs.backtest(k, b["model"], thr, plan, size, s, e, 52)
            rows.append(dict(묶음=gname, 코인=coin, 구간=pname, **r))
d = pd.DataFrame(rows)
d.to_csv(f"data/weekly_check_seed{SEED}.csv", index=False, encoding="utf-8-sig")
full = d[d.구간 == "전체 2022~"]
show = full[["묶음", "코인", "수익", "단순보유", "샤프", "최대낙폭", "거래수", "승률"]].copy()
for c_ in ("수익", "단순보유"):
    show[c_] = show[c_].map("{:+.1%}".format)
show["최대낙폭"] = show.최대낙폭.map("{:.1%}".format)
print(show.round(2).to_string(index=False))
print("\n== 요약 ==")
for (g, pn), x in d.groupby(["묶음", "구간"], sort=False):
    print(f"[{g} · {pn}] 평균 {x.수익.mean():+.1%}, 중앙값 {x.수익.median():+.1%}, 수익 난 코인 {(x.수익 > 0).sum()}/{len(x)}, 최악 {x.수익.min():+.1%}, 최고 {x.수익.max():+.1%}, "
          f"평균 거래 {x.거래수.mean():.0f}회, 단순 보유 평균 {x.단순보유.mean():+.1%}")
