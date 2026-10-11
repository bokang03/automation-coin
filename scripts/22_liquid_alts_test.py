"""거래대금이 큰 알트에서만 타이밍 전략이 통하는지, 지금까지 본 적 없는 차트로 확인한다.

앞선 알트 49개에서 '거래대금이 큰 코인에서만 수익'이라는 경향이 보였다(사후 분석). 그것을 미리 정한 규칙으로 바꿔 새 차트에 시험한다:
  규칙: 평가 시작 직전 30일의 일평균 거래대금이 1억 달러 이상인 알트만 매매한다 (그 시점에 알 수 있는 값만 쓴다).
바이낸스 USDT 무기한 선물 알트를 난수(seed)로 섞고, 2024-01-01 이후에서 시작일을 균등 난수로 뽑아 규칙을 통과한 것만 1년 구간으로 쓴다.
설정은 scripts/18_final_test.py 의 최종 설정 그대로다.
사용: python scripts/22_liquid_alts_test.py [차트 수=20] [seed=23]
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

args = sys.argv[1:]
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("final", "scripts/18_final_test.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

pd.set_option("display.width", 250)
N = int(args[0]) if args else 20
SEED = int(args[1]) if len(args) > 1 else 23
MIN_USD = 100e6
SEEN = set(pd.read_csv("data/alt_validation_all.csv").차트.str.split(" ").str[0])       # 이미 본 종목은 뺀다
ex = ft.ex
markets = ex.load_markets()
symbols = sorted(s for s, m in markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")
                 and m.get("base") not in ("BTC", "ETH") and s.split("/")[0] not in SEEN)
rng = np.random.default_rng(SEED)
rng.shuffle(symbols)
last_start = pd.Timestamp.now(tz="UTC").floor("D") - ft.LENGTH
picks, scanned = [], 0
for s in symbols:
    if len(picks) == N:
        break
    first = ex.fetch_ohlcv(s, "1d", since=0, limit=1)
    if not first:
        continue
    lo = max(ft.FIRST, (pd.Timestamp(first[0][0], unit="ms", tz="UTC") + ft.WARMUP).ceil("D"))
    if lo >= last_start:
        continue
    start = lo + pd.Timedelta(days=int(rng.integers(0, (last_start - lo).days + 1)))
    scanned += 1
    d = ex.fetch_ohlcv(s, "1d", since=int((start - pd.Timedelta(days=30)).timestamp() * 1000), limit=30)
    usd = float(np.mean([r[4] * r[5] for r in d])) if d else 0.0
    if usd >= MIN_USD:
        picks.append((s, start, usd))
        print(f"  통과: {s.split('/')[0]} {start:%Y-%m-%d} (직전 30일 일평균 {usd / 1e6:,.0f}백만 달러)", flush=True)
print(f"살펴본 알트 {scanned}개 중 거래대금 기준 통과 {len(picks)}개 (seed {SEED})", flush=True)

rows = []
for s, start, usd in picks:
    c = ft.fetch(s, start - ft.WARMUP, start + ft.LENGTH)
    for r in ft.run_all(f"{s.split('/')[0]} {start:%Y-%m-%d}~", c, start.tz_localize(None), (start + ft.LENGTH).tz_localize(None)):
        rows.append({**r, "직전30일_거래대금_백만달러": round(usd / 1e6)})
    print("  완료:", s, flush=True)
d = pd.DataFrame(rows)
d.to_csv(f"data/liquid_alts_seed{SEED}.csv", index=False, encoding="utf-8-sig")
m = d[d.설정 == "최종 설정 (시장가, 안전장치 있음)"]
show = m[["차트", "직전30일_거래대금_백만달러", "수익", "단순보유", "최대낙폭", "거래수", "승률", "거래당_수익bp"]].copy()
for k in ("수익", "단순보유"):
    show[k] = show[k].map("{:+.1%}".format)
for k in ("최대낙폭", "승률"):
    show[k] = show[k].map("{:.1%}".format)
print("\n== 거래대금 큰 알트: 최종 설정, 초기 10,000 USDT, 시장가 수수료, 계좌 안전장치 포함 ==")
print(show.round(1).to_string(index=False))
print("\n== 요약 ==")
for v, g in d.groupby("설정", sort=False):
    print(f"[{v}] 평균 {g.수익.mean():+.1%}, 중앙값 {g.수익.median():+.1%}, 수익 난 차트 {(g.수익 > 0).sum()}/{len(g)}, 최악 {g.수익.min():+.1%}, "
          f"최고 {g.수익.max():+.1%}, 평균 최대낙폭 {g.최대낙폭.mean():.1%}, 거래당 {g.거래당_수익bp.mean():+.1f}bp, 단순 보유 평균 {g.단순보유.mean():+.1%}")
