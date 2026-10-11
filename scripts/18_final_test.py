"""타이밍 전략의 최종 시험. 설정은 보완용 구간(2022~2023)에서 정한 뒤 고정했고, 여기 쓰는 차트는 그 전에 본 적이 없다.

  ① BTC·ETH 2024-01 이후 (바이낸스 5분봉)
  ② 무작위 차트: 바이낸스 USDT 무기한 선물 전 종목을 난수(seed)로 섞고, 2024-01-01 이후에서 시작일을 균등 난수로 뽑은 1년 구간
사용: python scripts/18_final_test.py [차트 수=20] [seed=7]
"""
import sys
from dataclasses import replace
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.features5 import WARMUP_BARS
from aoabot.policy import ROOT, load_bundle
from aoabot.timing import TimingParams, simulate, stats

pd.set_option("display.width", 250)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
FIRST, LENGTH = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timedelta(days=365)
WARMUP = pd.Timedelta(days=70)      # 지표 준비 구간. 전월 봉까지 계산할 수 있게 넉넉히 받는다 (5분봉 지표만이면 9일이면 된다)
CACHE = Path("data/random5m")
CACHE.mkdir(parents=True, exist_ok=True)
LIM, NOL = risk.Limits(daily_loss=0.03, max_drawdown=0.25), risk.Limits(daily_loss=9.9, max_drawdown=9.9)

b = load_bundle(ROOT / "models" / "aoa_timing.json")
FINAL = TimingParams(size=round(b["meta"]["big_buy_size"], 2), max_lev=round(b["meta"]["big_buy_size"], 2), min_prior=1)
VARIANTS = {
    "최종 설정 (시장가, 안전장치 있음)": (b, FINAL, LIM),
    "최종 설정, 안전장치 없음": (b, FINAL, NOL),
    "최종 설정, 지정가 진입": (b, replace(FINAL, entry="maker"), LIM),
    "최종 설정, 수수료 0": (b, replace(FINAL, taker_fee=0, slip=0), NOL),
    "(비교) 1차: 보완 전": ({**b, "thr_buy": b["thr_buy_all"]}, replace(FINAL, min_prior=0), LIM),
}
ex = ccxt.binanceusdm({"enableRateLimit": True})


def fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{symbol.split('/')[0]}_{start:%Y%m%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows, since = [], int(start.timestamp() * 1000)
    while since < int(end.timestamp() * 1000):
        batch = ex.fetch_ohlcv(symbol, "5m", since=since, limit=1500)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 300_000
    c = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    c["timestamp"] = pd.to_datetime(c.t, unit="ms", utc=True) + pd.Timedelta(minutes=5)
    c = c[c.timestamp <= end].drop(columns="t").reset_index(drop=True)
    c.to_parquet(path)
    return c


def pick_charts() -> list[tuple[str, pd.Timestamp]]:
    markets = ex.load_markets()
    symbols = sorted(s for s, m in markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")
                     and m.get("base") not in ("BTC", "ETH"))            # 알트만
    rng = np.random.default_rng(SEED)
    rng.shuffle(symbols)
    last_start = pd.Timestamp.now(tz="UTC").floor("D") - LENGTH
    picks = []
    for s in symbols:
        if len(picks) == N:
            break
        first = ex.fetch_ohlcv(s, "5m", since=0, limit=1)
        if not first:
            continue
        lo = max(FIRST, (pd.Timestamp(first[0][0], unit="ms", tz="UTC") + WARMUP).ceil("D"))
        if lo >= last_start:
            continue
        picks.append((s, lo + pd.Timedelta(days=int(rng.integers(0, (last_start - lo).days + 1)))))
    return picks


def run_all(name: str, c: pd.DataFrame, start, end) -> list[dict]:
    out = []
    for v, (bundle, p, lim) in VARIANTS.items():
        sim, tr = simulate(c, bundle, p, start, end, lim=lim)
        out.append(dict(차트=name, 설정=v, **stats(sim, tr)))
    return out


def main():
    rows = []
    print("== ① BTC·ETH 2024-01 이후 ==", flush=True)
    for coin in ("BTC", "ETH"):
        c = pd.read_parquet(f"data/binance_{coin}_5m_202110_202610.parquet")
        c = c[c.timestamp >= "2023-12-01"]
        rows += run_all(f"{coin} 2024-01~2026-10", c, "2024-01-01", None)
    picks = pick_charts()
    print(f"== ② 무작위 차트 {len(picks)}개 (seed {SEED}):", ", ".join(f"{s.split('/')[0]} {d:%Y-%m}" for s, d in picks), flush=True)
    for s, start in picks:
        c = fetch(s, start - WARMUP, start + LENGTH)
        rows += run_all(f"{s.split('/')[0]} {start:%Y-%m-%d}~", c, start.tz_localize(None), (start + LENGTH).tz_localize(None))
        print("  완료:", s, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"data/final_test_seed{SEED}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(symbol=s_, start=str(t_)) for s_, t_ in picks]).to_csv(f"data/random_picks_seed{SEED}.csv", index=False)

    main_ = d[d.설정 == "최종 설정 (시장가, 안전장치 있음)"]
    show = main_[["차트", "수익", "단순보유", "최대낙폭", "거래수", "승률", "거래당_수익bp", "시장노출시간"]].copy()
    for k in ("수익", "단순보유"):
        show[k] = show[k].map("{:+.1%}".format)
    for k in ("최대낙폭", "승률", "시장노출시간"):
        show[k] = show[k].map("{:.1%}".format)
    print("\n== 차트별 결과: 최종 설정, 초기 10,000 USDT, 시장가 수수료, 계좌 안전장치 포함 ==")
    print(show.round(1).to_string(index=False))
    print("\n== 요약 ==")
    is_main = d.차트.str.startswith(("BTC 2024", "ETH 2024"))
    for v in VARIANTS:
        for label, g in (("BTC·ETH", d[is_main & (d.설정 == v)]), ("무작위 차트", d[~is_main & (d.설정 == v)])):
            print(f"[{v} · {label}] 평균 {g.수익.mean():+.1%}, 중앙값 {g.수익.median():+.1%}, 수익 난 차트 {(g.수익 > 0).sum()}/{len(g)}, "
                  f"최악 {g.수익.min():+.1%}, 최고 {g.수익.max():+.1%}, 평균 최대낙폭 {g.최대낙폭.mean():.1%}, 거래당 {g.거래당_수익bp.mean():+.1f}bp, 단순 보유 평균 {g.단순보유.mean():+.1%}")


if __name__ == "__main__":
    main()
