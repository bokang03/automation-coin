"""학습한 정책을 워뇨띠가 매매하지 않은 무작위 차트에 돌린다.

차트 뽑는 법: 바이낸스 USDT 무기한 선물 전 종목을 난수(seed)로 섞고, 종목마다 2022-01-01 이후에서
시작일을 균등 난수로 뽑아 1년 구간을 쓴다. 워뇨띠의 자료는 2021-12에 끝나므로 전부 워뇨띠가 매매하지 않은 차트다.
사용: python scripts/11_random_backtest.py [차트 수=20] [seed=42]
"""
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.policy import load_bundle
from aoabot.policy_backtest import simulate, stats

pd.set_option("display.width", 250)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
WARMUP, LENGTH = pd.Timedelta(days=60), pd.Timedelta(days=365)   # 지표 계산용 준비 구간(50일 이동평균), 평가 구간
FIRST = pd.Timestamp("2022-01-01", tz="UTC")
CACHE = Path("data/random")
CACHE.mkdir(parents=True, exist_ok=True)
LIMITS = risk.Limits(daily_loss=0.03, max_drawdown=0.25)          # 계좌 안전장치(봇 설정과 동일)
NO_LIMITS = risk.Limits(daily_loss=9.9, max_drawdown=9.9)

ex = ccxt.binanceusdm({"enableRateLimit": True})


def fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{symbol.split('/')[0]}_{start:%Y%m%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows, since = [], int(start.timestamp() * 1000)
    while since < int(end.timestamp() * 1000):
        batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=1500)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 3_600_000
    c = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    c["timestamp"] = pd.to_datetime(c.t, unit="ms", utc=True) + pd.Timedelta(hours=1)   # 봉 마감 시각
    c = c[c.timestamp <= end].drop(columns="t").reset_index(drop=True)
    c.to_parquet(path)
    return c


def pick_charts() -> list[tuple[str, pd.Timestamp]]:
    markets = ex.load_markets()
    symbols = sorted(s for s, m in markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(symbols)
    last_start = pd.Timestamp.now(tz="UTC").floor("D") - LENGTH
    picks = []
    for s in symbols:
        if len(picks) == N:
            break
        first = ex.fetch_ohlcv(s, "1h", since=0, limit=1)
        if not first:
            continue
        lo = max(FIRST, pd.Timestamp(first[0][0], unit="ms", tz="UTC") + WARMUP)
        if lo >= last_start:
            continue                                   # 상장이 늦어 1년 구간을 못 채우는 종목
        start = lo + pd.Timedelta(days=int(rng.integers(0, (last_start - lo).days + 1)))
        picks.append((s, start))
    return picks


def shuffled(c: pd.DataFrame, rng) -> pd.DataFrame:
    """봉의 시간 순서를 섞은 가짜 차트: 수익률 분포는 같지만 패턴이 없다."""
    r = np.log(c.close / c.open).to_numpy()
    hi, lo = np.log(c.high / c.open).to_numpy(), np.log(c.low / c.open).to_numpy()
    idx = rng.permutation(len(c))
    o = c.open.iloc[0] * np.exp(np.r_[0, np.cumsum(r[idx])[:-1]])
    return pd.DataFrame({"timestamp": c.timestamp.to_numpy(), "open": o, "high": o * np.exp(hi[idx]), "low": o * np.exp(lo[idx]),
                         "close": o * np.exp(r[idx]), "volume": c.volume.to_numpy()[idx]}).assign(timestamp=lambda d: pd.to_datetime(d.timestamp, utc=True))


def main():
    bundle = load_bundle()
    picks = pick_charts()
    print(f"무작위 차트 {len(picks)}개 (seed {SEED}):", ", ".join(f"{s.split('/')[0]} {d:%Y-%m}" for s, d in picks))
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for s, start in picks:
        c = fetch(s, start - WARMUP, start + LENGTH)
        a, b = start.tz_localize(None), (start + LENGTH).tz_localize(None)
        name = f"{s.split('/')[0]} {start:%Y-%m-%d}~"
        for label, kw in {"시장가(0.05%+0.02%)": dict(fee=0.0005, slip=0.0002), "지정가 가정(0.02%)": dict(fee=0.0002, slip=0.0)}.items():
            rows.append(dict(차트=name, 비용=label, 안전장치="있음", **stats(simulate(c, bundle, a, b, lim=LIMITS, **kw))))
        rows.append(dict(차트=name, 비용="시장가(0.05%+0.02%)", 안전장치="없음", **stats(simulate(c, bundle, a, b, lim=NO_LIMITS))))
        rows.append(dict(차트=name, 비용="수수료 0", 안전장치="없음", **stats(simulate(c, bundle, a, b, fee=0, slip=0, lim=NO_LIMITS))))
        rows.append(dict(차트=name + " (순서 섞음)", 비용="수수료 0", 안전장치="없음",
                         **stats(simulate(shuffled(c, rng), bundle, a, b, fee=0, slip=0, lim=NO_LIMITS))))
        print("  완료:", name, flush=True)
    d = pd.DataFrame(rows)
    d["매매판단_기여"] = d.수익 - d.평균포지션만_보유
    d.to_csv(f"data/random_backtest_seed{SEED}.csv", index=False, encoding="utf-8-sig")

    main_ = d[(d.비용 == "시장가(0.05%+0.02%)") & (d.안전장치 == "있음")]
    show = main_[["차트", "수익", "단순보유", "최대낙폭", "평균포지션", "매매횟수_일"]].copy()
    for k in ("수익", "단순보유", "최대낙폭"):
        show[k] = show[k].map("{:+.1%}".format)
    print("\n== 차트별 결과: 초기 10,000 USDT, 시장가 수수료, 계좌 안전장치 포함 ==")
    print(show.round(2).to_string(index=False))

    print("\n== 요약 ==")
    real = ~d.차트.str.contains("섞음")
    for (cost, guard, kind), g in d.assign(종류=np.where(real, "실제 차트", "순서 섞은 차트")).groupby(["비용", "안전장치", "종류"], sort=False):
        print(f"[{kind} · {cost} · 안전장치 {guard}] 평균 {g.수익.mean():+.1%}, 중앙값 {g.수익.median():+.1%}, "
              f"수익 난 차트 {(g.수익 > 0).mean():.0%} ({(g.수익 > 0).sum()}/{len(g)}), 최악 {g.수익.min():+.1%}, 최고 {g.수익.max():+.1%}, "
              f"매매판단 기여 평균 {g.매매판단_기여.mean():+.1%}, 단순 보유 평균 {g.단순보유.mean():+.1%}")


if __name__ == "__main__":
    main()
