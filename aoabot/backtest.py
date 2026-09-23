"""1시간봉 백테스트. 실매매와 같은 decide()와 같은 안전장치를 쓴다.

사용: python -m aoabot.backtest data\\candles_XBTUSD_1h.parquet data\\candles_XBTUSD_1h_2021_2026.parquet --start 2022-01-01
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

from aoabot import risk
from aoabot.features import make_features
from aoabot.strategy import Add, Enter, Exit, Params, Position, decide, stop_hit


def load_candles(paths: list[str]) -> pd.DataFrame:
    c = pd.concat([pd.read_parquet(p) for p in paths]).drop_duplicates("timestamp").sort_values("timestamp")
    return c.reset_index(drop=True)


def run(candles: pd.DataFrame, p: Params, lim: risk.Limits, start=None, end=None,
        equity0: float = 10_000.0, fee: float = 0.0005, slip: float = 0.0002, symbol: str = "BTC"):
    f = make_features(candles)
    bars = candles.set_index(candles.timestamp.dt.tz_localize(None))[["open", "high", "low"]]
    f = f.join(bars).loc[start:end]

    cash, pos, rs = equity0, None, risk.RiskState()
    trades, curve = [], []
    t_fees = 0.0

    def fill(side: int, qty: float, px: float) -> float:
        nonlocal cash, t_fees
        fp = px * (1 + side * slip)
        cost = qty * fp * fee
        cash -= cost
        t_fees += cost
        return fp

    def close(t, px: float, reason: str):
        nonlocal cash, pos, t_fees
        fp = fill(-pos.side, pos.qty, px)
        pnl = pos.side * pos.qty * (fp - pos.entry_px)
        cash += pnl
        trades.append(dict(symbol=symbol, side=pos.side, opened=pos.opened_at, closed=t, entry=pos.entry_px, exit=fp,
                           qty=pos.qty, adds=pos.adds, pnl=pnl - t_fees, ret=pos.side * (fp / pos.entry_px - 1), reason=reason))
        pos, t_fees = None, 0.0

    for t, row in f.iterrows():
        if pos is not None and stop_hit(pos, row.low, row.high):
            gap = row.open < pos.stop_px if pos.side == 1 else row.open > pos.stop_px
            close(t, row.open if gap else pos.stop_px, "손절")
        equity = cash + (pos.side * pos.qty * (row.px - pos.entry_px) if pos else 0.0)
        risk.update(rs, t.strftime("%Y-%m-%d"), equity, lim)
        if rs.halted:
            if pos is not None:
                close(t, row.px, "최대 낙폭 정지")
            curve.append((t, cash))
            continue
        d = decide(row, pos, equity, p)
        if isinstance(d, Enter):
            ok, _ = risk.can_open(rs, equity, lim)
            if ok:
                fp = fill(d.side, d.qty, row.px)
                pos = Position(symbol, d.side, d.qty, fp, fp, d.qty, d.stop_px, d.atr_px, t.isoformat())
        elif isinstance(d, Add):
            fp = fill(pos.side, d.qty, row.px)
            pos.entry_px = (pos.entry_px * pos.qty + fp * d.qty) / (pos.qty + d.qty)
            pos.qty += d.qty
            pos.stop_px = d.stop_px
            pos.adds += 1
        elif isinstance(d, Exit):
            close(t, row.px, d.reason)
        curve.append((t, cash + (pos.side * pos.qty * (row.px - pos.entry_px) if pos else 0.0)))

    eq = pd.Series(dict(curve))
    return eq, pd.DataFrame(trades), rs


def summarize(eq: pd.Series, tr: pd.DataFrame, bench: pd.Series | None = None) -> str:
    r = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    lines = [
        f"기간 {eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}",
        f"총수익 {eq.iloc[-1] / eq.iloc[0] - 1:+.1%}  연복리 {(eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1:+.1%}  "
        f"샤프 {r.mean() / r.std() * np.sqrt(8760):.2f}  최대낙폭 {dd:.1%}",
    ]
    if bench is not None:
        b = bench.loc[eq.index[0]:eq.index[-1]]
        lines.append(f"(비교) 단순 보유 {b.iloc[-1] / b.iloc[0] - 1:+.1%}  최대낙폭 {(b / b.cummax() - 1).min():.1%}")
    if len(tr):
        win = tr[tr.pnl > 0]
        lose = tr[tr.pnl <= 0]
        lines.append(f"거래 {len(tr)}회  승률 {len(win) / len(tr):.0%}  평균 이익 {win.ret.mean():+.2%}  "
                     f"평균 손실 {lose.ret.mean():+.2%}  추가 진입 {int((tr.adds > 0).sum())}회")
        lines.append("청산 사유: " + ", ".join(f"{k} {v}" for k, v in tr.reason.str.replace(r"보유 \d+시간 경과", "시간 경과", regex=True)
                                                .value_counts().items()))
    yr = eq.resample("YE").last()
    yr = yr / pd.concat([pd.Series([eq.iloc[0]]), yr.iloc[:-1]]).values - 1
    lines.append("연도별: " + "  ".join(f"{i:%Y} {v:+.1%}" for i, v in yr.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="BitMEX 1시간봉 parquet (scripts/02_fetch_candles.py로 받은 것)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--fee", type=float, default=0.0005, help="편도 수수료 (바이낸스 선물 시장가 0.05%%)")
    ap.add_argument("--trades", help="거래 목록을 저장할 csv 경로")
    for k, v in asdict(Params()).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    p = Params(**{k: getattr(a, k) for k in asdict(Params())})
    c = load_candles(a.paths)
    eq, tr, _ = run(c, p, risk.Limits(), a.start, a.end, fee=a.fee)
    bench = c.set_index(c.timestamp.dt.tz_localize(None)).close
    print(summarize(eq, tr, bench))
    if a.trades:
        tr.to_csv(a.trades, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
