import numpy as np
import pandas as pd
import pytest

from aoabot import risk
from aoabot.backtest import run
from aoabot.strategy import Add, Enter, Exit, Params, Position, decide

T0 = pd.Timestamp("2024-01-01 12:00")


def row(r_4h=0.0, vol_regime=2.0, px=100.0, atr24=0.01, t=T0):
    return pd.Series({"px": px, "r_4h": r_4h, "vol_regime": vol_regime, "atr24": atr24}, name=t)


def pos(side=1, px=100.0, t=T0, adds=0):
    return Position("BTC", side, 10.0, px, px, 10.0, px - side * 3.0, 1.0, t.isoformat(), adds)


def test_enters_against_sharp_move_only_when_volatile():
    p = Params()
    d = decide(row(r_4h=-0.05), None, 10_000, p)
    assert isinstance(d, Enter) and d.side == 1 and d.stop_px < 100
    assert decide(row(r_4h=0.05), None, 10_000, p).side == -1
    assert decide(row(r_4h=-0.05, vol_regime=1.0), None, 10_000, p) is None
    assert decide(row(r_4h=-0.01), None, 10_000, p) is None


def test_size_risks_one_percent_and_respects_leverage_cap():
    p = Params()
    d = decide(row(r_4h=-0.05, atr24=0.01), None, 10_000, p)
    assert (100 - d.stop_px) * d.qty == pytest.approx(100)          # 손절 시 자산의 1%
    d = decide(row(r_4h=-0.05, atr24=0.0001), None, 10_000, p)       # 손절폭이 아주 좁아도
    assert d.qty * 100 == pytest.approx(20_000)                       # 명목가치는 자산 × 2배까지


def test_adds_only_in_profit_never_averages_down():
    p = Params()
    assert isinstance(decide(row(px=101.5, t=T0 + pd.Timedelta(hours=1)), pos(), 10_000, p), Add)
    assert decide(row(px=97.0, t=T0 + pd.Timedelta(hours=1)), pos(), 10_000, p) is None
    assert decide(row(px=101.5, t=T0 + pd.Timedelta(hours=1)), pos(adds=1), 10_000, p) is None


def test_time_exit():
    d = decide(row(t=T0 + pd.Timedelta(hours=12)), pos(), 10_000, Params())
    assert isinstance(d, Exit)


def test_risk_limits():
    lim = risk.Limits(daily_loss=0.03, max_drawdown=0.25)
    rs = risk.RiskState()
    risk.update(rs, "2024-01-01", 10_000, lim)
    assert risk.can_open(rs, 9_800, lim)[0]
    assert not risk.can_open(rs, 9_600, lim)[0]
    risk.update(rs, "2024-01-02", 9_600, lim)                        # 다음 날엔 기준이 초기화
    assert risk.can_open(rs, 9_600, lim)[0]
    risk.update(rs, "2024-01-03", 7_400, lim)
    assert rs.halted and not risk.can_open(rs, 7_400, lim)[0]


def test_backtest_stop_fills_at_stop_price():
    # 1000시간 횡보 → 한 봉에 100→97 급락해 롱 진입 → 다음 봉이 80으로 갭 하락
    n = 1000
    close = np.full(n, 100.0)
    close[-2:] = [97, 80]
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    c = pd.DataFrame({"timestamp": ts, "open": np.r_[100.0, close[:-1]], "close": close,
                      "high": close + 0.1, "low": close - 0.1, "volume": 1.0})
    c.loc[n - 2, ["open", "high", "low"]] = [100, 100, 97]
    c.loc[n - 1, ["open", "high", "low"]] = [80, 80, 80]
    _, tr, _ = run(c, Params(vol_gate=0.0), risk.Limits(), fee=0, slip=0)
    assert len(tr) == 1 and tr.iloc[0].side == 1 and tr.iloc[0].reason == "손절"
    assert tr.iloc[0].exit == pytest.approx(80.0)                     # 갭 하락이면 시가에 체결


def test_entry_only_in_allowed_kst_hours():
    p = Params(entry_hours_kst=[4, 5])
    utc_19 = pd.Timestamp("2024-01-01 19:00")   # 한국 04시
    utc_03 = pd.Timestamp("2024-01-01 03:00")   # 한국 12시
    assert isinstance(decide(row(r_4h=-0.05, t=utc_19), None, 10_000, p), Enter)
    assert decide(row(r_4h=-0.05, t=utc_03), None, 10_000, p) is None
    # 보유 중인 포지션의 청산은 시각과 무관
    assert isinstance(decide(row(t=utc_03 + pd.Timedelta(hours=12)), pos(t=utc_03), 10_000, p), Exit)


def test_daily_entry_cap_resets_next_day():
    lim = risk.Limits(max_entries_per_day=2)
    rs = risk.RiskState()
    risk.update(rs, "2024-01-01", 10_000, lim)
    risk.record_entry(rs); risk.record_entry(rs)
    assert not risk.can_open(rs, 10_000, lim)[0]
    risk.update(rs, "2024-01-02", 10_000, lim)
    assert risk.can_open(rs, 10_000, lim)[0]
