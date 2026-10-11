import numpy as np
import pandas as pd
import pytest

from aoabot.policy import BUY, FEATS, MARKET_FEATS, NONE, SELL, BinnedLogit, OwnState, apply_fill, choose, decide_x

T = pd.Timestamp("2024-01-01 12:00")


def test_apply_fill_average_reduce_and_flip():
    st = OwnState()
    assert apply_fill(st, 2.0, 100.0, T) == 0 and st.qty == 2 and st.avg_px == 100
    apply_fill(st, 2.0, 110.0, T)
    assert st.qty == 4 and st.avg_px == pytest.approx(105)                 # 추가 매수는 평균가를 옮긴다
    assert apply_fill(st, -1.0, 115.0, T) == pytest.approx(10) and st.avg_px == pytest.approx(105)   # 축소는 평균가 유지
    assert apply_fill(st, -5.0, 95.0, T) == pytest.approx(-30)             # 남은 3개 청산 손실
    assert st.qty == -2 and st.avg_px == 95                                # 넘친 2개는 새 숏
    assert apply_fill(st, 2.0, 90.0, T) == pytest.approx(10) and st.qty == 0 and st.avg_px == 0


class FixedModel:
    def __init__(self, p):
        self.p = np.array([p])

    def predict_proba(self, x):
        return self.p


def bundle(p):
    return dict(model=FixedModel(p), thr_buy=0.2, thr_sell=0.2, lev_min=-2.0, lev_max=1.0,
                size={(BUY, "open"): 0.4, (BUY, "reduce"): 0.5, (SELL, "open"): 0.3, (SELL, "reduce"): 0.6})


def test_choose_uses_calibrated_thresholds():
    assert choose(bundle([0.1, 0.8, 0.1]), np.array([0.1, 0.8, 0.1])) == NONE
    assert choose(bundle(None), np.array([0.1, 0.6, 0.3])) == BUY
    assert choose(bundle(None), np.array([0.5, 0.2, 0.3])) == SELL          # 둘 다 넘으면 더 크게 넘은 쪽


def test_decide_sizes_from_bundle_and_respects_position_limits():
    m = np.zeros(len(MARKET_FEATS))
    dq, act = decide_x(bundle([0.05, 0.55, 0.4]), m, 100.0, OwnState(), 10_000, T)
    assert act == BUY and dq * 100 / 10_000 == pytest.approx(0.4)          # 새 진입 크기
    st = OwnState(qty=90.0, avg_px=100.0)                                   # 롱 0.9배, 한계는 1.0배
    dq, _ = decide_x(bundle([0.05, 0.55, 0.4]), m, 100.0, st, 10_000, T)
    assert (st.qty + dq) * 100 / 10_000 == pytest.approx(1.0)
    dq, act = decide_x(bundle([0.4, 0.55, 0.05]), m, 100.0, st, 10_000, T)
    assert act == SELL and dq * 100 / 10_000 == pytest.approx(-0.6)        # 줄일 때 크기
    assert decide_x(bundle([0.4, 0.55, 0.05]), np.full(len(MARKET_FEATS), np.nan), 100.0, st, 10_000, T) == (0.0, NONE)


def test_binned_logit_learns_and_roundtrips():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, len(FEATS)))
    y = np.where(X[:, 0] > 0.8, BUY, np.where(X[:, 0] < -0.8, SELL, NONE))
    m = BinnedLogit(n_bins=5, l2=1.0).fit(X, y)
    p = m.predict_proba(X)
    assert p[X[:, 0] > 1.5, 2].mean() > 0.8 and p[X[:, 0] < -1.5, 0].mean() > 0.8
    m2 = BinnedLogit.from_dict(m.to_dict())
    assert np.allclose(m2.predict_proba(X[:50]), p[:50])


class SeqModel:
    """정해 둔 봉에서만 매수 확률 1을 내는 가짜 모델."""

    def __init__(self, buy_rows, n):
        self.p = np.tile([0.0, 1.0, 0.0], (n, 1))
        self.p[list(buy_rows)] = [0.0, 0.0, 1.0]

    def predict_proba(self, X):
        return self.p[-len(X):] if len(X) != len(self.p) else self.p


def _flat_candles(n):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    px = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"timestamp": ts, "open": px, "high": px + 0.05, "low": px - 0.05, "close": px, "volume": 1.0})


def test_timing_enters_only_after_a_prior_signal_and_exits_after_hold():
    from aoabot.features5 import WARMUP_BARS
    from aoabot.timing import TimingParams, simulate
    n = WARMUP_BARS + 400
    c = _flat_candles(n)
    first, second, lone = WARMUP_BARS + 50, WARMUP_BARS + 60, WARMUP_BARS + 300
    bundle = dict(model=SeqModel([first, second, lone], n), thr_buy=0.5, thr_sell=0.5)
    p = TimingParams(size=0.4, max_lev=0.4, hold_bars=48, min_prior=1, taker_fee=0, slip=0)
    sim, tr = simulate(c, bundle, p)
    assert len(tr) == 1                                             # 첫 신호와 외톨이 신호는 건너뛴다
    t = tr.iloc[0]
    assert t.opened == sim.index[second] and t.bars == 48
    assert t.ret == pytest.approx(c.close[second + 48] / c.close[second] - 1)
    assert sim.lev.iloc[second] == pytest.approx(0.4, rel=1e-3)
    sim0, tr0 = simulate(c, bundle, TimingParams(size=0.4, max_lev=0.4, hold_bars=48, min_prior=0, taker_fee=0, slip=0))
    assert len(tr0) == 2 and tr0.iloc[0].opened == sim0.index[first]


def test_weekly_decide_uses_plan_and_last_completed_week():
    from aoabot import weekly
    from aoabot.features_k import K_FEATS

    class M:
        def __init__(self, p):
            self.p = np.array([p])

        def predict_proba(self, x):
            assert x.shape == (1, len(K_FEATS))
            return self.p

    days = pd.date_range("2024-01-02", periods=100, freq="D")          # 일봉 마감 시각. 2024-01-01 은 월요일
    px = 100 + np.arange(100.0)
    d = pd.DataFrame({"open": px, "high": px + 2, "low": px - 2, "close": px + 1, "volume": 10.0}, index=days)
    b = dict(model=M([0.6, 0.3, 0.1]), thr_buy=0.5, thr_sell=0.5, plan={SELL: (1, 1)}, sizes={SELL: 1.0})
    wk, side, hold, size = weekly.decide(b, d)
    assert wk == pd.Timestamp("2024-04-08") and wk.day_name() == "Monday"      # 끝나지 않은 주는 쓰지 않는다
    assert (side, hold, size) == (1, 1, 1.0)                                      # 매도 신호에 롱 1주 (워뇨띠와 반대)
    b["model"] = M([0.2, 0.7, 0.1])
    assert weekly.decide(b, d)[1] == 0
