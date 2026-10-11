"""봉 단위 차트 상태. 1시간봉·일봉·주봉·월봉 어디에나 같은 정의를 쓴다 (단위는 '봉 개수').

그 봉이 끝난 시점까지의 정보만 쓴다. 과거 참조는 최대 10봉이다.
"""
import numpy as np
import pandas as pd

K_FEATS = ["r_1", "r_2", "r_3", "r_5", "r_10", "atr_ratio", "bar_range",
           "range_pos_3", "dd_high_3", "up_low_3", "range_pos_5", "dd_high_5", "up_low_5", "range_pos_10", "dd_high_10", "up_low_10",
           "vol_ratio", "vol_ratio_3", "body", "lower_wick", "upper_wick"]
K_NAMES = {"r_1": "이번 봉 수익률", "r_2": "최근 2봉 수익률", "r_3": "최근 3봉 수익률", "r_5": "최근 5봉 수익률", "r_10": "최근 10봉 수익률",
           "atr_ratio": "최근 3봉 변동폭 / 10봉 평균", "bar_range": "이번 봉 크기 / 10봉 평균",
           "range_pos_3": "최근 3봉 범위 안 위치", "dd_high_3": "3봉 고점 대비", "up_low_3": "3봉 저점 대비",
           "range_pos_5": "최근 5봉 범위 안 위치", "dd_high_5": "5봉 고점 대비", "up_low_5": "5봉 저점 대비",
           "range_pos_10": "최근 10봉 범위 안 위치", "dd_high_10": "10봉 고점 대비", "up_low_10": "10봉 저점 대비",
           "vol_ratio": "거래량 / 10봉 평균", "vol_ratio_3": "최근 3봉 거래량 / 10봉 평균",
           "body": "몸통(음봉 -1 ~ 양봉 +1)", "lower_wick": "아래꼬리 비중", "upper_wick": "위꼬리 비중"}


def make_features_k(k: pd.DataFrame) -> pd.DataFrame:
    """k: 인덱스 = 봉 마감 시각, 열 = open, high, low, close, volume."""
    o, h, l, c, v = k.open, k.high, k.low, k.close, k.volume
    f = pd.DataFrame(index=k.index)
    lc = np.log(c)
    for n in (1, 2, 3, 5, 10):
        f[f"r_{n}"] = lc.diff(n)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr10 = tr.rolling(10).mean()
    f["atr_ratio"] = tr.rolling(3).mean() / atr10
    f["bar_range"] = (h - l) / atr10
    for n in (3, 5, 10):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"range_pos_{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        f[f"dd_high_{n}"] = c / hi - 1
        f[f"up_low_{n}"] = c / lo - 1
    f["vol_ratio"] = v / v.rolling(10).mean()
    f["vol_ratio_3"] = v.rolling(3).mean() / v.rolling(10).mean()
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["px"] = c
    return f


def to_higher(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """일봉(인덱스 = 마감 시각)을 주봉('W') 또는 월봉('M')으로 묶는다. 끝나지 않은 마지막 봉은 버린다."""
    open_time = daily.index - pd.Timedelta(days=1)
    per = open_time.to_period("W-SUN" if rule == "W" else "M")
    g = daily.groupby(per)
    k = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    k["first"], k["last"] = g.apply(lambda d: d.index[0]), g.apply(lambda d: d.index[-1])
    end = (k.index + 1).start_time
    start = k.index.start_time
    whole = ((k["first"] - pd.Timedelta(days=1)) == start) & (k["last"] == end)       # 첫날부터 마지막 날까지 다 있는 봉만
    k.index = end
    return k[whole.to_numpy()].drop(columns=["first", "last"])
