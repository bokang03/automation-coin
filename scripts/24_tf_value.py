"""봉 단위(1시간·일·주·월)로 워뇨띠의 매매를 다시 집계해, 그 봉에서 워뇨띠의 판단이 실제로 값어치가 있었는지 본다. XBTUSD.

  ① 그 봉 동안 순매수한 봉 / 순매도한 봉 뒤의 가격 (다음 1·2·3·5봉, 시장 평균 흐름 제외)
  ② 봉이 끝날 때의 워뇨띠 포지션을 다음 봉 동안 그대로 따라 했을 때의 성과
"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
TFS = {"1시간봉": "1h", "일봉": "1D", "주봉": "W-SUN", "월봉": "ME"}
ANN = {"1시간봉": 8760, "일봉": 365, "주봉": 52, "월봉": 12}


def load():
    tr = pd.read_parquet("data/trades_pos.parquet")
    x = tr[tr.symbol == "XBTUSD"].set_index("ts")
    c = pd.read_parquet("data/candles_XBTUSD_1h.parquet")
    c = c.set_index(c.timestamp.dt.tz_localize(None)).sort_index()
    w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
    w["date"] = pd.to_datetime(w.date)
    bal = w.groupby("date").walletbalance.last() / 1e8
    return x, c, bal


def candles(c: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1시간봉(마감 시각 기준)을 상위 봉으로 묶는다. 인덱스 = 봉 마감 시각."""
    if rule == "1h":
        return c[["open", "high", "low", "close", "volume"]].copy()
    o = c.copy()
    o.index = o.index - pd.Timedelta(hours=1)                       # 시작 시각 기준으로 묶은 뒤
    g = o.resample(rule, label="right", closed="left" if rule == "1D" else "right")
    if rule == "1D":
        g = o.resample("1D", label="left", closed="left")
    k = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), n=("close", "size")).dropna()
    if rule == "1D":
        k.index = k.index + pd.Timedelta(days=1)
    else:                                                            # W-SUN, ME 는 마지막 날 00:00 로 표시되므로 다음 날 00:00(마감 시각)로 옮긴다
        k.index = k.index.normalize() + pd.Timedelta(days=1)
    return k


def his_by_candle(x: pd.DataFrame, bal: pd.Series, k: pd.DataFrame) -> pd.DataFrame:
    pos = x.pos.groupby(level=0).last()
    p = pos.reindex(pos.index.union(k.index)).ffill().reindex(k.index).fillna(0)        # 봉 마감 시각의 포지션(계약, USD)
    eq = bal.reindex(bal.index.union(k.index.normalize())).ffill().reindex(k.index.normalize()).to_numpy() * k.close.to_numpy()
    d = k.copy()
    d["lev"] = (p / eq).to_numpy()                                    # 봉 마감 때 포지션(자산 대비 배수)
    d["flow"] = (p.diff() / eq).to_numpy()                            # 그 봉 동안의 순매수(+)·순매도(-)
    return d


if __name__ == "__main__":
    x, c, bal = load()
    for name, rule in TFS.items():
        k = candles(c, rule)
        d = his_by_candle(x, bal, k).loc["2018-04-01":"2021-12-31"]
        lc = np.log(d.close)
        print("\n" + "=" * 110)
        print(f"{name}: {len(d)}개 (2018-04 ~ 2021-12). 워뇨띠가 순매수한 봉 {int((d.flow > 0.02).sum())}, 순매도한 봉 {int((d.flow < -0.02).sum())}, 거의 변화 없는 봉 {int((d.flow.abs() <= 0.02).sum())}")
        rows = {}
        for label, m in (("순매수한 봉 뒤", d.flow > 0.02), ("크게 순매수(상위 절반)", d.flow >= d.flow[d.flow > 0.02].median()),
                         ("순매도한 봉 뒤", d.flow < -0.02), ("크게 순매도(상위 절반)", d.flow <= d.flow[d.flow < -0.02].median())):
            sgn = 1 if "매수" in label else -1
            r = {"봉수": int(m.sum())}
            for h in (1, 2, 3, 5):
                fwd = lc.shift(-h) - lc
                v = sgn * (fwd[m] - fwd.mean())
                r[f"+{h}봉(%)"] = round(float(v.mean() * 100), 2)
                r[f"+{h}봉 t"] = round(float(v.mean() / (v.std() / np.sqrt(v.notna().sum()))), 1) if v.notna().sum() > 2 else np.nan
            rows[label] = r
        print("① 그 봉 종가에 워뇨띠와 같은 방향으로 들어갔을 때 이후 수익률 (시장 평균 흐름 제외, + = 유리, t = 통계적 유의도·2 이상이면 의미 있음)")
        print(pd.DataFrame(rows).T.to_string())
        nxt = d.close.pct_change().shift(-1)
        res = {}
        for label, sig in (("계약 포지션 방향(롱 +1/숏 -1)", np.sign(d.lev.where(d.lev.abs() > 0.05, 0))), ("계약 포지션 배수 그대로", d.lev.clip(-3, 3)),
                           ("BTC 증거금 포함 순노출(1+배수)", (1 + d.lev).clip(-3, 3)), ("(비교) 항상 1배 보유", pd.Series(1.0, index=d.index))):
            r = (sig * nxt).dropna()
            res[label] = {"누적(%)": round(float(((1 + r).prod() - 1) * 100), 0), "샤프": round(float(r.mean() / r.std() * np.sqrt(ANN[name])), 2),
                          "평균 포지션": round(float(sig.mean()), 2)}
        print("② 봉이 끝날 때의 워뇨띠 포지션을 다음 봉 동안 그대로 따라 했다면 (수수료 없음)")
        print(pd.DataFrame(res).T.to_string())
        d.to_parquet(f"data/his_candles_{rule.replace('-', '')}.parquet")
