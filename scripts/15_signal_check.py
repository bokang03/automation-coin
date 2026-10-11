"""학습한 타이밍 신호가 다른 기간·다른 차트에서도 통하는지: 신호 뒤 가격 움직임을 본다.
사용: python scripts/15_signal_check.py <5분봉 parquet> <시작> <끝> [이름]"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.features5 import FEATS_5M, make_features_5m
from aoabot.policy import ROOT, load_bundle

HZ = {"+15분": 3, "+1시간": 12, "+2시간": 24, "+4시간": 48, "+8시간": 96, "+24시간": 288, "+72시간": 864}


def check(c: pd.DataFrame, start, end, name="") -> pd.DataFrame:
    b = load_bundle(ROOT / "models" / "aoa_timing.json")
    f = make_features_5m(c).loc[start:end].dropna(subset=["vol_regime"])
    p = b["model"].predict_proba(f[FEATS_5M].to_numpy(dtype=float))
    lc = np.log(f.px.to_numpy())
    n = len(f)
    days = (f.index[-1] - f.index[0]).days + 1
    rows = []
    for label, sig, sgn in (("매수 신호", p[:, 2] >= b["thr_buy"], 1), ("매도 신호", p[:, 0] >= b["thr_sell"], -1)):
        idx = np.flatnonzero(sig)
        r = dict(차트=name, 신호=label, 신호수=len(idx), 하루=round(len(idx) / days, 2), 신호있는날=int(pd.Series(f.index[idx].date).nunique()))
        for hn, k in HZ.items():
            j = idx[idx + k < n]
            drift = np.mean(lc[k:] - lc[:-k])
            v = sgn * (lc[j + k] - lc[j] - drift) * 1e4
            # 같은 날 신호는 서로 얽혀 있으므로 날짜별 평균을 낸 뒤 표준오차를 구한다
            daily = pd.Series(v).groupby(f.index[j].date).mean()
            r[hn] = round(float(v.mean()), 1)
            if hn in ("+4시간", "+24시간"):
                r[hn + " t"] = round(float(daily.mean() / (daily.std() / np.sqrt(len(daily)))), 1)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    c = pd.read_parquet(sys.argv[1])
    print(check(c, sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "").to_string(index=False))
