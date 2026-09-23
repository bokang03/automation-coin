"""분석에서 드러난 습관을 규칙으로 옮겨 검증한다: 변동성·거래량이 클 때만, 직전 4시간 움직임의 반대로 진입."""
import pandas as pd, numpy as np, sys, itertools
sys.path.insert(0, ".")
from aoabot.features import make_features
FEE = 0.00075
c = pd.read_parquet("data/candles_XBTUSD_1h.parquet"); f = make_features(c).loc["2018-04":"2021-12"]
f["fwd"] = np.log(f.px).shift(-1) - np.log(f.px); f = f.dropna(subset=["fwd"])

def run(th, vr, hold, s):
    g = f.loc[s]
    sig = np.where((g.vol_regime > vr) & (g.r_4h < -th), 1, np.where((g.vol_regime > vr) & (g.r_4h > th), -1, np.nan))
    p = pd.Series(sig, g.index).ffill(limit=hold - 1).fillna(0)
    net = p * g.fwd - p.diff().abs().fillna(0) * FEE
    return np.expm1(net.sum()), net.mean() / net.std() * np.sqrt(24 * 365) if net.std() > 0 else 0

rows = []
for th, vr, hold in itertools.product([0.01, 0.02, 0.03], [1.0, 1.3, 1.6], [2, 6, 12]):
    r_is, s_is = run(th, vr, hold, slice("2018", "2020")); r_os, s_os = run(th, vr, hold, slice("2021", "2021"))
    rows.append(dict(th=th, vr=vr, hold=hold, is_ret=r_is, is_sharpe=s_is, oos_ret=r_os, oos_sharpe=s_os))
d = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False)
pd.set_option("display.width", 200); print(d.round(3).head(10).to_string(index=False))
print("\n학습기간 최고 파라미터의 2021 성과:", d.iloc[0][["oos_ret", "oos_sharpe"]].round(3).to_dict())
print("27개 조합 중 2021 샤프>0 비율:", (d.oos_sharpe > 0).mean().round(2))
