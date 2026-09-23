"""행동 복제 실험: 시간봉 피처 → 워뇨띠 XBTUSD 포지션 방향 예측, 그리고 백테스트."""
import pandas as pd, numpy as np, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
sys.path.insert(0, ".")
from aoabot.features import make_features
pd.set_option("display.width", 220)
FEE = 0.00075  # 테이커 수수료(편도)

c = pd.read_parquet("data/candles_XBTUSD_1h.parquet"); f = make_features(c)
# BitMEX 버킷 timestamp = 봉 종료 시각. f.index=t 는 t 시점까지 알려진 정보.
tr = pd.read_parquet("data/trades_pos.parquet"); x = tr[tr.symbol == "XBTUSD"].set_index("ts")
pos = x.pos.resample("1h", label="right", closed="right").last().reindex(f.index).ffill().fillna(0)  # t 시점 포지션(USD 계약)
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"]); w["date"] = pd.to_datetime(w.date)
bal = (w.groupby("date").walletbalance.last() / 1e8).reindex(f.index.normalize()).ffill().values
f["lev"] = pos.values / (bal * f.px.values)          # 포지션 레버리지(부호 포함)
f["net"] = 1 + f.lev                                  # 순 달러 노출: BTC 증거금(+1) + 인버스 계약 포지션
f["y"] = np.select([f.net > 0.25, f.net < -0.25], [1, -1], 0)  # 1 순롱, -1 순숏, 0 거의 중립(헤지)
f["fwd"] = np.log(f.px).shift(-1) - np.log(f.px)     # 다음 1시간 수익률
f = f.loc["2018-04":"2021-12"].dropna(subset=["fwd"])

def bt(p, name):
    p = pd.Series(p, index=f.index).clip(-1, 1)
    gross = p * f.fwd; cost = p.diff().abs().fillna(0) * FEE
    net = gross - cost
    out = {}
    for per, s in (("2018-2020", slice("2018", "2020")), ("2021(테스트)", slice("2021", "2021"))):
        n = net.loc[s]
        out[per] = f"수익 {np.expm1(n.sum())*100:8.1f}%  샤프 {n.mean()/n.std()*np.sqrt(24*365):5.2f}  회전 {p.loc[s].diff().abs().sum():6.0f}"
    print(f"{name:28s}", " | ".join(f"{k}: {v}" for k, v in out.items()))

print("== 기준선: 1시간 해상도로 본 워뇨띠의 실제 포지션 ==")
bt(f.y, "워뇨띠의 순노출 방향 1x 따라하기")
bt(f.net.clip(-3, 3) / 3, "워뇨띠의 순노출 비율(/3) 따라하기")
bt(np.ones(len(f)), "비트코인 단순 보유(1x)")
print("\n무포지션 비율:", (f.y == 0).mean().round(3), " 롱:", (f.y == 1).mean().round(3), " 숏:", (f.y == -1).mean().round(3))
print("계약 레버리지 분포:", f.lev.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
print("순노출 분포:", f.net.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())

feats = [k for k in f.columns if k not in ("lev", "net", "y", "fwd", "px")]
tr_m = f.index < "2020-07-01"; va_m = (f.index >= "2020-07-01") & (f.index < "2021-01-01"); te_m = f.index >= "2021-01-01"
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
m.fit(f.loc[tr_m, feats], f.y[tr_m])
proba = m.predict_proba(f[feats])
pred = m.classes_[proba.argmax(1)]
for nm, mk in (("학습", tr_m), ("검증", va_m), ("테스트2021", te_m)):
    acc = (pred[mk] == f.y[mk]).mean(); maj = f.y[mk].value_counts(normalize=True).max()
    print(f"{nm}: 방향 일치율 {acc:.3f} (최빈값만 찍을 때 {maj:.3f})")
print("\n== 복제 모델 백테스트 ==")
bt(proba[:, 2] - proba[:, 0], "복제모델(롱확률-숏확률)")
bt(pred, "복제모델(argmax)")
pi = permutation_importance(m, f.loc[te_m, feats], f.y[te_m], n_repeats=3, random_state=0)
print("\n2021 순열 중요도:", pd.Series(pi.importances_mean, feats).sort_values(ascending=False).head(8).round(4).to_dict())
