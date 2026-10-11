"""일봉 전략 재시도: 일봉 차트 상태에 '그날 안의 5분봉 신호'를 더해 워뇨띠가 크게 산 날 / 판 날을 찾는다.

일봉 차트만으로는 워뇨띠가 산 날을 구분하지 못했다(AUC 0.61, scripts/25_tf_strategies.py). 워뇨띠의 일 단위 매매는 그날 안의
급락·급등에서 나왔으므로, 5분봉 타이밍 모델(models/aoa_timing.json)이 그날 낸 신호 수와 최대 확률을 입력에 더한다.
학습은 5분봉 예측이 있는 2020-01 ~ 2021-12. 채택 규칙과 검증 구간은 scripts/25_tf_strategies.py 와 같다.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.features5 import FEATS_5M, make_features_5m
from aoabot.features_k import K_FEATS, make_features_k
from aoabot.policy import BUY, CLASSES, NONE, ROOT, SELL, load_bundle

spec = importlib.util.spec_from_file_location("tfs", "scripts/25_tf_strategies.py")
tfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfs)

pd.set_option("display.width", 250)
tb = load_bundle(ROOT / "models" / "aoa_timing.json")
INTRA = ["n_buy_sig", "max_p_buy", "n_sell_sig", "max_p_sell"]
FEATS = K_FEATS + INTRA
INTRA_NAMES = {"n_buy_sig": "그날 5분봉 강한 매수 신호 수", "max_p_buy": "그날 5분봉 매수 확률 최댓값", "n_sell_sig": "그날 5분봉 매도 신호 수", "max_p_sell": "그날 5분봉 매도 확률 최댓값"}


def intraday(p: pd.DataFrame) -> pd.DataFrame:
    """p: 인덱스 = 5분봉 마감 시각, 열 = p_buy, p_sell. 돌려주는 표의 인덱스 = 일봉 마감 시각."""
    day_close = (p.index - pd.Timedelta(minutes=5)).floor("D") + pd.Timedelta(days=1)
    g = p.groupby(day_close)
    return pd.DataFrame({"n_buy_sig": g.p_buy.apply(lambda s: int((s >= tb["thr_buy"]).sum())), "max_p_buy": g.p_buy.max(),
                         "n_sell_sig": g.p_sell.apply(lambda s: int((s >= tb["thr_sell"]).sum())), "max_p_sell": g.p_sell.max()})


def bt(f, model, thr, plan, size, start, end):
    f = f.loc[start:end]
    ok = f[FEATS].notna().all(axis=1).to_numpy()
    p = model.predict_proba(f[FEATS].fillna(0).to_numpy(dtype=float))
    sig = {BUY: ok & (p[:, 2] >= thr[BUY]), SELL: ok & (p[:, 0] >= thr[SELL])}
    px = f.px.to_numpy()
    cash, qty, entry, ent_i, side, hold = 1.0, 0.0, 0.0, -1, 0, 0
    eq, trades = np.zeros(len(f)), []
    for i in range(len(f)):
        if side and i - ent_i >= hold:
            x = px[i] * (1 - side * tfs.SLIP)
            cash += qty * (x - entry) * side - qty * x * tfs.FEE
            trades.append(side * (x / entry - 1))
            side, qty = 0, 0.0
        if not side:
            s_ = BUY if (BUY in plan and sig[BUY][i]) else SELL if (SELL in plan and sig[SELL][i]) else 0
            if s_:
                side, hold = plan[s_]
                entry = px[i] * (1 + side * tfs.SLIP)
                qty = size[s_] * cash / entry
                cash -= qty * entry * tfs.FEE
                ent_i = i
        eq[i] = cash + (qty * (px[i] - entry) * side if side else 0.0)
    e = pd.Series(eq, index=f.index)
    r = e.pct_change().dropna()
    tr = np.array(trades)
    return dict(수익=e.iloc[-1] - 1, 샤프=float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0, 최대낙폭=float((e / e.cummax() - 1).min()),
                거래수=len(tr), 승률=float((tr > 0).mean()) if len(tr) else np.nan, 단순보유=px[-1] / px[0] - 1)


# ---------- 워뇨띠의 기간 ----------
k = tfs.his_candles("1d")
f = make_features_k(k).join(intraday(pd.read_parquet("data/timing_oof_2020_2021.parquet")[["p_buy", "p_sell"]]))
tr = pd.read_parquet("data/trades_pos.parquet")
pos = tr[tr.symbol == "XBTUSD"].set_index("ts").pos.groupby(level=0).last()
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
w["date"] = pd.to_datetime(w.date)
bal = w.groupby("date").walletbalance.last() / 1e8
p_ = pos.reindex(pos.index.union(k.index)).ffill().reindex(k.index).fillna(0)
eq = bal.reindex(bal.index.union(k.index.normalize())).ffill().reindex(k.index.normalize()).to_numpy() * k.close.to_numpy()
f["flow"] = (p_.diff() / eq).to_numpy()
d = f.loc["2020-01-02":"2021-12-31"].dropna(subset=FEATS + ["flow"]).copy()
med_b, med_s = d.flow[d.flow > 0.02].median(), d.flow[d.flow < -0.02].median()
d["y"] = np.select([d.flow >= med_b, d.flow <= med_s], [BUY, SELL], NONE)
n = len(d)
rate = {c: float((d.y == c).mean()) for c in CLASSES}
print(f"일봉 {n}개 (2020~2021). 워뇨띠가 크게 산 날 {int((d.y == BUY).sum())} (자산의 {med_b:.2f}배 이상), 크게 판 날 {int((d.y == SELL).sum())} ({-med_s:.2f}배 이상)")
for col in INTRA:
    q = pd.qcut(d[col], 4, duplicates="drop")
    g = d.groupby(q, observed=True).y.agg(n="size", 매수=lambda s: (s == BUY).mean() / rate[BUY], 매도=lambda s: (s == SELL).mean() / rate[SELL])
    print(f"  {INTRA_NAMES[col]}: " + " | ".join(f"{iv.left:.2g}~{iv.right:.2g} → 매수 {r.매수:.1f}배, 매도 {r.매도:.1f}배" for iv, r in g.iterrows()))

X, y = d[FEATS].to_numpy(dtype=float), d.y.to_numpy()
p_oof, model, desc, ll = tfs.fit_cv("1d", X, y, tfs.fold_ids(d.index, "H"))
print(f"모델: {desc}. 워뇨띠가 크게 산 날 구분 AUC {tfs.auc(y == BUY, p_oof[:, 2]):.2f}, 판 날 {tfs.auc(y == SELL, p_oof[:, 0]):.2f}  (일봉 차트만 썼을 때는 0.61 / 0.55)")
thr = {BUY: float(np.quantile(p_oof[:, 2], 1 - rate[BUY])), SELL: float(np.quantile(p_oof[:, 0], 1 - rate[SELL]))}
lc = np.log(d.px.to_numpy())
rows, plan = {}, {}
for label, mask, sgn, s_ in (("워뇨띠가 실제로 크게 산 날", y == BUY, 1, None), ("매수 신호 (학습 밖)", p_oof[:, 2] >= thr[BUY], 1, BUY),
                             ("워뇨띠가 실제로 크게 판 날", y == SELL, -1, None), ("매도 신호 (학습 밖)", p_oof[:, 0] >= thr[SELL], -1, SELL)):
    a = tfs.after(lc, mask, sgn)
    rows[label] = {"일수": int(mask.sum()), **{f"+{h}일": f"{a[h][0]:+.2f}% (t {a[h][1]:+.1f})" for h in tfs.HZ}}
    if s_ is not None and tfs.decide_side(a, sgn):
        plan[s_] = tfs.decide_side(a, sgn)
print("신호 뒤 가격 (그날 종가에 워뇨띠와 같은 방향으로 들어갔을 때, 시장 평균 흐름 제외, + = 유리):")
print(pd.DataFrame(rows).T.to_string())
size = {BUY: float(min(1.0, med_b)), SELL: float(min(1.0, -med_s))}
if not plan:
    print("→ 채택 없음: 5분봉 신호를 더해도 신호 뒤 가격이 한쪽으로 뚜렷하게 움직이지 않았다.")
    sys.exit()
print("→ 채택:", ", ".join(f"{'매수' if s_ == BUY else '매도'} 신호에 {'롱' if dr > 0 else '숏'} {h}일 보유 (크기 {size[s_]:.2f}배)" + ("" if dr == (1 if s_ == BUY else -1) else " ← 워뇨띠와 반대 방향")
                         for s_, (dr, h) in plan.items()))
for coin in ("BTC", "ETH"):
    c5 = pd.read_parquet(f"data/binance_{coin}_5m_202110_202610.parquet")
    f5 = make_features_5m(c5)
    pr = tb["model"].predict_proba(f5[FEATS_5M].to_numpy(dtype=float))
    intra = intraday(pd.DataFrame({"p_buy": pr[:, 2], "p_sell": pr[:, 0]}, index=f5.index)[f5.vol_regime.notna().to_numpy()])
    fd = make_features_k(tfs.test_candles("1d", coin)).join(intra)
    for pname, (s, e) in {"2022~2023 (확인용)": ("2022-01-01", "2023-12-31"), "2024~ (최종)": ("2024-01-01", None)}.items():
        r = bt(fd, model, thr, plan, size, s, e)
        print(f"  {coin} {pname}: 수익 {r['수익']:+.1%}, 샤프 {r['샤프']:.2f}, 최대낙폭 {r['최대낙폭']:.1%}, 거래 {r['거래수']}회, 승률 {r['승률']:.0%} | 단순 보유 {r['단순보유']:+.1%}")
