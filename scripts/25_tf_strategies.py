"""봉마다(1시간·일·주·월) 워뇨띠의 매매에서 전략을 만들고 검증한다. 네 봉 모두 같은 절차를 쓴다.

  1. 그 봉 단위로 워뇨띠의 순매수·순매도를 집계한다 (XBTUSD). 크게 산 봉 / 크게 판 봉 = 각 방향 중앙값 이상.
  2. 그 봉의 차트 상태(aoabot.features_k)로 '워뇨띠가 이 봉에서 크게 샀나/팔았나'를 학습한다. 설정은 교차 검증으로 고른다.
  3. 신호 기준선 = 워뇨띠가 크게 산(판) 봉의 빈도. 학습에 안 쓴 구간의 신호 뒤 가격을 1·2·3·5·10봉 뒤까지 본다.
  4. 채택 규칙(미리 정함): 신호가 30개 이상이고, 신호 뒤 수익의 t값이 2 이상이면 워뇨띠와 같은 방향으로,
     -2 이하이면 반대 방향으로 쓴다(워뇨띠의 매매가 그 봉에서는 거꾸로 맞았다는 뜻). 보유 기간 = |t|가 가장 큰 봉 수.
  5. 채택된 방향으로 바이낸스 BTC·ETH 2022~2023(확인용)과 2024~(최종)에 돌린다.
크기는 워뇨띠가 그 봉에서 크게 살(팔) 때의 순매수 중앙값(자산 대비, 상한 1배). 수수료 0.05% + 슬리피지 0.02%.
"""
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, ".")
from aoabot.features_k import K_FEATS, K_NAMES, make_features_k, to_higher
from aoabot.policy import BUY, CLASSES, NONE, ROOT, SELL, BinnedLogit, FastGBM, save_bundle

pd.set_option("display.width", 250)
HZ = (1, 2, 3, 5, 10)
FEE, SLIP = 0.0005, 0.0002
TF = {   # 이름: (학습 시작, 교차 검증 블록, 1년의 봉 수, 구간 수)
    "1h": dict(name="1시간봉", start="2020-01-01", fold="Q", ann=8760, bins=5),
    "1d": dict(name="일봉", start="2018-04-01", fold="H", ann=365, bins=5),
    "1w": dict(name="주봉", start="2018-04-01", fold="Y", ann=52, bins=3),
    "1M": dict(name="월봉", start="2018-04-01", fold="Y", ann=12, bins=3),
}
END = "2021-12-31 23:59"


def idx(df):
    return df.set_index(df.timestamp.dt.tz_localize(None)).sort_index()[["open", "high", "low", "close", "volume"]]


def his_candles(tf):
    if tf == "1h":
        return idx(pd.read_parquet("data/candles_XBTUSD_1h.parquet"))
    d = idx(pd.read_parquet("data/candles_XBTUSD_1d_2015_2022.parquet"))
    return d if tf == "1d" else to_higher(d, "W" if tf == "1w" else "M")


def test_candles(tf, coin):
    if tf == "1h":
        return idx(pd.read_parquet(f"data/binance_{coin}_1h.parquet"))
    d = idx(pd.read_parquet(f"data/binance_{coin}_1d.parquet"))
    return d if tf == "1d" else to_higher(d, "W" if tf == "1w" else "M")


def auc(label, score):
    if label.sum() in (0, len(label)):
        return np.nan
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))


def log_loss(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), np.searchsorted(CLASSES, y)], 1e-12, 1)).mean())


def fold_ids(index, kind):
    if kind == "Q":
        return (index.year * 4 + index.quarter).to_numpy()
    if kind == "H":
        return (index.year * 2 + (index.month > 6)).to_numpy()
    return index.year.to_numpy()


def candidates(tf, n):
    out = []
    if tf in ("1h", "1d"):
        for msl in ((200, 1000) if tf == "1h" else (30, 80)):
            out.append(("gbm", dict(max_leaf_nodes=4, min_samples_leaf=msl), (50, 100, 200, 300) if tf == "1h" else (30, 60, 120)))
    if tf != "1h":
        for nb, l2 in product((2, 3) if tf in ("1w", "1M") else (3, 5), (10.0, 100.0)):
            out.append(("logit", dict(n_bins=nb, l2=l2, interact=False), None))
    return out


def fit_cv(tf, X, y, fold):
    """(가장 좋은 설정의 학습 밖 예측 확률, 전체 자료로 학습한 모델, 설명)"""
    n, best = len(y), None
    for kind, kw, iters in candidates(tf, n):
        if kind == "gbm":
            st = {it: np.zeros((n, 3)) for it in iters}
            for k in np.unique(fold):
                m = HistGradientBoostingClassifier(early_stopping=False, random_state=0, learning_rate=0.05, max_iter=max(iters), **kw).fit(X[fold != k], y[fold != k])
                cls = list(m.classes_)
                for it, p in enumerate(m.staged_predict_proba(X[fold == k]), 1):
                    if it in st:
                        full = np.full((p.shape[0], 3), 1e-9)
                        for j, c_ in enumerate(cls):
                            full[:, CLASSES.index(c_)] = p[:, j]
                        st[it][fold == k] = full / full.sum(axis=1, keepdims=True)
            for it in iters:
                ll = log_loss(y, st[it])
                if best is None or ll < best[0]:
                    best = (ll, kind, {**kw, "max_iter": it}, st[it])
        else:
            p = np.zeros((n, 3))
            for k in np.unique(fold):
                p[fold == k] = BinnedLogit(**kw).fit(X[fold != k], y[fold != k]).predict_proba(X[fold == k])
            ll = log_loss(y, p)
            if best is None or ll < best[0]:
                best = (ll, kind, kw, p)
    ll, kind, kw, p = best
    if kind == "gbm":
        m = HistGradientBoostingClassifier(early_stopping=False, random_state=0, learning_rate=0.05, **kw).fit(X, y)
        model = FastGBM.from_sklearn(m)
    else:
        model = BinnedLogit(**kw).fit(X, y)
    return p, model, f"{'부스팅 트리' if kind == 'gbm' else '구간 로지스틱'} {kw}", ll


def after(lc, mask, sgn):
    """신호 봉 종가에 sgn 방향으로 들어갔을 때 h봉 뒤 수익률(%, 시장 평균 흐름 제외), t값, 표본 수."""
    out = {}
    n = len(lc)
    ii = np.flatnonzero(mask)
    for h in HZ:
        j = ii[ii + h < n]
        if len(j) < 3:
            out[h] = (np.nan, np.nan, len(j))
            continue
        v = sgn * (lc[j + h] - lc[j] - np.mean(lc[h:] - lc[:-h]))
        out[h] = (float(v.mean() * 100), float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std() > 0 else np.nan, len(j))
    return out


def decide_side(a, sgn):
    """채택 규칙. (방향, 보유 봉 수) 또는 None."""
    ok = [h for h in HZ if a[h][2] >= 30 and not np.isnan(a[h][1])]
    if not ok:
        return None
    h = max(ok, key=lambda h_: abs(a[h_][1]))
    if a[h][1] >= 2:
        return sgn, h
    if a[h][1] <= -2:
        return -sgn, h
    return None


def backtest(k, model, thr, plan, size, start, end, ann, min_prior=0, window=4):
    """plan: {BUY 또는 SELL 신호: (방향 +1 롱/-1 숏, 보유 봉 수)}. min_prior>0 이면 직전 window 봉 안에 같은 신호가 그만큼 있어야 진입."""
    f = make_features_k(k)
    f = f.loc[start:end]
    ok = f[K_FEATS].notna().all(axis=1).to_numpy()
    p = model.predict_proba(f[K_FEATS].fillna(0).to_numpy(dtype=float))
    sig = {BUY: ok & (p[:, 2] >= thr[BUY]), SELL: ok & (p[:, 0] >= thr[SELL])}
    if min_prior:
        for s_ in sig:
            prev = pd.Series(sig[s_].astype(float)).rolling(window + 1, min_periods=1).sum().to_numpy() - sig[s_]
            sig[s_] = sig[s_] & (prev >= min_prior)
    px = f.px.to_numpy()
    cash, qty, entry, ent_i, side, hold = 1.0, 0.0, 0.0, -1, 0, 0
    eq, trades = np.zeros(len(f)), []
    for i in range(len(f)):
        if side and i - ent_i >= hold:
            x = px[i] * (1 - side * SLIP)
            cash += qty * (x - entry) * side - qty * x * FEE
            trades.append(side * (x / entry - 1))
            side, qty = 0, 0.0
        if not side:
            s_ = BUY if (BUY in plan and sig[BUY][i]) else SELL if (SELL in plan and sig[SELL][i]) else 0
            if s_:
                side, hold = plan[s_]
                entry = px[i] * (1 + side * SLIP)
                qty = size[s_] * cash / entry
                cash -= qty * entry * FEE
                ent_i = i
        eq[i] = cash + (qty * (px[i] - entry) * side if side else 0.0)
    e = pd.Series(eq, index=f.index)
    r = e.pct_change().dropna()
    tr = np.array(trades)
    return dict(수익=e.iloc[-1] - 1, 샤프=float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0, 최대낙폭=float((e / e.cummax() - 1).min()),
                거래수=len(tr), 승률=float((tr > 0).mean()) if len(tr) else np.nan, 단순보유=px[-1] / px[0] - 1)


def run_tests(tf, cfg, model, thr, plan, size, summary, label, **kw):
    for coin in ("BTC", "ETH"):
        kc = test_candles(tf, coin)
        for pname, (s_, e_) in {"2022~2023 (확인용)": ("2022-01-01", "2023-12-31"), "2024~ (최종)": ("2024-01-01", None)}.items():
            r = backtest(kc, model, thr, plan, size, s_, e_, cfg["ann"], **kw)
            print(f"    {coin} {pname}: 수익 {r['수익']:+.1%}, 샤프 {r['샤프']:.2f}, 최대낙폭 {r['최대낙폭']:.1%}, 거래 {r['거래수']}회, 승률 {r['승률']:.0%} | 단순 보유 {r['단순보유']:+.1%}")
            summary.append(dict(봉=cfg["name"], 전략=label, 코인=coin, 구간=pname, **r))


def main():
    tr = pd.read_parquet("data/trades_pos.parquet")
    pos = tr[tr.symbol == "XBTUSD"].set_index("ts").pos.groupby(level=0).last()
    w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
    w["date"] = pd.to_datetime(w.date)
    bal = w.groupby("date").walletbalance.last() / 1e8
    summary = []
    for tf, cfg in TF.items():
        k = his_candles(tf)
        f = make_features_k(k)
        p_ = pos.reindex(pos.index.union(k.index)).ffill().reindex(k.index).fillna(0)
        eq = bal.reindex(bal.index.union(k.index.normalize())).ffill().reindex(k.index.normalize()).to_numpy() * k.close.to_numpy()
        f["flow"] = (p_.diff() / eq).to_numpy()
        d = f.loc[cfg["start"]:END].dropna(subset=K_FEATS + ["flow"]).copy()
        med_b, med_s = d.flow[d.flow > 0.02].median(), d.flow[d.flow < -0.02].median()
        d["y"] = np.select([d.flow >= med_b, d.flow <= med_s], [BUY, SELL], NONE)
        n = len(d)
        rate = {c: float((d.y == c).mean()) for c in CLASSES}
        print("\n" + "=" * 112)
        print(f"{cfg['name']}: 학습 {d.index[0]:%Y-%m} ~ {d.index[-1]:%Y-%m}, {n}봉. 워뇨띠가 크게 산 봉 {int((d.y == BUY).sum())} (자산의 {med_b:.2f}배 이상), "
              f"크게 판 봉 {int((d.y == SELL).sum())} ({-med_s:.2f}배 이상)")

        # 워뇨띠가 이 봉에서 사고 판 지점
        lifts = []
        for col in K_FEATS:
            q = pd.qcut(d[col], cfg["bins"], duplicates="drop")
            g = d.groupby(q, observed=True).y.agg(n="size", buy=lambda s: (s == BUY).mean() / rate[BUY], sell=lambda s: (s == SELL).mean() / rate[SELL])
            for iv, r in g.iterrows():
                lifts.append(dict(항목=K_NAMES[col], 구간=f"{iv.left:.3g} ~ {iv.right:.3g}", 봉수=int(r.n), 매수배수=r.buy, 매도배수=r.sell))
        L = pd.DataFrame(lifts)
        print("  크게 산 지점 상위:", " | ".join(f"{r.항목} {r.구간} → {r.매수배수:.1f}배" for r in L.nlargest(4, "매수배수").itertuples()))
        print("  크게 판 지점 상위:", " | ".join(f"{r.항목} {r.구간} → {r.매도배수:.1f}배" for r in L.nlargest(4, "매도배수").itertuples()))

        X, y = d[K_FEATS].to_numpy(dtype=float), d.y.to_numpy()
        p_oof, model, desc, ll = fit_cv(tf, X, y, fold_ids(d.index, cfg["fold"]))
        prior = np.tile([rate[c] for c in CLASSES], (n, 1))
        print(f"  모델: {desc}. 로그손실 {ll:.4f} (빈도만 알 때 {log_loss(y, prior):.4f}). 워뇨띠가 크게 산 봉 구분 AUC {auc(y == BUY, p_oof[:, 2]):.2f}, 판 봉 {auc(y == SELL, p_oof[:, 0]):.2f}")
        thr = {BUY: float(np.quantile(p_oof[:, 2], 1 - rate[BUY])), SELL: float(np.quantile(p_oof[:, 0], 1 - rate[SELL]))}
        lc = np.log(d.px.to_numpy())
        rows, plan = {}, {}
        for label, mask, sgn, s_ in (("워뇨띠가 실제로 크게 산 봉", y == BUY, 1, None), ("매수 신호 (학습 밖)", p_oof[:, 2] >= thr[BUY], 1, BUY),
                                     ("워뇨띠가 실제로 크게 판 봉", y == SELL, -1, None), ("매도 신호 (학습 밖)", p_oof[:, 0] >= thr[SELL], -1, SELL)):
            a = after(lc, mask, sgn)
            rows[label] = {"봉수": int(mask.sum()), **{f"+{h}봉": f"{a[h][0]:+.2f}% (t {a[h][1]:+.1f})" for h in HZ}}
            if s_ is not None:
                dec = decide_side(a, sgn)
                if dec:
                    plan[s_] = dec
        print("  신호 뒤 가격 (그 봉 종가에 워뇨띠와 같은 방향으로 들어갔을 때, 시장 평균 흐름 제외, + = 유리):")
        print("  " + pd.DataFrame(rows).T.to_string().replace("\n", "\n  "))
        size = {BUY: float(min(1.0, med_b)), SELL: float(min(1.0, -med_s))}
        names = {BUY: "매수 신호", SELL: "매도 신호"}
        if not plan:
            print(f"  → 채택 없음: {cfg['name']}에서는 워뇨띠의 매매 지점 뒤에 가격이 한쪽으로 뚜렷하게 움직이지 않았다 (신호 30개 이상, |t| 2 이상인 경우가 없음).")
            summary.append(dict(봉=cfg["name"], 전략="채택 없음"))
        else:
            desc_plan = ", ".join(f"{names[s_]}에 {'롱' if dr > 0 else '숏'} {h}봉 보유 (크기 {size[s_]:.2f}배)" + ("" if dr == (1 if s_ == BUY else -1) else " ← 워뇨띠와 반대 방향")
                                  for s_, (dr, h) in plan.items())
            print("  → 채택: " + desc_plan)
            run_tests(tf, cfg, model, thr, plan, size, summary, desc_plan)
            save_bundle(dict(model=model, thr_buy=thr[BUY], thr_sell=thr[SELL], feats=K_FEATS, size={(s_, "open"): size[s_] for s_ in plan},
                             meta=dict(tf=tf, plan={int(s_): [int(dr), int(h)] for s_, (dr, h) in plan.items()}, model=desc, rate=rate)),
                        ROOT / "models" / f"aoa_tf_{tf}.json")

        # 5분봉에서 효과가 있었던 두 보완(강한 신호만, 연속 신호의 첫 번째는 건너뛰기)을 같은 방식으로 적용 — 신호가 충분한 봉만
        if (p_oof[:, 2] >= thr[BUY]).sum() >= 300:
            pb = p_oof[:, 2]
            thr2 = {BUY: float(np.median(pb[pb >= thr[BUY]])), SELL: thr[SELL]}
            strong = pb >= thr2[BUY]
            prev = pd.Series(strong.astype(float)).rolling(5, min_periods=1).sum().to_numpy() - strong
            late = strong & (prev >= 1)
            rows2 = {}
            for label, mask in (("강한 매수 신호 (상위 절반)", strong), ("강한 신호 중 첫 번째 (직전 4봉에 없음)", strong & (prev == 0)), ("강한 신호 중 2번째 이후", late)):
                a = after(lc, mask, 1)
                rows2[label] = {"봉수": int(mask.sum()), **{f"+{h}봉": f"{a[h][0]:+.2f}% (t {a[h][1]:+.1f})" for h in HZ}}
            print("  보완 (5분봉과 같은 방식): 강한 신호만, 연속 신호의 첫 번째는 건너뛰기")
            print("  " + pd.DataFrame(rows2).T.to_string().replace("\n", "\n  "))
            dec = decide_side(after(lc, late, 1), 1)
            if dec:
                print(f"  → 보완판 채택: 강한 매수 신호가 이어진 뒤 {'롱' if dec[0] > 0 else '숏'} {dec[1]}봉 보유")
                run_tests(tf, cfg, model, thr2, {BUY: dec}, size, summary, f"보완판: 강한 신호 2번째 이후 {dec[1]}봉", min_prior=1, window=4)
            else:
                print("  → 보완판 채택 없음")
    pd.DataFrame(summary).to_csv("data/tf_strategies.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
