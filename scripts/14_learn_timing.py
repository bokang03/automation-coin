"""워뇨띠의 매수·매도 타이밍을 5분봉 차트에서 학습한다 (XBTUSD 2020-01 ~ 2021-12) → models/aoa_timing.json

정답: 그 5분봉 안에서 워뇨띠가 순매수했는가(BUY) / 순매도했는가(SELL) / 매매가 없었는가(NONE).
입력: 그 봉이 끝난 시점의 차트 상태(aoabot.features5). 봉이 끝나면 알 수 있는 값만 쓴다.
학습 뒤에는 '봇이 고른 지점 이후의 가격 움직임'이 워뇨띠의 실제 지점 이후와 같은지 본다.
"""
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, ".")
from aoabot.features5 import FEATS_5M, make_features_5m
from aoabot.features_tf import TF_FEATS, make_tf_features
from aoabot.policy import BUY, CLASSES, NONE, SELL, FastGBM, save_bundle
from aoabot.policy import ROOT

pd.set_option("display.width", 250)
START, END = "2020-01-01", "2021-12-31 23:55"
ITERS = [50, 100, 200, 300]
HZ = {"+15분": 3, "+1시간": 12, "+2시간": 24, "+4시간": 48, "+8시간": 96, "+24시간": 288}


def auc(label, score):
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))


def log_loss(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), np.searchsorted(CLASSES, y)], 1e-12, 1)).mean())


def after(lc: np.ndarray, idx: np.ndarray, sgn: int) -> dict:
    """idx 봉의 종가에 sgn 방향으로 들어갔을 때 이후 수익률(bp). 같은 기간의 평균 흐름(시장 전체 상승·하락분)을 뺀 값."""
    out = {}
    for name, k in HZ.items():
        j = idx[idx + k < len(lc)]
        drift = np.nanmean(lc[k:] - lc[:-k])
        out[name] = float(sgn * (np.mean(lc[j + k] - lc[j]) - drift) * 1e4)
    return out


if __name__ == "__main__":
    # --tf        : 1시간·일·주·월봉 안의 위치(scripts/20_tf_analysis.py)를 입력에 더한다
    # --tf=1h,1d  : 그중 고른 봉만 더한다
    tf_arg = next((a for a in sys.argv[1:] if a.startswith("--tf")), None)
    USE_TF = tf_arg is not None
    tfs = tf_arg.split("=")[1].split(",") if USE_TF and "=" in tf_arg else None
    FEATS = FEATS_5M + ([k for k in TF_FEATS if tfs is None or k.split("_")[0] in tfs] if USE_TF else [])
    TAG = "" if not USE_TF else "_tf" if tfs is None else "_tf_" + "_".join(tfs)
    c = pd.read_parquet("data/candles_XBTUSD_5m.parquet")
    c = c[(c.timestamp >= "2019-10-01") & (c.timestamp < "2022-01-01")]
    f = make_features_5m(c)
    if USE_TF:
        f = f.join(make_tf_features(c))
    ev = pd.read_parquet("data/trade_events_5m_2020_2021.parquet")          # scripts/12_timing_events.py 결과
    net = (ev.sgn * ev.usd).groupby(ev.bar).sum()
    sz = (ev.sgn * ev["size"]).groupby(ev.bar).sum()
    f["act"] = np.sign(net.reindex(f.index).fillna(0)).astype(int)
    f["size"] = sz.reindex(f.index).fillna(0).abs()
    d = f.loc[START:END].dropna(subset=["vol_regime"]).copy()
    n = len(d)
    rate = d.act.value_counts(normalize=True)
    print(f"학습 구간 {d.index[0]} ~ {d.index[-1]}, 5분봉 {n:,}개. 워뇨띠가 산 봉 {rate[BUY]:.2%}, 판 봉 {rate[SELL]:.2%} "
          f"(하루 매수 {rate[BUY] * 288:.1f}봉, 매도 {rate[SELL] * 288:.1f}봉)", flush=True)

    X, y = d[FEATS].to_numpy(dtype=float), d.act.to_numpy()
    fold = ((d.index.year - 2020) * 4 + (d.index.quarter - 1)).to_numpy()
    scores, stages = [], {}
    for leaf, msl in product([4, 8, 16], [200, 1000]):
        st = {it: np.zeros((n, 3)) for it in ITERS}
        for k in np.unique(fold):
            m = HistGradientBoostingClassifier(early_stopping=False, random_state=0, learning_rate=0.05, max_leaf_nodes=leaf,
                                               min_samples_leaf=msl, max_iter=max(ITERS)).fit(X[fold != k], y[fold != k])
            for it, p in enumerate(m.staged_predict_proba(X[fold == k]), 1):
                if it in st:
                    st[it][fold == k] = p
        for it in ITERS:
            g = dict(max_leaf_nodes=leaf, min_samples_leaf=msl, max_iter=it)
            scores.append((log_loss(y, st[it]), g))
            stages[(leaf, msl, it)] = st[it]
        print(f"  검증 완료: 잎 {leaf}, 잎 최소 표본 {msl} → 로그손실 {min(log_loss(y, st[it]) for it in ITERS):.5f}", flush=True)
    best_ll, best = min(scores, key=lambda s: s[0])
    p_oof = stages[(best["max_leaf_nodes"], best["min_samples_leaf"], best["max_iter"])]
    prior = np.tile(rate.reindex(CLASSES).to_numpy(), (n, 1))
    print(f"\n교차 검증으로 고른 설정: {best} (로그손실 {best_ll:.5f}, 빈도만 아는 경우 {log_loss(y, prior):.5f})")
    print(f"워뇨띠가 산 봉을 맞히는 정도 AUC {auc(y == BUY, p_oof[:, 2]):.3f}, 판 봉 {auc(y == SELL, p_oof[:, 0]):.3f} (학습에 안 쓴 분기 예측)")

    thr_buy_all = float(np.quantile(p_oof[:, 2], 1 - rate[BUY]))    # 1차 기준선: 봇의 매수 신호 빈도 = 워뇨띠가 산 봉 전체의 빈도
    # 최종 기준선: 워뇨띠가 중앙값 이상 크기로 산 봉의 빈도에 맞춘다. 작게 산 곳은 이후 반등이 없었다 (scripts/17_gap_analysis.py)
    big = ((d.act == BUY) & (d["size"] >= d["size"][d.act == BUY].median())).to_numpy()
    thr_buy = float(np.quantile(p_oof[:, 2], 1 - big.mean()))
    thr_sell = float(np.quantile(p_oof[:, 0], 1 - rate[SELL]))
    sig_buy, sig_sell = p_oof[:, 2] >= thr_buy, p_oof[:, 0] >= thr_sell
    print(f"확률 기준선: 매수 {thr_buy:.4f} (워뇨띠의 매수 전체 빈도 기준이면 {thr_buy_all:.4f}), 매도 {thr_sell:.4f}")

    def near(sig, act):   # 신호 봉 앞뒤 30분 안에 워뇨띠가 실제로 그 방향으로 매매했나
        a = pd.Series((y == act).astype(float), index=d.index).rolling(13, center=True, min_periods=1).max().to_numpy() > 0
        return float(a[sig].mean())
    print(f"봇 매수 신호 중 워뇨띠도 ±30분 안에 실제로 산 비율 {near(sig_buy, BUY):.0%} (아무 봉이나 찍으면 {near(np.ones(n, bool), BUY):.0%}), "
          f"매도 신호 {near(sig_sell, SELL):.0%} (아무 봉이나 {near(np.ones(n, bool), SELL):.0%})")

    lc = np.log(d.px.to_numpy())
    idx = np.arange(n)
    rows = {"워뇨띠가 실제로 산 봉": after(lc, idx[y == BUY], 1), "봇 매수 신호 (학습에 안 쓴 분기)": after(lc, idx[sig_buy], 1),
            "워뇨띠가 실제로 판 봉": after(lc, idx[y == SELL], -1), "봇 매도 신호 (학습에 안 쓴 분기)": after(lc, idx[sig_sell], -1)}
    print("\n== 그 봉의 종가에 따라 들어갔을 때 이후 수익률(bp, 시장 평균 흐름 제외, + = 유리) : 워뇨띠의 차트 2020~2021 ==")
    print(pd.DataFrame(rows).T.round(1).to_string())

    model = HistGradientBoostingClassifier(early_stopping=False, random_state=0, learning_rate=0.05, **best).fit(X, y)
    fast = FastGBM.from_sklearn(model)
    assert np.allclose(fast.predict_proba(X[:3000]), model.predict_proba(X[:3000]), atol=1e-9)
    base = log_loss(y, model.predict_proba(X))
    rng = np.random.default_rng(0)
    imp = {}
    for j, k in enumerate(FEATS):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[k] = log_loss(y, model.predict_proba(Xp)) - base
    imp = pd.Series(imp).sort_values(ascending=False)
    print("\n== 타이밍 판단을 좌우하는 입력 상위 10개 ==")
    print(imp.head(10).round(5).to_string())

    a = d[d.act != NONE]
    size = {BUY: float(a["size"][a.act == BUY].median()), SELL: float(a["size"][a.act == SELL].median())}
    print(f"\n한 봉에서의 매매 크기 중앙값(자산 대비): 매수 {size[BUY]:.3f}배, 매도 {size[SELL]:.3f}배")
    meta = dict(trained=f"{d.index[0]} ~ {d.index[-1]}", bars=n, rate={int(k): float(v) for k, v in rate.items()}, hyper=best,
                auc_buy=auc(y == BUY, p_oof[:, 2]), auc_sell=auc(y == SELL, p_oof[:, 0]), after={k: v for k, v in rows.items()},
                top_inputs=imp.head(10).round(6).to_dict(), big_buy_median_size=float(d["size"][d.act == BUY].median()),
                big_buy_size=float(d["size"][big].median()))
    save_bundle(dict(model=fast, thr_buy=thr_buy, thr_buy_all=thr_buy_all, thr_sell=thr_sell, thr_scale=1.0, feats=FEATS,
                     size={(BUY, "open"): size[BUY], (SELL, "open"): size[SELL]}, meta=meta), ROOT / "models" / f"aoa_timing{TAG}.json")
    pd.DataFrame({"p_sell": p_oof[:, 0], "p_buy": p_oof[:, 2], "act": y, "px": d.px.to_numpy()}, index=d.index).to_parquet(f"data/timing_oof{TAG}_2020_2021.parquet")
    print("\n저장: models/aoa_timing.json")
