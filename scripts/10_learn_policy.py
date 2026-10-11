"""워뇨띠의 2020~2021 XBTUSD 매매에서 '어느 지점에서 사고 파는지'를 학습해 models/aoa_policy.joblib 로 저장한다.

매 1시간봉 마감 시각 t 마다:
  입력 = t 시점의 차트 상태(22개) + 워뇨띠의 포지션 상태(자산 대비 배수, 평가손익률)
  정답 = t ~ t+1시간 동안 워뇨띠가 한 일: 순매수(BUY) / 순매도(SELL) / 포지션 변화 없음(NONE)
숫자 기준은 전부 이 자료에서 계산한다. 사람이 정한 것은 맨 아래 '사람이 정한 것' 목록에 모아 출력한다.
"""
import json
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, ".")
from aoabot.features import make_features
from aoabot.policy import BUY, CLASSES, FEATS, MARKET_FEATS, MODEL_PATH, NONE, OWN_FEATS, SELL, BinnedLogit, FastGBM, save_bundle


def auc(label: np.ndarray, score: np.ndarray) -> float:
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    idx = np.searchsorted(CLASSES, y)
    return float(-np.log(np.clip(p[np.arange(len(y)), idx], 1e-12, 1)).mean())

pd.set_option("display.width", 220)
START, END = "2020-01-01", "2021-12-31 22:00"

# ---------- 1. 워뇨띠의 포지션·평균 진입가를 체결마다 재구성 ----------
tr = pd.read_parquet("data/trades_pos.parquet")
x = tr[tr.symbol == "XBTUSD"].reset_index(drop=True)
sq, px_f, pos_f = x.sq.to_numpy(), x.lastpx.to_numpy(), x.pos.to_numpy()
avg = np.zeros(len(x))
q, cost = 0.0, 0.0                      # 계약 수(부호), 그 포지션의 BTC 원가 Σ계약/가격
for i in range(len(x)):
    d = sq[i]
    if q == 0 or np.sign(d) == np.sign(q):
        cost += abs(d) / px_f[i]
        q += d
    else:
        if abs(d) < abs(q):
            cost *= (abs(q) - abs(d)) / abs(q)
            q += d
        else:                           # 전량 청산 또는 반전
            q += d
            cost = abs(q) / px_f[i] if q else 0.0
    avg[i] = abs(q) / cost if q else 0.0
x["avg"] = avg

c = pd.read_parquet("data/candles_XBTUSD_1h.parquet")
f = make_features(c)
fills = x.set_index("ts")[["pos", "avg"]]
h = fills.resample("1h", label="right", closed="right").last().reindex(f.index).ffill().fillna(0)
last_bar = pd.Series(x.ts.dt.floor("h").to_numpy(), index=x.ts).resample("1h", label="right", closed="right").last().reindex(f.index).ffill()
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
w["date"] = pd.to_datetime(w.date)
bal = (w.groupby("date").walletbalance.last() / 1e8).reindex(f.index.normalize()).ffill().to_numpy()

upnl_btc = np.where(h.pos != 0, h.pos * (1 / h.avg.replace(0, np.nan) - 1 / f.px), 0.0)   # 인버스 계약 평가손익(BTC)
equity_usd = (bal + np.nan_to_num(upnl_btc)) * f.px
f["own_lev"] = h.pos / equity_usd
f["own_upnl"] = np.where(h.pos != 0, np.sign(h.pos) * (f.px / h.avg.replace(0, np.nan) - 1), 0.0)
f["hrs_since_action"] = (f.index.to_series() - last_bar).dt.total_seconds() / 3600   # 워뇨띠가 마지막으로 매매한 봉의 시작 시각부터
dpos = h.pos.shift(-1) - h.pos
f["act"] = np.sign(dpos).fillna(0).astype(int)
f["dlev"] = dpos.abs() / equity_usd
d = f.loc[START:END].dropna(subset=MARKET_FEATS + ["own_lev"]).copy()

n = len(d)
rate = d.act.value_counts(normalize=True)
print(f"학습 구간 {d.index[0]} ~ {d.index[-1]}, {n}시간")
print(f"워뇨띠의 행동: 매수 {rate[BUY]:.1%}, 매도 {rate[SELL]:.1%}, 변화 없음 {rate[NONE]:.1%}  (하루 평균 {(1 - rate[NONE]) * 24:.1f}시간 매매)")

# ---------- 2. 워뇨띠가 사고 판 지점 요약 (사람이 읽는 용도) ----------
print("\n== 워뇨띠가 사고 판 지점: 조건별 매수·매도 확률 (전체 평균 대비 배수) ==")
def lift(col, bins, labels=None):
    b = pd.qcut(d[col], bins, duplicates="drop") if isinstance(bins, int) else pd.cut(d[col], bins, labels=labels)
    t = d.groupby(b, observed=True).act.agg(시간수="size", 매수=lambda s: (s == BUY).mean() / rate[BUY], 매도=lambda s: (s == SELL).mean() / rate[SELL])
    print(f"[{col}]"); print(t.round(2).to_string())
for col in ("vol_regime", "r_4h", "r_24h", "range_pos_24h", "own_upnl"):
    lift(col, 5)
lift("own_lev", [-10, -1.5, -0.5, -0.05, 0.05, 0.5, 10], ["숏 1.5배 초과", "숏 0.5~1.5배", "숏 0.5배 미만", "거의 없음", "롱 0.5배 미만", "롱 0.5배 초과"])

# ---------- 3. 모델 종류와 설정을 워뇨띠의 자료 안에서 교차 검증으로 고른다 ----------
X, y = d[FEATS].to_numpy(dtype=float), d.act.to_numpy()
fold = ((d.index.year - 2020) * 4 + (d.index.quarter - 1)).to_numpy()       # 분기 8개 블록
folds = np.unique(fold)
GBM_ITERS = [50, 100, 200, 400]


def mask_cols(cols):
    """cols 에 없는 입력은 상수로 만들어 효과를 없앤다."""
    Xc = X.copy()
    for j, k in enumerate(FEATS):
        if k not in cols:
            Xc[:, j] = 0.0
    return Xc


def make(kind, **kw):
    return BinnedLogit(**kw) if kind == "logit" else HistGradientBoostingClassifier(early_stopping=False, random_state=0, learning_rate=0.05, **kw)


def oof(kind, cols=FEATS, **kw):
    """학습에 쓰지 않은 분기를 예측한 확률."""
    Xc, p = mask_cols(cols), np.zeros((n, 3))
    for k in folds:
        p[fold == k] = make(kind, **kw).fit(Xc[fold != k], y[fold != k]).predict_proba(Xc[fold == k])
    return p


scores = []
for a_, b_, c_ in product([3, 5, 8, 12], [1.0, 10.0, 100.0, 1000.0], [False, True]):
    g = dict(n_bins=a_, l2=b_, interact=c_)
    scores.append((log_loss(y, oof("logit", **g)), "logit", g))
print("구간 로지스틱 후보 32개 검증 완료, 최저 로그손실", round(min(s_[0] for s_ in scores), 4), flush=True)
for leaf, msl in product([4, 8, 16], [50, 200]):               # 반복 횟수 4가지는 한 번 학습해 단계별 예측으로 평가
    stage = {it: np.zeros((n, 3)) for it in GBM_ITERS}
    for k in folds:
        m = make("gbm", max_leaf_nodes=leaf, min_samples_leaf=msl, max_iter=max(GBM_ITERS)).fit(X[fold != k], y[fold != k])
        for it, p in enumerate(m.staged_predict_proba(X[fold == k]), 1):
            if it in stage:
                stage[it][fold == k] = p
    for it in GBM_ITERS:
        scores.append((log_loss(y, stage[it]), "gbm", dict(max_leaf_nodes=leaf, min_samples_leaf=msl, max_iter=it)))
print("부스팅 트리 후보 24개 검증 완료, 최저 로그손실", round(min(s_[0] for s_ in scores if s_[1] == "gbm"), 4), flush=True)

best_ll, kind, best = min(scores, key=lambda s_: s_[0])
prior = np.tile(rate.reindex(CLASSES).to_numpy(), (n, 1))
names = {"logit": "구간 로지스틱", "gbm": "부스팅 트리"}
print(f"\n\n교차 검증으로 고른 모델: {names[kind]} {best}")
for k_ in names:
    print(f"  {names[k_]} 최저 로그손실 {min(s_[0] for s_ in scores if s_[1] == k_):.4f}")
print(f"  (참고) 워뇨띠의 매매 빈도만 아는 경우 {log_loss(y, prior):.4f} — 낮을수록 워뇨띠의 행동을 잘 맞힘")

print("\n== 워뇨띠의 행동을 얼마나 맞히나 (학습에 안 쓴 분기를 예측, AUC: 0.5 = 동전 던지기, 1.0 = 완벽) ==")
P = {"차트만": oof(kind, MARKET_FEATS, **best), "포지션 상태만": oof(kind, OWN_FEATS, **best), "차트 + 포지션": oof(kind, **best)}
acted = y != NONE
for name, p in P.items():
    print(f"  {name:10s} 매수 시점 {auc(y == BUY, p[:, 2]):.3f}  매도 시점 {auc(y == SELL, p[:, 0]):.3f}  "
          f"매매한 시간 중 방향(매수 vs 매도) {auc(y[acted] == BUY, p[acted, 2] - p[acted, 0]):.3f}")
p_oof = P["차트 + 포지션"]

# ---------- 4. 기준값: 전부 워뇨띠의 자료에서 ----------
thr_buy = float(np.quantile(p_oof[:, 2], 1 - rate[BUY]))      # 봇의 매수 빈도 = 워뇨띠의 매수 빈도가 되는 확률 기준선
thr_sell = float(np.quantile(p_oof[:, 0], 1 - rate[SELL]))
a = d[d.act != NONE]
kind_ = np.where((a.own_lev == 0) | (np.sign(a.own_lev) == a.act), "open", "reduce")
size = {(int(k[0]), k[1]): float(v) for k, v in a.groupby([a.act, kind_]).dlev.median().items()}
lev_min, lev_max = (float(v) for v in d.own_lev.quantile([0.01, 0.99]))
print(f"\n확률 기준선: 매수 {thr_buy:.3f}, 매도 {thr_sell:.3f}")
print("1회 매매 크기(자산 대비 배수, 중앙값):", {f"{'매수' if k[0] == BUY else '매도'}·{'포지션 늘릴 때' if k[1] == 'open' else '포지션 줄일 때'}": round(v, 3) for k, v in size.items()})
print(f"포지션 한계(워뇨띠의 1~99백분위): {lev_min:+.2f}배 ~ {lev_max:+.2f}배")

model = make(kind, **best).fit(X, y)
if kind == "gbm":                                    # 실행용으로 numpy 배열로 옮기고, 같은 답을 내는지 확인
    fast = FastGBM.from_sklearn(model)
    assert np.allclose(fast.predict_proba(X[:2000]), model.predict_proba(X[:2000]), atol=1e-9)
    model = fast

# 기준선 보정: 봇은 자기 포지션을 입력으로 쓰므로, 워뇨띠의 실제 차트에 봇을 돌렸을 때 매매 빈도가 워뇨띠와 같아지게 맞춘다
from aoabot import risk
from aoabot.policy_backtest import simulate
bundle = dict(model=model, thr_buy=thr_buy, thr_sell=thr_sell, thr_scale=1.0, size=size, lev_min=lev_min, lev_max=lev_max)
target = 1 - rate[NONE]


def run_his_chart(k):
    bundle["thr_scale"] = k
    return simulate(c, bundle, START, END, fee=0, slip=0, lim=risk.Limits(daily_loss=9.9, max_drawdown=9.9))


lo, hi = 0.25, 4.0                       # 기준선이 높을수록 매매가 줄어든다
for _ in range(12):
    mid = float(np.sqrt(lo * hi))
    lo, hi = (mid, hi) if (run_his_chart(mid).act != 0).mean() > target else (lo, mid)
thr_scale = float(np.sqrt(lo * hi))
sim = run_his_chart(thr_scale)
j = sim.join(d[["own_lev"]], how="inner")
fid = dict(봇_매매시간_일=float((sim.act != 0).mean() * 24), 워뇨띠의_매매시간_일=float(target * 24),
           포지션_상관계수=float(j.lev.corr(j.own_lev)), 봇_평균포지션=float(j.lev.mean()), 워뇨띠의_평균포지션=float(j.own_lev.mean()),
           방향_일치율=float((np.sign(j.lev) == np.sign(j.own_lev))[j.own_lev.abs() > 0.05].mean()))
print(f"\n기준선 보정 배수 {thr_scale:.3f} → 워뇨띠의 차트에서 봇 매매 {fid['봇_매매시간_일']:.2f}시간/일 (워뇨띠는 {fid['워뇨띠의_매매시간_일']:.2f})")
print(f"워뇨띠의 차트에서 봇이 워뇨띠를 얼마나 닮았나: 포지션 상관계수 {fid['포지션_상관계수']:+.2f}, 방향 일치율 {fid['방향_일치율']:.0%}, "
      f"평균 포지션 봇 {fid['봇_평균포지션']:+.2f}배 / 그 {fid['워뇨띠의_평균포지션']:+.2f}배")

# ---------- 5. 어떤 입력이 판단을 좌우하나, 그리고 그 입력의 어느 구간에서 사고 파나 ----------
base = log_loss(y, model.predict_proba(X))
rng = np.random.default_rng(0)
imp = {}
for j, k in enumerate(FEATS):
    Xp = X.copy()
    Xp[:, j] = rng.permutation(Xp[:, j])
    imp[k] = log_loss(y, model.predict_proba(Xp)) - base
imp = pd.Series(imp).sort_values(ascending=False)
print("\n== 판단을 좌우하는 입력 (그 입력을 뒤섞었을 때 예측이 나빠지는 정도) ==")
print(imp.head(12).round(4).to_string())
print("\n== 상위 입력의 구간별 실제 매수·매도 빈도 (1.0 = 평균, 2.0 = 평소의 2배) ==")
tables = {}
for k in imp.index[:6]:
    b = pd.qcut(d[k], 5, duplicates="drop")
    t = d.groupby(b, observed=True).act.agg(시간수="size", 매수배수=lambda s_: (s_ == BUY).mean() / rate[BUY], 매도배수=lambda s_: (s_ == SELL).mean() / rate[SELL])
    tables[k] = {str(i): dict(매수배수=round(float(r.매수배수), 2), 매도배수=round(float(r.매도배수), 2)) for i, r in t.iterrows()}
    print(f"[{k}]"); print(t.round(2).to_string())

human = {
    "입력 후보 24개": "차트 지표 22개와 포지션 상태 2개의 종류. 각 입력을 쓸지, 어느 값에서 가를지는 학습으로 정해진다",
    "모델 후보 2종": "구간 로지스틱, 부스팅 트리(학습률 0.05) — 둘 중 선택은 교차 검증",
    "설정 후보": "로지스틱: 구간 수 3/5/8/12, 규제 1/10/100/1000, 포지션 방향별 효과 유무. 트리: 잎 4/8/16, 반복 50/100/200/400, 잎 최소 표본 50/200 — 선택은 교차 검증",
    "검증 방식": "분기 단위 8블록 교차 검증, 기준은 로그손실",
    "학습 구간과 봉": "2020-01 ~ 2021-12 (요청 기간 중 자료가 있는 구간), 1시간봉, XBTUSD",
    "크기 통계량": "1회 매매 크기는 중앙값, 포지션 한계는 1~99백분위",
    "행동 기준선 방식": "워뇨띠의 2020~2021 차트에 봇을 돌렸을 때 매매 빈도가 워뇨띠의 빈도와 같아지도록 확률 기준선을 맞춤",
    "뺀 입력": "'마지막 매매 후 경과 시간' — 넣으면 봇이 한 번 쉰 뒤 다시 매매하지 않아서 제외",
}
meta = dict(trained=f"{d.index[0]} ~ {d.index[-1]}", hours=n, rate={int(k): float(v) for k, v in rate.items()},
            model_kind=names[kind], hyper=best, cv_logloss=best_ll, prior_logloss=log_loss(y, prior),
            cv_by_kind={names[k_]: min(s_[0] for s_ in scores if s_[1] == k_) for k_ in names},
            auc={name: dict(buy=auc(y == BUY, p[:, 2]), sell=auc(y == SELL, p[:, 0]),
                            direction=auc(y[acted] == BUY, p[acted, 2] - p[acted, 0])) for name, p in P.items()},
            top_inputs=imp.head(10).round(5).to_dict(), where=tables, closed_loop=fid, human=human)
save_bundle(dict(model=model, thr_buy=thr_buy, thr_sell=thr_sell, thr_scale=thr_scale, size=size, lev_min=lev_min, lev_max=lev_max, meta=meta))
d[["px", "own_lev", "act"]].to_parquet("data/his_actions_2020_2021.parquet")
print(f"\n저장: {MODEL_PATH}")
