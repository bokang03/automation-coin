"""워뇨띠의 2020~2021 실제 매매에서 학습한 정책.

매 1시간봉 마감 때 (차트 상태, 내 포지션 상태)를 넣으면 다음 1시간 동안 워뇨띠가 했을 행동
(매수 / 매도 / 가만히)을 고른다. 진입·추가·축소·청산·반전을 따로 규칙으로 두지 않고,
전부 '지금 사느냐 파느냐' 하나로 다룬다. 기준값은 scripts/10_learn_policy.py가 워뇨띠의 자료에서 계산해
models/aoa_policy.json 에 저장한다.

포지션은 '자산 대비 배수'(lev)로 다룬다. +1.0 = 자산만큼 롱, -1.0 = 자산만큼 숏.
백테스트(aoabot.policy_backtest)와 실행 루프가 이 모듈의 함수를 똑같이 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "aoa_policy.json"

# 차트 상태: aoabot.features.make_features 가 만드는 값 중 가격 수준(px)만 뺀 전부
MARKET_FEATS = ["r_1h", "r_4h", "r_12h", "r_24h", "r_72h", "r_168h", "atr24", "vol_regime",
                "range_pos_24h", "dd_from_high_24h", "range_pos_168h", "dd_from_high_168h",
                "vol_ratio", "vol_ratio_4h", "ma20_gap", "ma50_gap", "ma200_gap", "ma_gap_1d_50",
                "body", "lower_wick", "upper_wick", "hour"]
# 내 포지션 상태. '마지막 매매 후 경과 시간'은 넣지 않는다: 워뇨띠는 몰아서 매매하는 습관이 있어 이 값을 넣으면
# 봇이 한 번 쉬기 시작한 뒤 다시는 매매하지 않는다 (2020~2021 차트에 돌려 확인).
OWN_FEATS = ["own_lev", "own_upnl"]
FEATS = MARKET_FEATS + OWN_FEATS
SELL, NONE, BUY = -1, 0, 1
CLASSES = [SELL, NONE, BUY]


class BinnedLogit:
    """분위수 구간 로지스틱 회귀. 각 입력을 학습 자료의 분위수로 구간을 나누고, 구간마다 매수/매도/가만히의
    로그 가능도를 더한다. interact=True 면 '포지션 방향(숏/중립/롱 3분위) × 차트 구간' 효과도 따로 둔다."""

    def __init__(self, n_bins: int = 5, l2: float = 10.0, interact: bool = False):
        self.n_bins, self.l2, self.interact = n_bins, l2, interact

    def _edges(self, X: np.ndarray) -> None:
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.nanquantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.offsets = np.r_[0, np.cumsum([len(e) + 1 for e in self.edges])]
        self.group_edges = np.nanquantile(X[:, FEATS.index("own_lev")], [1 / 3, 2 / 3]) if self.interact else np.zeros(2)

    def _design(self, X: np.ndarray) -> np.ndarray:
        n, total = len(X), self.offsets[-1]
        n_mkt = self.offsets[len(MARKET_FEATS)] if self.interact else 0
        D = np.zeros((n, total + 3 * n_mkt), dtype=np.float32)
        rows = np.arange(n)
        grp = np.searchsorted(self.group_edges, X[:, FEATS.index("own_lev")], side="right") if self.interact else None
        for j, e in enumerate(self.edges):
            col = X[:, j]
            b = np.where(np.isnan(col), len(e), np.searchsorted(e, col, side="right"))   # 값 없음 = 가장 큰 구간
            D[rows, self.offsets[j] + b] = 1
            if self.interact and j < len(MARKET_FEATS):
                D[rows, total + grp * n_mkt + self.offsets[j] + b] = 1
        return D

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinnedLogit":
        self._edges(X)
        D = self._design(X)
        n, d = D.shape
        Y = np.stack([(y == c) for c in CLASSES], axis=1).astype(np.float32)

        def loss(theta):
            W, b = theta[:d * 3].reshape(d, 3).astype(np.float32), theta[d * 3:].astype(np.float32)
            z = D @ W + b
            z -= z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
            P = np.exp(z - lse)
            val = -((z - lse) * Y).sum() / n + self.l2 / (2 * n) * (W ** 2).sum()
            G = P - Y
            return float(val), np.r_[(D.T @ G / n + self.l2 / n * W).ravel(), G.mean(axis=0)].astype(np.float64)

        res = minimize(loss, np.zeros(d * 3 + 3), jac=True, method="L-BFGS-B", options={"maxiter": 300})
        self.W, self.b = res.x[:d * 3].reshape(d, 3), res.x[d * 3:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._design(np.atleast_2d(X)) @ self.W + self.b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return dict(kind="logit", n_bins=self.n_bins, l2=self.l2, interact=self.interact, edges=[e.tolist() for e in self.edges],
                    group_edges=self.group_edges.tolist(), W=self.W.tolist(), b=self.b.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "BinnedLogit":
        m = cls(d["n_bins"], d["l2"], d["interact"])
        m.edges = [np.array(e) for e in d["edges"]]
        m.offsets = np.r_[0, np.cumsum([len(e) + 1 for e in m.edges])]
        m.group_edges, m.W, m.b = np.array(d["group_edges"]), np.array(d["W"]), np.array(d["b"])
        return m


class FastGBM:
    """scikit-learn 부스팅 트리를 numpy 배열로 옮긴 것. 한 줄 예측이 빠르고, 실행할 때 scikit-learn 이 필요 없다."""

    def __init__(self, d: dict):
        self.d = d
        self.feat, self.left, self.right = (np.array(d[k], dtype=np.int64) for k in ("feat", "left", "right"))
        self.thr, self.value = np.array(d["thr"], dtype=float), np.array(d["value"], dtype=float)
        self.leaf, self.miss_left = np.array(d["leaf"], dtype=bool), np.array(d["miss_left"], dtype=bool)
        self.roots, self.cls = np.array(d["roots"], dtype=np.int64), np.array(d["cls"], dtype=np.int64)
        self.baseline, self.depth = np.array(d["baseline"], dtype=float), int(d["depth"])

    @classmethod
    def from_sklearn(cls, m) -> "FastGBM":
        d = {k: [] for k in ("feat", "thr", "left", "right", "value", "leaf", "miss_left", "roots", "cls")}
        depth = 0
        for trees in m._predictors:
            for k, tree in enumerate(trees):
                nd, off = tree.nodes, len(d["feat"])
                d["roots"].append(off)
                d["cls"].append(k)
                d["feat"] += nd["feature_idx"].astype(int).tolist()
                d["thr"] += nd["num_threshold"].astype(float).tolist()
                d["left"] += (nd["left"].astype(int) + off).tolist()
                d["right"] += (nd["right"].astype(int) + off).tolist()
                d["value"] += nd["value"].astype(float).tolist()
                d["leaf"] += nd["is_leaf"].astype(bool).tolist()
                d["miss_left"] += nd["missing_go_to_left"].astype(bool).tolist()
                depth = max(depth, int(nd["depth"].max()))
        d["baseline"] = np.asarray(m._baseline_prediction, dtype=float).ravel().tolist()
        d["depth"] = depth
        return cls(d)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if len(X) > 20_000:                               # 큰 차트는 나눠서 계산해 메모리를 아낀다
            return np.vstack([self.predict_proba(X[i:i + 20_000]) for i in range(0, len(X), 20_000)])
        rows = np.arange(len(X))[:, None]
        node = np.tile(self.roots, (len(X), 1))
        for _ in range(self.depth):
            x = X[rows, self.feat[node]]
            go_left = np.where(np.isnan(x), self.miss_left[node], x <= self.thr[node])
            node = np.where(self.leaf[node], node, np.where(go_left, self.left[node], self.right[node]))
        raw = np.tile(self.baseline, (len(X), 1))
        v = self.value[node]
        for k in range(len(self.baseline)):
            raw[:, k] += v[:, self.cls == k].sum(axis=1)
        raw -= raw.max(axis=1, keepdims=True)
        p = np.exp(raw)
        return p / p.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"kind": "gbm", **self.d}


@dataclass
class OwnState:
    qty: float = 0.0          # 코인 수량(롱 +, 숏 -)
    avg_px: float = 0.0       # 평균 진입가
    last_action: str = ""     # 마지막으로 포지션을 바꾼 봉의 마감 시각(ISO)


def own_features(st: OwnState, px: float, equity: float, now: pd.Timestamp) -> dict:
    lev = st.qty * px / equity if equity > 0 else 0.0
    upnl = np.sign(st.qty) * (px / st.avg_px - 1) if st.qty and st.avg_px else 0.0
    hrs = (now - pd.Timestamp(st.last_action)).total_seconds() / 3600 if st.last_action else np.nan
    return {"own_lev": lev, "own_upnl": upnl, "hrs_since_action": hrs}


def save_bundle(bundle: dict, path: Path = MODEL_PATH) -> None:
    out = {**bundle, "model": bundle["model"].to_dict(), "size": {f"{k[0]}_{k[1]}": v for k, v in bundle["size"].items()}}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def load_bundle(path: Path = MODEL_PATH) -> dict:
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    m = b["model"]
    b["model"] = FastGBM(m) if m.get("kind") == "gbm" else BinnedLogit.from_dict(m)
    b["size"] = {(int(k.split("_")[0]), k.split("_")[1]): v for k, v in b["size"].items()}
    return b


def choose(bundle: dict, proba: np.ndarray) -> int:
    """확률 → 행동. 기준선은 워뇨띠의 실제 매수·매도 빈도에 맞춰 보정된 값(thr × thr_scale)."""
    k = bundle.get("thr_scale", 1.0)
    sb = proba[CLASSES.index(BUY)] / (bundle["thr_buy"] * k)
    ss = proba[CLASSES.index(SELL)] / (bundle["thr_sell"] * k)
    if max(sb, ss) < 1:
        return NONE
    return BUY if sb >= ss else SELL


def decide_x(bundle: dict, m: np.ndarray, px: float, st: OwnState, equity: float, now: pd.Timestamp) -> tuple[float, int]:
    """m: MARKET_FEATS 순서의 차트 상태 값. (매매할 코인 수량(부호 포함), 행동)을 돌려준다."""
    if equity <= 0 or np.isnan(m).any():
        return 0.0, NONE
    own = own_features(st, px, equity, now)
    x = np.r_[m, [own[k] for k in OWN_FEATS]]
    act = choose(bundle, bundle["model"].predict_proba(x.reshape(1, -1))[0])
    if act == NONE:
        return 0.0, NONE
    lev = own["own_lev"]
    kind = "open" if lev == 0 or np.sign(lev) == act else "reduce"
    new_lev = float(np.clip(lev + act * bundle["size"][(act, kind)], bundle["lev_min"], bundle["lev_max"]))
    return (new_lev - lev) * equity / px, act


def decide(bundle: dict, row: pd.Series, st: OwnState, equity: float) -> tuple[float, int]:
    """row: 방금 마감된 1시간봉의 피처(인덱스 = 봉 마감 시각)."""
    return decide_x(bundle, row[MARKET_FEATS].to_numpy(dtype=float), row["px"], st, equity, row.name)


def apply_fill(st: OwnState, dqty: float, px: float, now: pd.Timestamp) -> float:
    """체결을 반영하고 실현손익을 돌려준다."""
    realized = 0.0
    if st.qty and np.sign(dqty) != np.sign(st.qty):          # 줄이거나 반전
        closed = min(abs(dqty), abs(st.qty))
        realized = np.sign(st.qty) * closed * (px - st.avg_px)
        new_qty = st.qty + dqty
        if new_qty and np.sign(new_qty) != np.sign(st.qty):  # 반전: 남은 수량은 새 진입
            st.avg_px = px
        st.qty = new_qty
    else:                                                     # 새 진입이거나 추가
        st.avg_px = (st.avg_px * abs(st.qty) + px * abs(dqty)) / (abs(st.qty) + abs(dqty))
        st.qty += dqty
    if abs(st.qty) < 1e-12:
        st.qty, st.avg_px = 0.0, 0.0
    st.last_action = now.isoformat()
    return realized
