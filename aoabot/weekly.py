"""주봉 전략: 워뇨띠가 크게 팔던 모양의 주봉이 마감되면 1주일 동안 롱을 잡는다 (워뇨띠와 반대 방향).

근거(scripts/25_tf_strategies.py): 2018~2021년 워뇨띠는 주봉 단위로는 오르는 것을 팔았는데, 그 뒤 1주일 동안 가격은 평균보다
더 올랐다(크게 판 주 뒤 +3.0%, 그런 모양의 주봉 뒤 +5.0%, t −3.0). 워뇨띠의 주 단위 매매는 거꾸로 맞았으므로 반대로 쓴다.
신호 모델·기준선·보유 기간·크기는 models/aoa_tf_1w.json (같은 스크립트가 저장).
백테스트(scripts/25, 26)와 실행 루프(aoabot.timing_bot)가 decide() 를 같이 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aoabot.features_k import K_FEATS, make_features_k, to_higher
from aoabot.policy import BUY, SELL, load_bundle

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "aoa_tf_1w.json"
DAILY_BARS = 120        # 주봉 지표는 최근 10주가 필요하다


def load() -> dict:
    b = load_bundle(MODEL_PATH)
    b["plan"] = {int(k): (int(v[0]), int(v[1])) for k, v in b["meta"]["plan"].items()}
    b["sizes"] = {k[0]: v for k, v in b["size"].items()}
    return b


def decide(bundle: dict, daily: pd.DataFrame) -> tuple[pd.Timestamp, int, int, float]:
    """daily: 인덱스 = 일봉 마감 시각(UTC, tz 없음), 열 = open, high, low, close, volume.
    (가장 최근에 마감된 주봉의 마감 시각, 방향(+1 롱, -1 숏, 0 없음), 보유 주 수, 크기(자산 대비))를 돌려준다."""
    f = make_features_k(to_higher(daily, "W"))
    row = f.iloc[-1]
    if row[K_FEATS].isna().any():
        return f.index[-1], 0, 0, 0.0
    p = bundle["model"].predict_proba(row[K_FEATS].to_numpy(dtype=float).reshape(1, -1))[0]
    for sig, hit in ((BUY, p[2] >= bundle["thr_buy"]), (SELL, p[0] >= bundle["thr_sell"])):
        if sig in bundle["plan"] and hit:
            side, hold = bundle["plan"][sig]
            return f.index[-1], side, hold, float(bundle["sizes"][sig])
    return f.index[-1], 0, 0, 0.0
