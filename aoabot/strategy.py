"""워뇨띠 체결 분석에서 뽑은 습관을 매매 규칙으로 옮긴다.

- 변동성이 평소보다 클 때만 진입한다. 진입 시점의 변동성 레짐 > 1.5 비율이 평소의 2배였다.
- 직전 4시간 급변의 반대 방향으로 들어간다. 급락 뒤 롱 비중 57%, 급등 뒤 42%였다.
- 물타기는 하지 않고, 수익이 난 방향으로만 한 번 더 싣는다. 불타기 승률 83%, 물타기 67%였다.
- 손실 포지션을 오래 끌지 않는다. 손실 거래의 평균 보유가 43시간으로 나머지의 4배였다.
  그래서 손절가를 진입과 동시에 걸고, 정해진 시간이 지나면 정리한다.

백테스트와 실매매가 이 모듈의 decide()를 똑같이 호출한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class Params:
    move_th: float = 0.03        # 직전 4시간 로그수익률이 이만큼 움직이면 역방향 진입
    vol_gate: float = 1.6        # 변동성 레짐(24시간 ATR / 30일 평균)이 이 이상일 때만 진입
    stop_atr: float = 3.0        # 손절 거리 = 1시간봉 ATR × stop_atr
    max_hold_h: int = 12         # 이 시간이 지나면 손익과 무관하게 정리
    risk_per_trade: float = 0.01  # 손절 시 잃는 금액 = 자산 × risk_per_trade
    max_leverage: float = 2.0    # 포지션 명목가치 상한 = 자산 × max_leverage
    add_at_atr: float = 1.0      # 첫 진입가보다 ATR × add_at_atr 만큼 유리해지면 추가 진입
    add_frac: float = 0.5        # 추가 진입 수량 = 첫 진입 수량 × add_frac
    max_adds: int = 1


@dataclass
class Position:
    symbol: str
    side: int          # 1 롱, -1 숏
    qty: float         # 코인 수량
    entry_px: float    # 평균 진입가
    init_px: float     # 첫 진입가
    init_qty: float
    stop_px: float
    atr_px: float      # 진입 당시 1시간봉 ATR(가격 단위)
    opened_at: str     # ISO 시각(UTC)
    adds: int = 0


@dataclass
class Enter:
    side: int
    qty: float
    stop_px: float
    atr_px: float
    reason: str


@dataclass
class Add:
    qty: float
    stop_px: float
    reason: str


@dataclass
class Exit:
    reason: str


Decision = Enter | Add | Exit | None


def decide(row: pd.Series, pos: Position | None, equity: float, p: Params) -> Decision:
    """row: 방금 마감된 1시간봉의 피처(인덱스 = 봉 마감 시각). 손절 체결은 호출 쪽이 처리한다."""
    px = row["px"]
    if any(math.isnan(row[k]) for k in ("r_4h", "vol_regime", "atr24")) or equity <= 0:
        return None
    atr_px = row["atr24"] * px

    if pos is not None:
        held_h = (row.name - pd.Timestamp(pos.opened_at)).total_seconds() / 3600
        if held_h >= p.max_hold_h:
            return Exit(f"보유 {held_h:.0f}시간 경과")
        gain = pos.side * (px - pos.init_px)
        if pos.adds < p.max_adds and gain >= p.add_at_atr * pos.atr_px:
            add_qty = pos.init_qty * p.add_frac
            if (pos.qty + add_qty) * px <= p.max_leverage * equity:
                return Add(add_qty, pos.init_px, "수익 방향 추가 진입, 손절가를 첫 진입가로 올림")
        return None

    if row["vol_regime"] < p.vol_gate:
        return None
    if row["r_4h"] <= -p.move_th:
        side = 1
    elif row["r_4h"] >= p.move_th:
        side = -1
    else:
        return None
    stop_dist = p.stop_atr * atr_px
    qty = min(p.risk_per_trade * equity / stop_dist, p.max_leverage * equity / px)
    reason = f"4시간 {row['r_4h']:+.1%}, 변동성 레짐 {row['vol_regime']:.2f}"
    return Enter(side, qty, px - side * stop_dist, atr_px, reason)


def stop_hit(pos: Position, low: float, high: float) -> bool:
    return low <= pos.stop_px if pos.side == 1 else high >= pos.stop_px
