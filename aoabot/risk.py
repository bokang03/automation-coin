"""계좌 단위 안전장치: 하루 손실 한도와 최대 낙폭 정지.

워뇨띠의 일간 최대 손실은 −44%(2021년)였다. 봇은 그 레버리지를 따라 하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Limits:
    daily_loss: float = 0.03     # 당일 시작 자산 대비 이만큼 잃으면 그날은 신규 진입 금지
    max_drawdown: float = 0.25   # 최고 자산 대비 이만큼 빠지면 전부 청산하고 봇 정지
    max_entries_per_day: int = 0  # 하루 신규 진입 횟수 상한. 0이면 제한 없음


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    entries_today: int = 0


def update(rs: RiskState, day: str, equity: float, lim: Limits) -> None:
    if rs.day != day:
        rs.day, rs.day_start_equity, rs.entries_today = day, equity, 0
    rs.peak_equity = max(rs.peak_equity, equity)
    if not rs.halted and rs.peak_equity > 0 and equity <= rs.peak_equity * (1 - lim.max_drawdown):
        rs.halted = True
        rs.halt_reason = f"최고 자산 {rs.peak_equity:,.0f} 대비 {equity / rs.peak_equity - 1:.1%} 하락"


def can_open(rs: RiskState, equity: float, lim: Limits) -> tuple[bool, str]:
    if rs.halted:
        return False, "봇 정지 상태: " + rs.halt_reason
    if rs.day_start_equity > 0 and equity <= rs.day_start_equity * (1 - lim.daily_loss):
        return False, f"오늘 손실 {equity / rs.day_start_equity - 1:.1%}로 한도 초과"
    if lim.max_entries_per_day and rs.entries_today >= lim.max_entries_per_day:
        return False, f"오늘 진입 {rs.entries_today}회로 상한 도달"
    return True, ""


def record_entry(rs: RiskState) -> None:
    rs.entries_today += 1
