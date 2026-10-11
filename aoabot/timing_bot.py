"""워뇨띠의 5분봉 매수 타이밍으로 매매하는 실행 루프. config.yaml 의 algorithm: timing 일 때 쓰인다.

5분봉이 마감될 때마다: 보유 시간이 지난 포지션은 청산하고, 강한 매수 신호가 폭락이 이어진 뒤에 나오면 시장가로 산다.
규칙과 숫자는 aoabot.timing.TimingParams (백테스트와 같은 값)이고, 신호는 models/aoa_timing.json 이다.

config.yaml 의 weekly.enabled 가 true 면 주봉 전략(aoabot.weekly)도 같은 계좌에서 따로 포지션을 잡는다:
주봉이 마감될 때 지난주 포지션을 정리하고, 신호가 있으면 1주일 동안 보유한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from aoabot import risk, weekly
from aoabot.bot import ROOT, log, notify
from aoabot.brokers import LiveBroker, PaperBroker, fetch_closed_bars, make_exchange
from aoabot.features5 import WARMUP_BARS, make_features_5m
from aoabot.policy import load_bundle
from aoabot.timing import TimingParams, signals

KST = pd.Timedelta(hours=9)
BAR = pd.Timedelta(minutes=5)


@dataclass
class Holding:
    qty: float = 0.0
    avg_px: float = 0.0
    first_entry: str = ""     # 진입한 봉의 마감 시각(ISO)
    last_entry: str = ""
    hold: int = 0             # 주봉 전략에서만 쓴다: 보유할 주 수


class TimingBot:
    def __init__(self, cfg: dict, live: bool):
        self.cfg, self.live = cfg, live
        self.symbols = cfg["symbols"]
        self.bundle = load_bundle(ROOT / "models" / "aoa_timing.json")
        self.p = TimingParams(**cfg.get("timing", {}))
        if self.p.entry != "taker":
            raise SystemExit("실행 봇은 시장가 진입(entry: taker)만 지원합니다. 지정가 진입은 백테스트에서만 됩니다.")
        r = cfg.get("risk", {})
        self.limits = risk.Limits(daily_loss=r.get("daily_loss", 0.03), max_drawdown=r.get("max_drawdown", 0.25))
        self.state_path = ROOT / "state" / ("live_timing.json" if live else "paper_timing.json")
        self.state = self._load_state()
        self.positions = {s: Holding(**p) for s, p in self.state["positions"].items()}
        self.wk = weekly.load() if cfg.get("weekly", {}).get("enabled") else None
        self.wk_positions = {s: Holding(**p) for s, p in self.state.setdefault("weekly_positions", {}).items()}   # qty 는 롱 +, 숏 -
        self.state.setdefault("weekly_last", {})
        self.state.setdefault("weekly_checked", {})
        self.rs = risk.RiskState(**self.state["risk"])
        ex = make_exchange(cfg, with_keys=live)
        if live:
            self.broker = LiveBroker(ex, int(cfg.get("exchange_leverage", 3)), self.symbols)
            for s in self.symbols:
                ex_qty = self.broker.position_qty(s) or 0.0
                mine = self.positions.get(s, Holding()).qty + self.wk_positions.get(s, Holding()).qty
                if abs(ex_qty - mine) > max(abs(mine), abs(ex_qty)) * 0.02 + 1e-9:
                    raise SystemExit(f"{s}: 거래소 포지션 {ex_qty} 와 봇 기록 {mine} 이 다릅니다. "
                                     f"거래소에서 직접 정리하거나 {self.state_path} 를 맞춘 뒤 다시 시작하세요.")
        else:
            self.broker = PaperBroker(ex, self.state, self.p.taker_fee, self.p.slip)
        self.data_ex = ex

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"cash": float(self.cfg["paper"]["start_equity"]), "positions": {}, "last_bar": {},
                "risk": asdict(risk.RiskState())}

    def _save(self) -> None:
        self.state["positions"] = {s: asdict(p) for s, p in self.positions.items() if p.qty}
        self.state["weekly_positions"] = {s: asdict(p) for s, p in self.wk_positions.items() if p.qty}
        self.state["risk"] = asdict(self.rs)
        self.state_path.parent.mkdir(exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def equity(self) -> float:
        if self.live:
            return self.broker.equity({})
        held = [(s, p) for d in (self.positions, self.wk_positions) for s, p in d.items() if p.qty]
        return self.state["cash"] + sum(p.qty * (self.broker.last_price(s) - p.avg_px) for s, p in held)

    def _log_trade(self, **row) -> None:
        path = ROOT / "logs" / "timing_trades.csv"
        pd.DataFrame([dict(time=pd.Timestamp.now(tz="UTC").isoformat(), **row)]).to_csv(
            path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")

    def _close(self, s: str, reason: str) -> None:
        h = self.positions[s]
        px = self.broker.market(s, -1, h.qty, reduce_only=True)
        pnl = h.qty * (px - h.avg_px)
        self.broker.realize(pnl)
        self._log_trade(symbol=s, action="청산", qty=h.qty, price=px, entry=h.avg_px, pnl=round(pnl, 4), reason=reason)
        notify(f"[{'실매매' if self.live else '모의'}] {s} 청산 ({reason}) {h.avg_px:,.1f} → {px:,.1f}, 손익 {pnl:+,.2f} USDT")
        del self.positions[s]

    def _buy(self, s: str, qty: float, bar: pd.Timestamp, why: str) -> None:
        px = self.broker.market(s, 1, qty)
        h = self.positions.setdefault(s, Holding())
        h.avg_px = (h.avg_px * h.qty + px * qty) / (h.qty + qty)
        h.qty += qty
        h.last_entry = bar.isoformat()
        h.first_entry = h.first_entry or bar.isoformat()
        self._log_trade(symbol=s, action="매수", qty=qty, price=px, entry=h.avg_px, pnl=0.0, reason=why)
        notify(f"[{'실매매' if self.live else '모의'}] {s} 매수 {qty:.4f} @ {px:,.1f} ({why}), {self.p.hold_bars * 5 // 60}시간 뒤 청산 예정")

    def _weekly_close(self, s: str, reason: str) -> None:
        h = self.wk_positions[s]
        side = 1 if h.qty > 0 else -1
        px = self.broker.market(s, -side, abs(h.qty), reduce_only=True)
        pnl = h.qty * (px - h.avg_px)
        self.broker.realize(pnl)
        self._log_trade(symbol=s, action="주봉 청산", qty=h.qty, price=px, entry=h.avg_px, pnl=round(pnl, 4), reason=reason)
        notify(f"[{'실매매' if self.live else '모의'}] {s} 주봉 전략 청산 ({reason}) {h.avg_px:,.1f} → {px:,.1f}, 손익 {pnl:+,.2f} USDT")
        del self.wk_positions[s]

    def _weekly_step(self, s: str, now: pd.Timestamp, equity: float) -> None:
        """하루에 한 번 일봉을 받아, 새 주봉이 마감됐으면 지난 포지션을 정리하고 신호가 있으면 새로 잡는다."""
        today = now.strftime("%Y-%m-%d")
        if self.state["weekly_checked"].get(s) == today:
            return
        d = fetch_closed_bars(self.data_ex, s, limit=weekly.DAILY_BARS, timeframe="1d")
        d = d.set_index(d.timestamp.dt.tz_localize(None))[["open", "high", "low", "close", "volume"]]
        wk, side, hold, size = weekly.decide(self.wk, d)
        self.state["weekly_checked"][s] = today
        if self.state["weekly_last"].get(s) == wk.isoformat():
            return
        self.state["weekly_last"][s] = wk.isoformat()         # 주문이 실패해도 같은 주봉으로 다시 주문하지 않는다
        try:
            h = self.wk_positions.get(s)
            if h and h.qty and round((wk - pd.Timestamp(h.last_entry)) / pd.Timedelta(days=7)) >= h.hold:
                self._weekly_close(s, f"{h.hold}주 경과")
            late = now.tz_localize(None) - wk > pd.Timedelta(hours=6)    # 주봉 마감 직후에만 진입한다(백테스트는 마감 종가에 진입)
            if side and s not in self.wk_positions:
                ok, why = risk.can_open(self.rs, equity, self.limits)
                if late or not ok:
                    log.info("%s 주봉 신호 건너뜀: %s", s, "주봉 마감 후 6시간이 지나 봇이 켜짐" if late else why)
                else:
                    px = self.broker.market(s, side, size * equity / float(d.close.iloc[-1]))
                    qty = side * size * equity / float(d.close.iloc[-1])
                    self.wk_positions[s] = Holding(qty=qty, avg_px=px, first_entry=wk.isoformat(), last_entry=wk.isoformat(), hold=hold)
                    self._log_trade(symbol=s, action="주봉 매수" if side > 0 else "주봉 매도", qty=qty, price=px, entry=px, pnl=0.0, reason="주봉 신호")
                    notify(f"[{'실매매' if self.live else '모의'}] {s} 주봉 전략 {'롱' if side > 0 else '숏'} {abs(qty):.4f} @ {px:,.1f}, {hold}주 뒤 청산 예정")
        finally:
            self._save()
        log.info("%s 주봉 %s 처리: 신호 %s, 주봉 포지션 %s", s, wk.date(), {1: "롱", -1: "숏", 0: "없음"}[side],
                 round(self.wk_positions[s].qty, 4) if s in self.wk_positions else "없음")

    def step(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        equity = self.equity()
        risk.update(self.rs, (now + KST).strftime("%Y-%m-%d"), equity, self.limits)
        if self.rs.halted:
            for s in list(self.positions):
                self._close(s, "최대 낙폭 정지")
            for s in list(self.wk_positions):
                self._weekly_close(s, "최대 낙폭 정지")
            self._save()
            notify(f"봇 정지: {self.rs.halt_reason}. 확인 후 state 파일의 risk.halted 를 false 로 바꿔야 다시 매매합니다.")
            raise SystemExit(1)

        for s in self.symbols:
            if self.wk is not None:
                self._weekly_step(s, now, equity)
            bars = fetch_closed_bars(self.data_ex, s, limit=WARMUP_BARS + 100, timeframe="5m")
            last_bar = bars.timestamp.iloc[-1].isoformat()
            if self.state["last_bar"].get(s) == last_bar:
                continue
            f = make_features_5m(bars)
            if pd.isna(f.vol_regime.iloc[-1]):
                log.warning("%s 지표를 계산할 봉이 부족해 이번 봉은 건너뜁니다 (받은 봉 %d개)", s, len(bars))
                self.state["last_bar"][s] = last_bar
                continue
            buy, sell = signals(self.bundle, f.iloc[-self.p.prior_window:])     # 백테스트와 같은 구간: 이번 봉 + 직전 봉들
            prior, t, px = int(buy[:-1].sum()), f.index[-1], float(f.px.iloc[-1])
            self.state["last_bar"][s] = last_bar            # 주문이 실패해도 같은 봉으로 다시 주문하지 않는다
            decision = "없음"
            try:
                h = self.positions.get(s)
                if h and h.qty:
                    age = int((t - pd.Timestamp(h.last_entry)) / BAR)
                    if age >= self.p.hold_bars or (self.p.exit_on_sell and sell[-1]):
                        self._close(s, "시간 경과" if age >= self.p.hold_bars else "매도 신호")
                        decision = "청산"
                if buy[-1] and prior >= self.p.min_prior:
                    ok, why = risk.can_open(self.rs, equity, self.limits)
                    held = self.positions.get(s, Holding()).qty * px / equity
                    add = min(self.p.size, self.p.max_lev - held)
                    if not ok:
                        log.info("%s 매수 신호 무시: %s", s, why)
                    elif add > 1e-9:
                        self._buy(s, add * equity / px, t, f"강한 매수 신호, 직전 {self.p.prior_window * 5 // 60}시간 신호 {prior}회")
                        decision = "매수"
            finally:
                self._save()
            log.info("%s 5분봉 %s 처리: 가격 %.1f, 매수 신호 %s (직전 신호 %d회), 결정 %s", s, last_bar, px, bool(buy[-1]), prior, decision)
        self._save()
        log.info("자산 %.2f USDT, 5분봉 포지션 %s, 주봉 포지션 %s", self.equity(), {s: round(p.qty, 4) for s, p in self.positions.items() if p.qty} or "없음",
                 {s: round(p.qty, 4) for s, p in self.wk_positions.items() if p.qty} or "없음")
