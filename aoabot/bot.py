"""매매봇 실행 루프.

  모의매매:  python -m aoabot.bot
  한 번만:   python -m aoabot.bot --once
  실매매:    config.yaml 의 mode: live  그리고  python -m aoabot.bot --live   (둘 다 있어야 실제 주문)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests
import yaml

from aoabot import risk
from aoabot.brokers import LiveBroker, PaperBroker, fetch_closed_bars, make_exchange
from aoabot.features import make_features
from aoabot.strategy import Add, Enter, Exit, Params, Position, decide

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("aoabot")


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "bot.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def notify(msg: str) -> None:
    log.info(msg)
    url = os.environ.get("AOA_DISCORD_WEBHOOK")
    if url:
        try:
            requests.post(url, json={"content": msg[:1900]}, timeout=10)
        except requests.RequestException as e:
            log.warning("디스코드 알림 실패: %s", e)


class Bot:
    def __init__(self, cfg: dict, live: bool):
        self.cfg, self.live = cfg, live
        self.symbols = cfg["symbols"]
        self.params = Params(**cfg.get("strategy", {}))
        self.limits = risk.Limits(**cfg.get("risk", {}))
        self.state_path = ROOT / "state" / ("live.json" if live else "paper.json")
        self.state = self._load_state()
        self.positions = {s: Position(**p) for s, p in self.state["positions"].items()}
        self.rs = risk.RiskState(**self.state["risk"])
        if live:
            ex = make_exchange(cfg, with_keys=True)
            self.broker = LiveBroker(ex, int(cfg.get("exchange_leverage", 3)), self.symbols)
            self._check_live_sync()
        else:
            ex = make_exchange(cfg, with_keys=False)
            self.broker = PaperBroker(ex, self.state, cfg["paper"]["fee"], cfg["paper"]["slippage"])
        self.data_ex = ex

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"cash": float(self.cfg["paper"]["start_equity"]), "positions": {}, "last_bar": {},
                "risk": asdict(risk.RiskState())}

    def _save(self) -> None:
        self.state["positions"] = {s: asdict(p) for s, p in self.positions.items()}
        self.state["risk"] = asdict(self.rs)
        self.state_path.parent.mkdir(exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _check_live_sync(self) -> None:
        """봇 기록과 거래소 포지션이 다르면 사람이 정리하기 전까지 시작하지 않는다."""
        for s in self.symbols:
            ex_qty = self.broker.position_qty(s) or 0.0
            p = self.positions.get(s)
            mine = p.side * p.qty if p else 0.0
            if abs(ex_qty - mine) > max(abs(mine), abs(ex_qty)) * 0.02 + 1e-9:
                raise SystemExit(f"{s}: 거래소 포지션 {ex_qty} 와 봇 기록 {mine} 이 다릅니다. "
                                 f"거래소에서 직접 정리하거나 {self.state_path} 를 맞춘 뒤 다시 시작하세요.")

    def _record_trade(self, s: str, p: Position, exit_px: float, reason: str) -> None:
        pnl = p.side * p.qty * (exit_px - p.entry_px)
        path = ROOT / "logs" / "trades.csv"
        row = pd.DataFrame([dict(symbol=s, side="롱" if p.side > 0 else "숏", opened=p.opened_at,
                                 closed=pd.Timestamp.now(tz="UTC").isoformat(), entry=p.entry_px, exit=exit_px,
                                 qty=p.qty, adds=p.adds, gross_pnl=round(pnl, 4), reason=reason)])
        row.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        notify(f"[{'실매매' if self.live else '모의'}] {s} {'롱' if p.side > 0 else '숏'} 청산 ({reason}) "
               f"{p.entry_px:,.1f} → {exit_px:,.1f}, 손익 {pnl:+,.2f} USDT")

    def _close(self, s: str, reason: str) -> None:
        p = self.positions[s]
        self.broker.cancel_stops(s)
        px = self.broker.market(s, -p.side, p.qty, reduce_only=True)
        self.broker.realize(p.side * p.qty * (px - p.entry_px))
        self._record_trade(s, p, px, reason)
        del self.positions[s]

    def step(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        for s in self.symbols:
            p = self.positions.get(s)
            if p is not None:
                if self.live:
                    if (self.broker.position_qty(s) or 0.0) == 0.0:  # 거래소 손절 주문이 체결됨
                        self.broker.cancel_stops(s)
                        self._record_trade(s, p, p.stop_px, "손절(거래소)")
                        del self.positions[s]
                else:
                    last = self.broker.last_price(s)
                    if (p.side > 0 and last <= p.stop_px) or (p.side < 0 and last >= p.stop_px):
                        self._close(s, "손절")

        equity = self.broker.equity(self.positions)
        risk.update(self.rs, now.strftime("%Y-%m-%d"), equity, self.limits)
        if self.rs.halted:
            for s in list(self.positions):
                self._close(s, "최대 낙폭 정지")
            self._save()
            notify(f"봇 정지: {self.rs.halt_reason}. 확인 후 state 파일의 risk.halted 를 false 로 바꿔야 다시 진입합니다.")
            raise SystemExit(1)

        for s in self.symbols:
            bars = fetch_closed_bars(self.data_ex, s)
            last_bar = bars.timestamp.iloc[-1].isoformat()
            if self.state["last_bar"].get(s) == last_bar:
                continue
            row = make_features(bars).iloc[-1]
            p = self.positions.get(s)
            d = decide(row, p, equity, self.params)
            # 주문이 실패해도 같은 봉으로 매분 다시 주문하지 않도록 먼저 처리 완료로 기록한다
            self.state["last_bar"][s] = last_bar
            try:
                self._execute(s, row, p, d, equity)
            finally:
                self._save()
            log.info("%s 봉 %s 처리: 가격 %.1f, 4시간 %+.2f%%, 변동성 레짐 %.2f, 결정 %s",
                     s, last_bar, row.px, row.r_4h * 100, row.vol_regime, type(d).__name__ if d else "없음")
        self._save()
        log.info("자산 %.2f USDT, 포지션 %s", self.broker.equity(self.positions),
                 {s: f"{'롱' if p.side > 0 else '숏'} {p.qty:.4f}" for s, p in self.positions.items()} or "없음")

    def _execute(self, s: str, row: pd.Series, p: Position | None, d, equity: float) -> None:
        mode = "실매매" if self.live else "모의"
        if isinstance(d, Enter):
            ok, why = risk.can_open(self.rs, equity, self.limits)
            if not ok:
                log.info("%s 진입 신호 무시: %s", s, why)
                return
            px = self.broker.market(s, d.side, d.qty)
            stop = px - d.side * abs(row.px - d.stop_px)
            self.positions[s] = p = Position(s, d.side, d.qty, px, px, d.qty, stop, d.atr_px, row.name.isoformat())
            notify(f"[{mode}] {s} {'롱' if d.side > 0 else '숏'} 진입 {d.qty:.4f} @ {px:,.1f}, 손절 {stop:,.1f} ({d.reason})")
            self._protect(s, p)
        elif isinstance(d, Add):
            px = self.broker.market(s, p.side, d.qty)
            p.entry_px = (p.entry_px * p.qty + px * d.qty) / (p.qty + d.qty)
            p.qty += d.qty
            p.stop_px = d.stop_px
            p.adds += 1
            notify(f"[{mode}] {s} 추가 진입 {d.qty:.4f} @ {px:,.1f}, 손절가 {p.stop_px:,.1f}로 이동")
            self.broker.cancel_stops(s)
            self._protect(s, p)
        elif isinstance(d, Exit):
            self._close(s, d.reason)

    def _protect(self, s: str, p: Position) -> None:
        """손절 주문을 못 걸면 손절 없는 포지션을 남기지 않고 바로 청산한다."""
        try:
            self.broker.place_stop(s, p.side, p.qty, p.stop_px)
        except Exception as e:
            notify(f"{s} 손절 주문 실패({e}) — 즉시 청산합니다")
            self._close(s, "손절 주문 실패")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--live", action="store_true", help="config 의 mode: live 와 함께 있어야 실제 주문")
    ap.add_argument("--once", action="store_true", help="한 번만 확인하고 끝낸다")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    live = cfg.get("mode") == "live" and a.live
    if a.live != (cfg.get("mode") == "live"):
        raise SystemExit("실매매는 config.yaml 의 mode: live 와 --live 옵션이 둘 다 있어야 합니다.")
    setup_logging()
    bot = Bot(cfg, live)
    log.info("시작: %s, 거래소 %s%s, 종목 %s", "실매매" if live else "모의매매", cfg["exchange"],
             " (데모 계정)" if cfg.get("demo") else "", bot.symbols)
    while True:
        try:
            bot.step()
        except SystemExit:
            raise
        except Exception as e:  # 네트워크 오류 등은 기록만 하고 다음 주기에 다시 시도
            log.exception("처리 중 오류: %s", e)
        if a.once:
            break
        time.sleep(cfg.get("poll_seconds", 60))


if __name__ == "__main__":
    main()
