"""학습한 정책(aoabot.policy)으로 매매하는 실행 루프. config.yaml 의 algorithm: policy 일 때 쓰인다.

매 1시간봉 마감 때 매수/매도/가만히를 고르고 시장가로 포지션을 조절한다. 워뇨띠는 손절 주문을 걸지 않았으므로
종목별 손절 주문은 없고, 계좌 단위 안전장치(하루 손실 한도, 최대 낙폭 정지)만 둔다.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from aoabot import risk
from aoabot.bot import ROOT, log, notify
from aoabot.brokers import LiveBroker, PaperBroker, fetch_closed_bars, make_exchange
from aoabot.features import make_features
from aoabot.policy import BUY, MARKET_FEATS, NONE, OwnState, apply_fill, decide, load_bundle

KST = pd.Timedelta(hours=9)


class PolicyBot:
    def __init__(self, cfg: dict, live: bool):
        self.cfg, self.live = cfg, live
        self.symbols = cfg["symbols"]
        self.bundle = load_bundle()
        r = cfg.get("risk", {})
        # 매매 빈도는 정책이 정하므로 하루 진입 횟수 상한은 쓰지 않는다
        self.limits = risk.Limits(daily_loss=r.get("daily_loss", 0.03), max_drawdown=r.get("max_drawdown", 0.25))
        self.state_path = ROOT / "state" / ("live_policy.json" if live else "paper_policy.json")
        self.state = self._load_state()
        self.positions = {s: OwnState(**p) for s, p in self.state["positions"].items()}
        self.rs = risk.RiskState(**self.state["risk"])
        ex = make_exchange(cfg, with_keys=live)
        if live:
            self.broker = LiveBroker(ex, int(cfg.get("exchange_leverage", 3)), self.symbols)
            for s in self.symbols:
                ex_qty, mine = self.broker.position_qty(s) or 0.0, self.positions.get(s, OwnState()).qty
                if abs(ex_qty - mine) > max(abs(mine), abs(ex_qty)) * 0.02 + 1e-9:
                    raise SystemExit(f"{s}: 거래소 포지션 {ex_qty} 와 봇 기록 {mine} 이 다릅니다. "
                                     f"거래소에서 직접 정리하거나 {self.state_path} 를 맞춘 뒤 다시 시작하세요.")
        else:
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

    def equity(self) -> float:
        if self.live:
            return self.broker.equity({})
        return self.state["cash"] + sum(p.qty * (self.broker.last_price(s) - p.avg_px) for s, p in self.positions.items() if p.qty)

    def _trade(self, s: str, dqty: float, t: pd.Timestamp, why: str) -> None:
        st = self.positions.setdefault(s, OwnState())
        before = st.qty
        px = self.broker.market(s, int(np.sign(dqty)), abs(dqty))
        realized = apply_fill(st, dqty, px, t)
        self.broker.realize(realized)
        path = ROOT / "logs" / "policy_trades.csv"
        pd.DataFrame([dict(time=pd.Timestamp.now(tz="UTC").isoformat(), bar=t.isoformat(), symbol=s, qty=dqty, price=px,
                           position_before=before, position_after=st.qty, realized=round(realized, 4), reason=why)]
                     ).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        notify(f"[{'실매매' if self.live else '모의'}] {s} {'매수' if dqty > 0 else '매도'} {abs(dqty):.4f} @ {px:,.1f} "
               f"({why}), 포지션 {before:+.4f} → {st.qty:+.4f}, 실현손익 {realized:+,.2f} USDT")

    def step(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        equity = self.equity()
        risk.update(self.rs, (now + KST).strftime("%Y-%m-%d"), equity, self.limits)
        if self.rs.halted:
            for s, p in self.positions.items():
                if p.qty:
                    self._trade(s, -p.qty, now.tz_localize(None).floor("h"), "최대 낙폭 정지")
            self._save()
            notify(f"봇 정지: {self.rs.halt_reason}. 확인 후 state 파일의 risk.halted 를 false 로 바꿔야 다시 매매합니다.")
            raise SystemExit(1)

        for s in self.symbols:
            bars = fetch_closed_bars(self.data_ex, s)
            last_bar = bars.timestamp.iloc[-1].isoformat()
            if self.state["last_bar"].get(s) == last_bar:
                continue
            row = make_features(bars).iloc[-1]
            st = self.positions.setdefault(s, OwnState())
            if row[MARKET_FEATS].isna().any():
                log.warning("%s 지표를 계산할 봉이 부족해 이번 봉은 건너뜁니다 (받은 봉 %d개): %s", s, len(bars),
                            list(row[MARKET_FEATS].index[row[MARKET_FEATS].isna()]))
            dqty, act = decide(self.bundle, row, st, equity)
            self.state["last_bar"][s] = last_bar        # 주문이 실패해도 같은 봉으로 다시 주문하지 않는다
            try:
                if act != NONE and dqty:
                    ok, why = risk.can_open(self.rs, equity, self.limits)
                    if abs(st.qty + dqty) > abs(st.qty) and not ok:
                        log.info("%s %s 신호 무시: %s", s, "매수" if act == BUY else "매도", why)
                    else:
                        self._trade(s, dqty, row.name, "정책")
            finally:
                self._save()
            log.info("%s 봉 %s 처리: 가격 %.1f, 결정 %s, 포지션 %+.4f (자산 대비 %+.2f배)", s, last_bar, row.px,
                     {BUY: "매수", NONE: "없음"}.get(act, "매도"), st.qty, st.qty * row.px / equity if equity else 0.0)
        self._save()
        log.info("자산 %.2f USDT", self.equity())
