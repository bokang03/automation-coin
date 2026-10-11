"""주문 실행. PaperBroker는 실제 시세로 가상 체결만 하고, LiveBroker는 거래소에 실제 주문을 넣는다."""
from __future__ import annotations

import os

import ccxt
import pandas as pd


def make_exchange(cfg: dict, with_keys: bool) -> ccxt.Exchange:
    opts = {"enableRateLimit": True}
    if with_keys:
        key, secret = os.environ.get("AOA_API_KEY"), os.environ.get("AOA_API_SECRET")
        if not key or not secret:
            raise SystemExit("실매매에는 환경변수 AOA_API_KEY, AOA_API_SECRET 이 필요합니다.")
        opts |= {"apiKey": key, "secret": secret}
    ex = getattr(ccxt, cfg["exchange"])(opts)
    if cfg.get("demo"):
        ex.enable_demo_trading(True)
    return ex


def fetch_closed_bars(ex: ccxt.Exchange, symbol: str, limit: int = 1500, timeframe: str = "1h") -> pd.DataFrame:
    """마감된 봉만 돌려준다. timestamp는 봉 마감 시각(UTC) — 백테스트 데이터와 같은 기준.
    1시간봉은 50일 이동평균(1,200봉)을 계산할 수 있도록 1,500봉을 받는다. 거래소가 한 번에 주는 개수가 적으면 나눠 받는다."""
    step = pd.Timedelta(ex.parse_timeframe(timeframe), unit="s")
    step_ms = int(step.total_seconds() * 1000)
    now = pd.Timestamp.now(tz="UTC")
    since, rows = int((now - step * (limit + 1)).timestamp() * 1000), []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        rows += batch
        if len(batch) < 2 or batch[-1][0] + step_ms > now.timestamp() * 1000:
            break
        since = batch[-1][0] + step_ms
    c = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    c["timestamp"] = pd.to_datetime(c.t, unit="ms", utc=True) + step
    return c[c.timestamp <= now].drop(columns="t").reset_index(drop=True)


class PaperBroker:
    live = False

    def __init__(self, ex: ccxt.Exchange, state: dict, fee: float, slip: float):
        self.ex, self.state, self.fee, self.slip = ex, state, fee, slip

    def last_price(self, symbol: str) -> float:
        return float(self.ex.fetch_ticker(symbol)["last"])

    def equity(self, positions: dict) -> float:
        upnl = sum(p.side * p.qty * (self.last_price(s) - p.entry_px) for s, p in positions.items())
        return self.state["cash"] + upnl

    def market(self, symbol: str, side: int, qty: float, reduce_only: bool = False) -> float:
        px = self.last_price(symbol) * (1 + side * self.slip)
        self.state["cash"] -= qty * px * self.fee
        return px

    def realize(self, pnl: float) -> None:
        self.state["cash"] += pnl

    def place_stop(self, symbol: str, side: int, qty: float, stop_px: float) -> None:
        pass  # 모의매매는 봇 루프가 시세를 보고 직접 손절한다

    def cancel_stops(self, symbol: str) -> None:
        pass

    def position_qty(self, symbol: str) -> float | None:
        return None


class LiveBroker:
    live = True

    def __init__(self, ex: ccxt.Exchange, leverage: int, symbols: list[str]):
        self.ex = ex
        ex.load_markets()
        for s in symbols:
            try:
                ex.set_leverage(leverage, s)
            except Exception as e:  # 이미 설정돼 있거나 거래소가 지원하지 않으면 넘어간다
                print(f"레버리지 설정 건너뜀 {s}: {e}")

    def last_price(self, symbol: str) -> float:
        return float(self.ex.fetch_ticker(symbol)["last"])

    def equity(self, positions: dict) -> float:
        bal = self.ex.fetch_balance()
        info = bal.get("info") or {}
        return float(info.get("totalMarginBalance") or bal["total"]["USDT"])

    def _amount(self, symbol: str, qty: float) -> float:
        return float(self.ex.amount_to_precision(symbol, qty))

    def market(self, symbol: str, side: int, qty: float, reduce_only: bool = False) -> float:
        params = {"reduceOnly": True} if reduce_only else {}
        o = self.ex.create_order(symbol, "market", "buy" if side > 0 else "sell", self._amount(symbol, qty), None, params)
        return float(o.get("average") or o.get("price") or self.last_price(symbol))

    def realize(self, pnl: float) -> None:
        pass  # 거래소 잔고가 곧 기록

    def place_stop(self, symbol: str, side: int, qty: float, stop_px: float) -> None:
        """side: 보유 포지션 방향. 반대 방향 STOP_MARKET(reduceOnly)을 건다."""
        self.ex.create_order(symbol, "market", "sell" if side > 0 else "buy", self._amount(symbol, qty), None,
                             {"stopLossPrice": float(self.ex.price_to_precision(symbol, stop_px)), "reduceOnly": True})

    def cancel_stops(self, symbol: str) -> None:
        for o in self.ex.fetch_open_orders(symbol):
            self.ex.cancel_order(o["id"], symbol)

    def position_qty(self, symbol: str) -> float | None:
        for p in self.ex.fetch_positions([symbol]):
            n = float(p.get("contracts") or 0) * float(p.get("contractSize") or 1)
            if n:
                return n if p.get("side") == "long" else -n
        return 0.0
