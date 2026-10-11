"""타이밍 전략을 워뇨띠가 실제로 거래한 코인(체결 내역 기준, BTC·ETH 제외 10개)의 2024-01 이후 차트에 돌린다.
설정은 scripts/18_final_test.py 의 최종 설정 그대로다."""
import importlib.util
import sys

import pandas as pd

sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("final", "scripts/18_final_test.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

pd.set_option("display.width", 250)
COINS = ["XRP", "LTC", "BCH", "DOT", "DOGE", "LINK", "BNB", "EOS", "ADA", "TRX"]     # 워뇨띠 체결 수가 많은 순
START, END = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp.now(tz="UTC").floor("D")

rows = []
markets = ft.ex.load_markets()
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    if sym not in markets:
        print(f"  건너뜀: {coin} (바이낸스 선물에 없음)", flush=True)
        continue
    c = ft.fetch(sym, START - ft.WARMUP, END)
    if len(c) < 100_000:
        print(f"  건너뜀: {coin} (자료 {len(c)}봉뿐)", flush=True)
        continue
    rows += ft.run_all(f"{coin} {START:%Y-%m}~{c.timestamp.iloc[-1]:%Y-%m}", c, START.tz_localize(None), None)
    print("  완료:", coin, flush=True)
d = pd.DataFrame(rows)
d.to_csv("data/his_coins_test.csv", index=False, encoding="utf-8-sig")
m = d[d.설정 == "최종 설정 (시장가, 안전장치 있음)"]
show = m[["차트", "수익", "단순보유", "최대낙폭", "거래수", "승률", "거래당_수익bp", "시장노출시간"]].copy()
for k in ("수익", "단순보유"):
    show[k] = show[k].map("{:+.1%}".format)
for k in ("최대낙폭", "승률", "시장노출시간"):
    show[k] = show[k].map("{:.1%}".format)
print("\n== 워뇨띠가 거래한 코인: 최종 설정, 초기 10,000 USDT, 시장가 수수료, 계좌 안전장치 포함 ==")
print(show.round(1).to_string(index=False))
print("\n== 요약 ==")
for v, g in d.groupby("설정", sort=False):
    print(f"[{v}] 평균 {g.수익.mean():+.1%}, 중앙값 {g.수익.median():+.1%}, 수익 난 차트 {(g.수익 > 0).sum()}/{len(g)}, 최악 {g.수익.min():+.1%}, "
          f"최고 {g.수익.max():+.1%}, 평균 최대낙폭 {g.최대낙폭.mean():.1%}, 거래당 {g.거래당_수익bp.mean():+.1f}bp, 단순 보유 평균 {g.단순보유.mean():+.1%}")
