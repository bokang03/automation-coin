"""봉별 위치(1시간·일·주·월봉)를 입력에 넣은 신호가 실제 매매 성과를 바꾸는지: 보완용 구간(2022~2023)에서 같은 규칙으로 비교한다."""
import sys

import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.policy import ROOT, load_bundle
from aoabot.timing import TimingParams, simulate, stats

pd.set_option("display.width", 250)
DEV = ("2022-01-01", "2023-12-31")
NOL = risk.Limits(daily_loss=9.9, max_drawdown=9.9)
MODELS = {"현재 (5분봉 지표 34개)": "aoa_timing.json", "+ 1시간·일봉 위치": "aoa_timing_tf_1h_1d.json", "+ 1시간·일·주·월봉 위치": "aoa_timing_tf.json"}
charts = {"BTC": pd.read_parquet("data/binance_BTC_5m_202110_202610.parquet"), "ETH": pd.read_parquet("data/binance_ETH_5m_202110_202610.parquet")}
charts = {k: v[v.timestamp < "2024-01-02"] for k, v in charts.items()}      # 2024년 이후는 읽지 않는다
rows = []
for name, file in MODELS.items():
    if not (ROOT / "models" / file).exists():
        continue
    b = load_bundle(ROOT / "models" / file)
    p = TimingParams(size=round(b["meta"]["big_buy_size"], 2), max_lev=round(b["meta"]["big_buy_size"], 2))
    for coin, c in charts.items():
        sim, tr = simulate(c, b, p, *DEV, lim=NOL)
        rows.append(dict(신호=name, 코인=coin, 교차검증_로그손실=round(min(1, b["meta"].get("cv_logloss", float("nan"))), 5) if "cv_logloss" in b["meta"] else None,
                         워뇨띠_매수_AUC=round(b["meta"]["auc_buy"], 3), 기준선=round(b["thr_buy"], 3), **stats(sim, tr)))
d = pd.DataFrame(rows).drop(columns=["교차검증_로그손실"])
for k in ("수익", "최대낙폭", "단순보유", "승률", "시장노출시간"):
    d[k] = d[k].map("{:+.1%}".format if k in ("수익", "단순보유") else "{:.1%}".format)
print(d.round(2).to_string(index=False))
