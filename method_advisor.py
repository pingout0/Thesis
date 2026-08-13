"""
실험 B: 예측 방법 검증·권장 시스템 (Method Advisor)

연구 결론:
  RSM 단독이 대부분 상황에서 우수하나 "항상"은 아니다.
  → 방법을 맹신하지 말고, 매 상황마다 검증 후 선택해야 한다.

이 시스템은 자동으로 하나를 강제하지 않는다. 대신:
  1. 여러 방법을 동일 조건에서 검증
  2. 권장안을 제시하되 근거(MAPE·방향성·안정성)를 함께 노출
  3. 사용자가 상황을 이해하고 판단하도록 지원
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from rsm_hybrid import (
    evaluate, fit_rsm, predict_rsm, set_rsm_scaler,
    scenario1_direct,
)
from step_forecast import scenario2_step_forecast


def method_rsm(Xtr, ytr, Xte, data_tr=None, data_te=None):
    set_rsm_scaler(Xtr)
    beta = fit_rsm(Xtr, ytr, order=2)
    return predict_rsm(Xte, beta, order=2)


def method_direct_svr(Xtr, ytr, Xte, data_tr=None, data_te=None):
    set_rsm_scaler(Xtr)
    return scenario1_direct(Xtr, ytr, Xte, "SVR")


def method_step(Xtr, ytr, Xte, data_tr=None, data_te=None):
    pred, _ = scenario2_step_forecast(data_tr, data_te, "RSM")
    return pred


METHODS = {
    "RSM 단독": method_rsm,
    "직접 예측(SVR)": method_direct_svr,
    "2단계 예측": method_step,
}


def advise(df, horizon):
    """
    각 방법을 검증하고 권장안 + 근거를 반환.
    강제 선택이 아니라 '검증된 정보'를 제공하는 것이 목적.
    """
    X = df[["구리", "환율", "유가WTI"]].values.astype(float)
    y = df["LME가격"].values.astype(float)
    n = len(y)

    n_val = min(horizon, n // 3)
    n_tr = n - n_val
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr:], y[n_tr:]

    data_tr = {"유가": df["유가WTI"].values[:n_tr], "구리": df["구리"].values[:n_tr],
               "환율": df["환율"].values[:n_tr], "LME": df["LME가격"].values[:n_tr]}
    data_te = {"유가": df["유가WTI"].values[n_tr:], "구리": df["구리"].values[n_tr:],
               "환율": df["환율"].values[n_tr:], "LME": df["LME가격"].values[n_tr:]}

    results = {}
    for name, fn in METHODS.items():
        try:
            pred = fn(Xtr, ytr, Xte, data_tr, data_te)
            results[name] = evaluate(yte, pred)
        except Exception:
            results[name] = {"MAPE": 999, "RMSE": 999, "Dstat": 0, "R2": -999}

    recommended = min(results.keys(),
                      key=lambda k: (results[k]["MAPE"], -results[k]["Dstat"]))

    # 2등과의 격차로 '확신도' 판단
    sorted_m = sorted(results.items(), key=lambda kv: kv[1]["MAPE"])
    gap = sorted_m[1][1]["MAPE"] - sorted_m[0][1]["MAPE"]
    if gap > 3:
        confidence = "높음 — 권장안이 뚜렷하게 우수"
    elif gap > 1:
        confidence = "보통 — 권장안이 다소 우수"
    else:
        confidence = "낮음 — 방법 간 차이가 작아 다른 방법도 고려 가능"

    return recommended, results, confidence, n_val


def horizon_band(horizon):
    if horizon <= 3:
        return "단기", "발주 타이밍 결정용"
    elif horizon <= 6:
        return "중기", "분기 구매 계획용"
    else:
        return "장기", "연간 계약·예산 계획용"


if __name__ == "__main__":
    df = pd.read_csv("sample_multivariate.csv")
    df.columns = [c.strip() for c in df.columns]

    print("=" * 72)
    print("  예측 방법 검증·권장 시스템 (Method Advisor)")
    print("  ── 방법을 강제하지 않고, 검증 결과와 근거를 제공 ──")
    print("=" * 72)

    for horizon in [3, 6, 12, 24]:
        band, desc = horizon_band(horizon)
        rec, results, conf, n_val = advise(df, horizon)
        print(f"\n▶ 예측 {horizon}개월 [{band}] — {desc} (검증 {n_val}개월)")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["MAPE"]):
            mark = " ← 권장" if name == rec else ""
            print(f"    {name:16s} MAPE {m['MAPE']:6.2f}%  방향성 {m['Dstat']:5.1f}%{mark}")
        print(f"  권장: 【{rec}】  |  확신도: {conf}")

    print("\n" + "=" * 72)
    print("  ⚠ 핵심 원칙: RSM이 대체로 우수하나 '항상'은 아니다.")
    print("     상황마다 검증 후 선택하는 것이 올바른 사용법이다.")
    print("=" * 72)

