"""
논문 원본 방식: 2단계 예측 (Step Forecast)
- 1단계: 유가(x1) → 구리(x2) 예측  [원유가 금속가에 선행]
- 2단계: 유가(x1) + 예측된 구리(x2_hat) → LME(y) 예측

논문의 유가→PP→PO 구조를 우리 데이터(유가→구리→LME)에 대응.
β 예측 방식(scenario2_ml_rsm)과 비교하기 위한 대안 접근.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from rsm_hybrid import (
    evaluate, fit_rsm, predict_rsm, set_rsm_scaler,
    rsm_design_matrix,
)


def _ml_fit_predict(Xtr, ytr, Xte, model_name):
    """공통 ML 학습·예측 (표준화 포함)"""
    from sklearn.preprocessing import StandardScaler
    Xtr = np.atleast_2d(Xtr); Xte = np.atleast_2d(Xte)
    if Xtr.shape[0] == 1 and Xtr.shape[1] > 1 and len(ytr) > 1:
        Xtr = Xtr.T
    scx, scy = StandardScaler(), StandardScaler()
    Xtr_s = scx.fit_transform(Xtr)
    Xte_s = scx.transform(Xte)
    ytr_s = scy.fit_transform(ytr.reshape(-1, 1)).ravel()

    if model_name == "SVR":
        from sklearn.svm import SVR
        m = SVR(kernel="rbf", C=10, epsilon=0.05)
        m.fit(Xtr_s, ytr_s)
        p = m.predict(Xte_s)
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                         random_state=42, verbosity=0)
        m.fit(Xtr_s, ytr_s)
        p = m.predict(Xte_s)
    elif model_name == "RSM":
        # 1차 다항 RSM (단순 회귀)
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(Xtr_s))] + [Xtr_s[:, k] for k in range(Xtr_s.shape[1])]),
            ytr_s, rcond=None)
        Phi = np.column_stack([np.ones(len(Xte_s))] + [Xte_s[:, k] for k in range(Xte_s.shape[1])])
        p = Phi @ beta
    else:
        raise ValueError(model_name)

    return scy.inverse_transform(p.reshape(-1, 1)).ravel()


def scenario2_step_forecast(data_tr, data_te, model_name):
    """
    2단계 예측
    data_tr/te: dict {유가, 구리, LME} 각 배열

    1단계: 유가 → 구리 예측
    2단계: [유가, 예측구리] → LME 예측
    """
    oil_tr, cu_tr, lme_tr = data_tr["유가"], data_tr["구리"], data_tr["LME"]
    oil_te, cu_te, lme_te = data_te["유가"], data_te["구리"], data_te["LME"]

    # ── 1단계: 유가 → 구리
    cu_hat = _ml_fit_predict(oil_tr.reshape(-1, 1), cu_tr,
                             oil_te.reshape(-1, 1), model_name)

    # ── 2단계: [유가, 예측구리] → LME
    X2_tr = np.column_stack([oil_tr, cu_tr])        # 학습은 실제 구리
    X2_te = np.column_stack([oil_te, cu_hat])       # 예측은 예측된 구리
    lme_hat = _ml_fit_predict(X2_tr, lme_tr, X2_te, model_name)

    return lme_hat, cu_hat


def scenario1_direct_simple(data_tr, data_te, model_name):
    """비교용 시나리오 1: [유가, 실제구리, 환율] → LME 직접"""
    Xtr = np.column_stack([data_tr["유가"], data_tr["구리"], data_tr["환율"]])
    Xte = np.column_stack([data_te["유가"], data_te["구리"], data_te["환율"]])
    return _ml_fit_predict(Xtr, data_tr["LME"], Xte, model_name)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("sample_multivariate.csv")
    df.columns = [c.strip() for c in df.columns]

    def split(ratio):
        n = len(df); k = int(n * ratio)
        tr = {"유가": df["유가WTI"].values[:k], "구리": df["구리"].values[:k],
              "환율": df["환율"].values[:k], "LME": df["LME가격"].values[:k]}
        te = {"유가": df["유가WTI"].values[k:], "구리": df["구리"].values[k:],
              "환율": df["환율"].values[k:], "LME": df["LME가격"].values[k:]}
        return tr, te

    print("=" * 68)
    print("  2단계 예측(Step) vs 직접 예측(Direct) 비교")
    print("  구조: 유가 → 구리 → LME")
    print("=" * 68)

    for ratio, label in [(0.8, "단기 80:20"), (0.7, "중기 70:30"), (0.5, "장기 50:50")]:
        tr, te = split(ratio)
        print(f"\n[{label}]")
        for mdl in ["RSM", "SVR", "XGBoost"]:
            direct = scenario1_direct_simple(tr, te, mdl)
            step, cu_hat = scenario2_step_forecast(tr, te, mdl)
            md = evaluate(te["LME"], direct)
            ms = evaluate(te["LME"], step)
            # 1단계 구리 예측 정확도도 확인
            mc = evaluate(te["구리"], cu_hat)
            print(f"  {mdl:8s}  직접 {md['MAPE']:6.2f}%   2단계 {ms['MAPE']:6.2f}%   "
                  f"(1단계 구리예측 {mc['MAPE']:.1f}%)")
