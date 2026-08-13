"""
RSM + 머신러닝 하이브리드 LME 가격 예측 연구
- 논문: Demand Forecasting of PO price using RSM and SVM (Wang et al.)
- 교수님 실험계획: x1=구리, x2=환율, x3=유가 → y=LME

시나리오 1: 머신러닝으로 직접 예측
시나리오 2: 머신러닝으로 RSM 계수(β) 예측 → RSM 식으로 시계열 예측
평가: MAE, RMSE, MAPE, R²
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# 평가지표
# ══════════════════════════════════════════════════════════
def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    # D-stat: 방향성 정확도 (다음 시점 상승/하락을 맞춘 비율, %)
    # 논문 평가지표. 실무에서 "오를지 내릴지"가 가격 절대값보다 중요할 때 유용
    if n >= 2:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dstat = np.mean(true_dir == pred_dir) * 100
    else:
        dstat = 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "Dstat": dstat}


# ══════════════════════════════════════════════════════════
# RSM 설계행렬 (논문 Table: Model 1/2/3)
# ══════════════════════════════════════════════════════════
# 변수 스케일이 매우 다르므로(구리~6000, 환율~1300, 유가~50)
# 표준화 후 다항항을 구성해야 수치 안정. 전역 스케일러를 fit 단계에서 고정.
_RSM_SCALER = {"mean": None, "std": None}


def set_rsm_scaler(X):
    """RSM 입력 변수의 표준화 기준을 설정 (학습 전체 기준)"""
    _RSM_SCALER["mean"] = X.mean(axis=0)
    _RSM_SCALER["std"] = X.std(axis=0) + 1e-9


def _std_X(X):
    if _RSM_SCALER["mean"] is None:
        return X
    return (X - _RSM_SCALER["mean"]) / _RSM_SCALER["std"]


def rsm_design_matrix(X, order=2):
    """
    X: (n, 3) 배열 — [x1=구리, x2=환율, x3=유가] (표준화 후 다항 구성)
    order=1: [1, x1, x2, x3]
    order=2: order1 + [x1^2, x2^2, x3^2]
    order=3: order2 + [x1^3, x2^3, x1^2*x2, x1*x2^2, x1*x2*x3]
    """
    Xs = _std_X(X)
    x1, x2, x3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    cols = [np.ones(len(Xs)), x1, x2, x3]

    if order >= 2:
        cols += [x1**2, x2**2, x3**2]

    if order >= 3:
        cols += [x1**3, x2**3, x1**2 * x2, x1 * x2**2, x1 * x2 * x3]

    return np.column_stack(cols)


def fit_rsm(X, y, order=2, ridge=1.0):
    """릿지 정규화 최소제곱으로 RSM 계수 β 추정 (소표본 안정화)"""
    Phi = rsm_design_matrix(X, order)
    p = Phi.shape[1]
    # (ΦᵀΦ + λI)β = Φᵀy  — 상수항은 정규화 제외
    reg = ridge * np.eye(p)
    reg[0, 0] = 0
    beta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
    return beta


def predict_rsm(X, beta, order=2):
    Phi = rsm_design_matrix(X, order)
    return Phi @ beta


# ══════════════════════════════════════════════════════════
# 시나리오 1 — 머신러닝 직접 예측
# ══════════════════════════════════════════════════════════
def scenario1_direct(Xtr, ytr, Xte, model_name):
    """외부변수(구리·환율·유가)로 LME를 직접 예측"""
    from sklearn.preprocessing import StandardScaler

    scx, scy = StandardScaler(), StandardScaler()
    Xtr_s = scx.fit_transform(Xtr)
    Xte_s = scx.transform(Xte)
    ytr_s = scy.fit_transform(ytr.reshape(-1, 1)).ravel()

    if model_name == "SVR":
        from sklearn.svm import SVR
        m = SVR(kernel="linear", C=10, epsilon=0.1)
        m.fit(Xtr_s, ytr_s)
        pred_s = m.predict(Xte_s)

    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=42, verbosity=0)
        m.fit(Xtr_s, ytr_s)
        pred_s = m.predict(Xte_s)

    elif model_name == "LSTM":
        return _lstm_direct(Xtr, ytr, Xte)

    else:
        raise ValueError(model_name)

    return scy.inverse_transform(pred_s.reshape(-1, 1)).ravel()


def _lstm_direct(Xtr, ytr, Xte):
    """LSTM 직접 예측 (외부변수 → LME)"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import StandardScaler

    scx, scy = StandardScaler(), StandardScaler()
    Xtr_s = scx.fit_transform(Xtr)
    Xte_s = scx.transform(Xte)
    ytr_s = scy.fit_transform(ytr.reshape(-1, 1)).ravel()

    # (samples, timesteps=1, features)
    Xtr_3d = Xtr_s.reshape(-1, 1, Xtr_s.shape[1])
    Xte_3d = Xte_s.reshape(-1, 1, Xte_s.shape[1])

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(32, input_shape=(1, Xtr_s.shape[1])),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr_3d, ytr_s, epochs=100, batch_size=8, verbose=0)
    pred_s = model.predict(Xte_3d, verbose=0).ravel()
    return scy.inverse_transform(pred_s.reshape(-1, 1)).ravel()


# ══════════════════════════════════════════════════════════
# 시나리오 2 — 머신러닝으로 RSM 계수(β) 예측 후 RSM 식으로 예측
# ══════════════════════════════════════════════════════════
def scenario2_ml_rsm(Xtr, ytr, Xte, model_name, order=2, window=18,
                     stabilize=True):
    """
    머신러닝으로 RSM 계수(β)를 예측 후, RSM 식으로 시계열 예측

    stabilize=True 이면 RSM 구조를 활용한 안정화 적용:
      ① 저차 우선: 1차 계수만 ML 예측, 2차 계수는 학습구간 평균 고정
      ② 평활: 예측 β를 직전 β와 지수평활로 혼합
      ③ 앵커링: 학습구간 β 범위를 벗어나면 클리핑
    """
    from sklearn.preprocessing import StandardScaler

    # 1) 학습구간 rolling RSM 계수 시퀀스
    beta_seq = []
    for i in range(window, len(Xtr) + 1):
        Xw = Xtr[i - window:i]
        yw = ytr[i - window:i]
        beta_seq.append(fit_rsm(Xw, yw, order))
    beta_seq = np.array(beta_seq)

    if len(beta_seq) < 3:
        beta = fit_rsm(Xtr, ytr, order)
        return predict_rsm(Xte, beta, order)

    n_params = beta_seq.shape[1]
    n_test = len(Xte)

    if stabilize:
        pred_betas = _forecast_beta_stable(beta_seq, model_name, n_test, order)
    else:
        pred_betas = _forecast_beta(beta_seq, model_name, n_test)

    # RSM 식에 대입
    preds = []
    for t in range(n_test):
        Phi_t = rsm_design_matrix(Xte[t:t+1], order)
        val = (Phi_t @ pred_betas[t]).ravel()[0]
        preds.append(float(val))
    return np.array(preds)


def _forecast_beta_stable(beta_seq, model_name, n_test, order, lookback=4,
                          smooth=0.5):
    """RSM 구조 활용 안정적 β 예측.

    핵심: 1차 항(상수+선형, 인덱스 0~3)만 ML로 예측하고,
    고차 항(제곱·교차)은 학습구간 평균으로 고정 → 자유도 절약·안정화
    """
    from sklearn.preprocessing import StandardScaler
    n_params = beta_seq.shape[1]

    # ① 저차/고차 분리: 1차 항(0~3)만 동적 예측, 나머지는 평균 고정
    n_linear = 4  # [상수, x1, x2, x3]
    fixed_high = beta_seq[:, n_linear:].mean(axis=0)  # 고차 계수 = 학습 평균

    # 학습구간 β 범위 (앵커링용)
    beta_min = beta_seq.min(axis=0)
    beta_max = beta_seq.max(axis=0)
    beta_range = beta_max - beta_min

    # ② 1차 항 시퀀스만 ML 학습
    linear_seq = beta_seq[:, :n_linear]

    if model_name == "LSTM":
        pred_linear = _lstm_beta(linear_seq, n_test)
    else:
        if len(linear_seq) < lookback + 2:
            pred_linear = np.tile(linear_seq[-1], (n_test, 1))
        else:
            sc = StandardScaler()
            ls = sc.fit_transform(linear_seq)
            models = []
            for j in range(n_linear):
                Xj, yj = [], []
                for i in range(lookback, len(ls)):
                    Xj.append(ls[i - lookback:i, j]); yj.append(ls[i, j])
                Xj, yj = np.array(Xj), np.array(yj)
                if model_name == "SVR":
                    from sklearn.svm import SVR
                    m = SVR(kernel="rbf", C=5, epsilon=0.05)
                elif model_name == "XGBoost":
                    from xgboost import XGBRegressor
                    m = XGBRegressor(n_estimators=80, max_depth=2,
                                     learning_rate=0.1, random_state=42, verbosity=0)
                else:
                    m = None
                if m is not None and len(Xj) >= 2:
                    m.fit(Xj, yj)
                models.append(m)

            hist = [ls[i].copy() for i in range(len(ls))]
            out = []
            prev = ls[-1].copy()
            for t in range(n_test):
                nxt = np.zeros(n_linear)
                for j in range(n_linear):
                    feat = np.array([hist[-k][j] for k in range(lookback, 0, -1)]).reshape(1, -1)
                    raw = float(models[j].predict(feat)[0]) if models[j] is not None else hist[-1][j]
                    # ② 평활: 직전 값과 혼합
                    nxt[j] = smooth * prev[j] + (1 - smooth) * raw
                out.append(nxt); hist.append(nxt); prev = nxt
            pred_linear = sc.inverse_transform(np.array(out))

    # ③ 조립: 1차 예측 + 고차 고정
    pred_betas = np.zeros((n_test, n_params))
    pred_betas[:, :n_linear] = pred_linear
    pred_betas[:, n_linear:] = fixed_high

    # ③ 앵커링: 학습 β 범위 ±50% 벗어나면 클리핑
    lo = beta_min - 0.5 * beta_range
    hi = beta_max + 0.5 * beta_range
    pred_betas = np.clip(pred_betas, lo, hi)

    return pred_betas


def _forecast_beta(beta_seq, model_name, n_test, lookback=4):
    """β 시퀀스를 받아 향후 n_test 스텝의 β를 예측.
    각 계수를 lag 기반으로 ML이 학습 → 재귀 예측."""
    from sklearn.preprocessing import StandardScaler
    n_params = beta_seq.shape[1]

    if model_name == "LSTM":
        return _lstm_beta(beta_seq, n_test)

    if len(beta_seq) < lookback + 2:
        return np.tile(beta_seq[-1], (n_test, 1))

    # 계수별로 독립 ML 모델 학습 (입력: 직전 lookback개, 출력: 다음 값)
    sc = StandardScaler()
    bs = sc.fit_transform(beta_seq)

    models = []
    for j in range(n_params):
        Xj, yj = [], []
        for i in range(lookback, len(bs)):
            Xj.append(bs[i - lookback:i, j])
            yj.append(bs[i, j])
        Xj, yj = np.array(Xj), np.array(yj)

        if model_name == "SVR":
            from sklearn.svm import SVR
            m = SVR(kernel="rbf", C=10, epsilon=0.05)
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
                             random_state=42, verbosity=0)
        else:
            m = None

        if m is not None and len(Xj) >= 2:
            m.fit(Xj, yj)
        models.append(m)

    # 재귀 예측
    hist = [bs[i].copy() for i in range(len(bs))]
    out = []
    for t in range(n_test):
        nxt = np.zeros(n_params)
        for j in range(n_params):
            feat = np.array([hist[-k][j] for k in range(lookback, 0, -1)]).reshape(1, -1)
            if models[j] is not None:
                nxt[j] = float(models[j].predict(feat)[0])
            else:
                nxt[j] = hist[-1][j]
        out.append(nxt)
        hist.append(nxt)

    return sc.inverse_transform(np.array(out))


def _lstm_beta(beta_seq, n_test, lookback=6):
    """LSTM으로 β 시퀀스를 학습해 미래 β 예측"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import StandardScaler

    if len(beta_seq) < lookback + 2:
        return np.tile(beta_seq[-1], (n_test, 1))

    sc = StandardScaler()
    bs = sc.fit_transform(beta_seq)

    X, Y = [], []
    for i in range(lookback, len(bs)):
        X.append(bs[i - lookback:i])
        Y.append(bs[i])
    X, Y = np.array(X), np.array(Y)

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(32, input_shape=(lookback, bs.shape[1])),
        Dense(bs.shape[1]),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=150, batch_size=4, verbose=0)

    # 재귀 예측
    hist = list(bs)
    out = []
    for t in range(n_test):
        feat = np.array(hist[-lookback:]).reshape(1, lookback, bs.shape[1])
        pv = model.predict(feat, verbose=0)[0]
        out.append(pv)
        hist.append(pv)
    return sc.inverse_transform(np.array(out))


# ══════════════════════════════════════════════════════════
# Model 5 — 분산-공분산 결합 (논문의 Combined Model)
# ══════════════════════════════════════════════════════════
def combine_variance_covariance(pred_a, pred_b, y_true_val):
    """
    두 모델 예측을 분산-공분산 가중으로 결합
    검증 구간의 오차 공분산으로 최적 가중치 산출
    """
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    y = np.asarray(y_true_val, dtype=float)
    n = min(len(pred_a), len(pred_b), len(y))
    ea = y[:n] - pred_a[:n]
    eb = y[:n] - pred_b[:n]

    var_a = np.var(ea)
    var_b = np.var(eb)
    cov_ab = np.cov(ea, eb)[0, 1] if n > 1 else 0

    denom = var_a + var_b - 2 * cov_ab
    if abs(denom) < 1e-12:
        w = 0.5
    else:
        w = (var_b - cov_ab) / denom
        w = np.clip(w, 0, 1)  # 가중치 0~1 제한

    return w  # pred_a의 가중치 (pred_b는 1-w)
