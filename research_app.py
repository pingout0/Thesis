"""
RSM-ML 하이브리드 LME 예측 연구 — 전용 대시보드
동남전자부품 × 동아대학교 산업경영공학과

실행: streamlit run research_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go

st.set_page_config(page_title="RSM 하이브리드 예측 연구", page_icon=" ",
                   layout="wide", initial_sidebar_state="expanded")

# ── 스타일
st.markdown("""
<style>
.hero { background:linear-gradient(135deg,#13233A,#1E3556); color:#fff;
        border-radius:14px; padding:26px 30px; margin-bottom:8px; }
.finding { background:#F4F6F9; color:#24303F; border-left:4px solid #C87941;
           border-radius:0 10px 10px 0; padding:14px 18px; margin:8px 0; font-size:14px; }
.keybox { background:#EEF3F8; color:#13233A; border-radius:10px; padding:16px 20px;
          margin:8px 0; font-size:14px; border:1px solid #D5DEE8; }
.warn { background:#FBEEE2; color:#5A3A1A; border-left:4px solid #C87941;
        border-radius:0 10px 10px 0; padding:14px 18px; margin:8px 0; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드
@st.cache_data
def load():
    return json.load(open("precomputed.json"))

try:
    D = load()
except Exception:
    st.error("precomputed.json이 필요합니다. precompute 스크립트를 먼저 실행하세요.")
    st.stop()

COPPER="#C87941"; NAVY="#13233A"; STEEL="#5B7A99"; GREEN="#2E7D5B"; RED="#B0413E"

# ══════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("##  RSM 하이브리드 연구")
    st.markdown("동남전자 × 동아대\n산업경영공학과")
    st.divider()
    st.markdown("**참고 논문**\nDemand Forecasting of PO price using RSM and SVM")
    st.divider()
    page = st.radio("섹션", [
        "① 연구 개요",
        "② 방법 비교 대시보드",
        "③ 핵심 발견: β 예측의 한계",
        "④ 차수·window 민감도",
        "⑤ 방법 권장 시스템",
        "⑥ 구간 분리 분석",
        "⑦ 데이터 증강",
    ])
    st.divider()
    st.caption("변수: 구리·환율·유가 → LME\n데이터: 108개월 (2017~2025)")

# ══════════════════════════════════════════
# ① 연구 개요
# ══════════════════════════════════════════
if page.startswith("①"):
    st.markdown("""
<div class="hero">
<div style="font-size:13px;color:#E8A66B;letter-spacing:2px;font-weight:bold">RESEARCH OVERVIEW</div>
<div style="font-size:30px;font-weight:bold;margin-top:6px">RSM–머신러닝 하이브리드 예측 연구</div>
<div style="font-size:15px;color:#DCE6F2;margin-top:10px">
논문의 방법론을 LME 알루미늄 가격(구리·환율·유가 → LME)에 적용하고,
RSM 단독·직접 예측·β 예측·2단계 예측을 비교 검증</div>
</div>
""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("데이터", "108 개월", "2017~2025")
    c2.metric("비교 방법", "4 종", "RSM·직접·β·2단계")
    c3.metric("평가지표", "5 종", "MAE·RMSE·MAPE·R²·D-stat")

    st.markdown("### 연구 접근")
    st.markdown("""
<div class="keybox">
<b>시나리오 1 — 직접 예측</b><br>
외부변수(구리·환율·유가)로 머신러닝(SVR/XGBoost/LSTM)이 LME를 직접 예측<br><br>
<b>시나리오 2 — ML + RSM 하이브리드</b><br>
① β 예측: 머신러닝이 RSM 회귀계수를 예측 → RSM 식으로 시계열 예측<br>
② 2단계 예측(논문 원본): 유가 → 구리 → LME 순차 예측<br><br>
<b>결합 — Model 5</b><br>
분산-공분산 기법으로 최우수 모델들을 가중 결합
</div>
""", unsafe_allow_html=True)

    st.markdown("### 핵심 발견 4가지")
    for t, d in [
        ("① RSM 입력변수 표준화 필수", "구리~6000, 환율~1300, 유가~50의 스케일 차이로 2차항 조건수 2×10¹¹까지 폭발 → 표준화·정규화로 안정화"),
        ("② β 예측의 근본적 한계", "RSM 계수가 시간에 따라 불규칙하게 요동침(환율² 변동계수 40.26) → 'ML로 β 예측'은 계수 안정성 가정에 의존"),
        ("③ 차수·window 민감도", "2차 RSM이 최적, 3차는 과적합 붕괴. window 최적값도 상황별로 다름 → 튜닝 필요"),
        ("④ RSM 단독의 범용적 우수성", "복잡한 하이브리드가 항상 낫지 않음. 단, '항상' RSM이 최고는 아니므로 상황별 검증 필요"),
    ]:
        st.markdown(f'<div class="finding"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)

    st.markdown("### 원자재 가격 추이 (학습 데이터)")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=D["dates"],y=D["lme"],name="LME(목표)",line=dict(color=NAVY,width=2.5)))
    fig.add_trace(go.Scatter(x=D["dates"],y=D["cu"],name="구리",line=dict(color=COPPER,width=1),yaxis="y2"))
    fig.update_layout(height=340,margin=dict(t=20,b=20),hovermode="x unified",
        yaxis=dict(title="LME ($/톤)"),yaxis2=dict(title="구리",overlaying="y",side="right"),
        legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════
# ② 방법 비교 대시보드
# ══════════════════════════════════════════
elif page.startswith("②"):
    st.markdown("## ② 방법 비교 대시보드")
    st.caption("학습:검증 비율별로 4개 방법의 성능 비교 (MAPE 낮을수록, D-stat 높을수록 우수)")

    band = st.radio("예측 기간", ["단기","중기","장기"], horizontal=True,
        help="단기=80:20, 중기=70:30, 장기=50:50")
    comp = D["comparison"][band]

    # 표
    rows=[]
    for method, m in comp.items():
        rows.append({"방법":method,"MAPE(%)":m["MAPE"],"RMSE":m["RMSE"],
                     "R²":m["R2"],"방향성(%)":m["Dstat"]})
    dfc=pd.DataFrame(rows).sort_values("MAPE(%)")
    best=dfc.iloc[0]["방법"]

    st.dataframe(dfc.style.highlight_min(subset=["MAPE(%)"],color="#D6EFE2")
                 .highlight_max(subset=["방향성(%)"],color="#FBEEE2"),
                 use_container_width=True, hide_index=True)
    st.markdown(f'<div class="keybox">🏆 <b>{band} 최적: {best}</b> — MAPE {dfc.iloc[0]["MAPE(%)"]:.2f}%, 방향성 {dfc.iloc[0]["방향성(%)"]:.0f}%</div>',unsafe_allow_html=True)

    # 막대그래프 2개
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(go.Bar(x=dfc["방법"],y=dfc["MAPE(%)"],
            marker_color=[GREEN if v==dfc["MAPE(%)"].min() else STEEL for v in dfc["MAPE(%)"]]))
        fig.update_layout(title="MAPE (낮을수록 우수)",height=320,margin=dict(t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure(go.Bar(x=dfc["방법"],y=dfc["방향성(%)"],
            marker_color=[COPPER if v==dfc["방향성(%)"].max() else STEEL for v in dfc["방향성(%)"]]))
        fig.add_hline(y=50,line_dash="dash",line_color=RED,annotation_text="랜덤(50%)")
        fig.update_layout(title="방향성 정확도 D-stat (높을수록 우수)",height=320,margin=dict(t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
<div class="finding">
📌 <b>해석</b> — MAPE는 '가격을 얼마나 정확히', 방향성(D-stat)은 '오를지 내릴지를 얼마나 맞추는지'를 봅니다.
발주 결정에서는 방향성이 특히 중요합니다. 어떤 모델은 MAPE는 낮아도 방향성이 랜덤 이하일 수 있어 함께 봐야 합니다.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ③ β 예측의 한계
# ══════════════════════════════════════════
elif page.startswith("③"):
    st.markdown("## ③ 핵심 발견: 왜 β 예측이 어려운가")
    st.markdown("""
<div class="keybox">
'머신러닝으로 RSM 계수(β)를 예측한다'는 아이디어는 <b>β가 시간에 따라 부드럽게 변한다</b>는 가정에 의존합니다.
실제로 그런지 rolling window로 β 시퀀스를 만들어 변동성을 측정했습니다.
</div>
""", unsafe_allow_html=True)

    bv = D["beta_volatility"]
    # CV 막대
    fig=go.Figure(go.Bar(
        x=[b["name"] for b in bv], y=[b["cv"] for b in bv],
        marker_color=[GREEN if b["cv"]<0.3 else (COPPER if b["cv"]<5 else RED) for b in bv],
        text=[f"{b['cv']:.2f}" for b in bv], textposition="outside"))
    fig.update_layout(title="RSM 계수별 변동계수(CV) — 클수록 불안정",
        height=360,margin=dict(t=40,b=20),yaxis_title="변동계수 (CV)")
    st.plotly_chart(fig,use_container_width=True)

    c1,c2,c3=st.columns(3)
    c1.metric("β(상수)","CV 0.08","안정 ✅")
    c2.metric("β(환율)","CV 11.60","불안정 ⚠️")
    c3.metric("β(환율²)","CV 40.26","극도 불안정 🔴")

    # β 시계열 선택 표시
    st.markdown("### 계수별 시간 변화 (직접 확인)")
    sel = st.multiselect("표시할 계수", [b["name"] for b in bv],
        default=["상수","β(환율)","β(환율²)"])
    fig=go.Figure()
    colors=[NAVY,COPPER,RED,STEEL,GREEN,"#8854C0","#C0A054"]
    for i,b in enumerate(bv):
        if b["name"] in sel:
            fig.add_trace(go.Scatter(y=b["series"],name=b["name"],
                line=dict(width=2,color=colors[i%7])))
    fig.update_layout(height=340,margin=dict(t=20,b=20),
        xaxis_title="rolling window 순서",yaxis_title="계수 값",hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
<div class="warn">
💡 <b>연구적 의미</b> — 상수항만 안정적이고 나머지 계수(특히 고차항·환율)는 부호까지 반전하며 요동칩니다.
이는 β 예측 접근의 <b>실패가 아니라, 방법의 적용 조건(계수 안정성)을 규명한 기여</b>입니다.
"이 방법은 언제 쓸 수 있고 언제 못 쓰는가"를 밝힌 것이 연구의 가치입니다.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ④ 차수·window 민감도
# ══════════════════════════════════════════
elif page.startswith("④"):
    st.markdown("## ④ 차수·window 민감도 분석")

    st.markdown("### RSM 차수별 과적합 검증")
    st.caption("학습 오차는 낮은데 검증 오차가 크게 벌어지면 = 과적합")
    band = st.radio("예측 기간", ["단기","중기","장기"], horizontal=True, key="deg")
    dd = D["degree"][band]

    fig=go.Figure()
    orders=list(dd.keys())
    fig.add_trace(go.Bar(name="학습 MAPE",x=orders,y=[dd[o]["학습"] for o in orders],marker_color=STEEL))
    fig.add_trace(go.Bar(name="검증 MAPE",x=orders,y=[dd[o]["검증"] for o in orders],marker_color=COPPER))
    fig.update_layout(barmode="group",height=340,margin=dict(t=20,b=20),
        yaxis_title="MAPE (%)",legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig,use_container_width=True)

    v3=dd["3차"]["검증"]; t3=dd["3차"]["학습"]
    st.markdown(f'<div class="finding">📌 <b>3차 과적합</b> — 학습 {t3:.1f}%로 가장 좋지만 검증 {v3:.1f}%로 폭발. <b>2차가 복잡도-일반화 균형점</b>입니다.</div>',unsafe_allow_html=True)

    st.divider()
    st.markdown("### window 크기 민감도 (β 예측)")
    st.caption("rolling RSM 계수를 만드는 기간. 최적값이 상황마다 다름")
    wband = st.radio("예측 기간", ["단기","중기"], horizontal=True, key="win")
    wd = D["window"][wband]

    fig=go.Figure()
    ws=list(wd.keys())
    for mdl,col in [("SVR",COPPER),("XGBoost",STEEL)]:
        fig.add_trace(go.Scatter(x=ws,y=[wd[w].get(mdl) for w in ws],
            name=mdl,mode="lines+markers",line=dict(width=2.5,color=col),marker=dict(size=9)))
    fig.update_layout(height=340,margin=dict(t=20,b=20),
        xaxis_title="window 크기 (개월)",yaxis_title="MAPE (%)",hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="finding">📌 <b>고정 최적값 없음</b> — 단기는 window 18, 중기는 30이 유리. 예측 기간에 맞춘 튜닝이 필요합니다.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════
# ⑤ 방법 권장 시스템
# ══════════════════════════════════════════
elif page.startswith("⑤"):
    st.markdown("## ⑤ 방법 권장 시스템 (Method Advisor)")
    st.markdown("""
<div class="keybox">
이 시스템은 <b>하나를 강제하지 않습니다.</b> 여러 방법을 같은 조건에서 검증하고,
권장안과 <b>확신도</b>를 함께 제시해 사용자가 상황을 이해하고 판단하도록 돕습니다.
</div>
""", unsafe_allow_html=True)

    horizon = st.select_slider("예측하려는 기간 (개월)", options=[3,6,12,24], value=6)
    band = "단기" if horizon<=3 else ("중기" if horizon<=6 else "장기")
    desc = {"단기":"발주 타이밍 결정용","중기":"분기 구매 계획용","장기":"연간 계약·예산 계획용"}[band]

    st.markdown(f"#### 예측 {horizon}개월 · [{band}] — {desc}")

    # 미리계산된 비교표에서 해당 band 결과 사용
    comp = D["comparison"][band]
    # RSM단독/직접(SVR)/2단계/β예측 중심으로 표시
    show={"RSM 단독":comp["RSM 단독"],"직접(SVR)":comp["직접(SVR)"],
          "β예측(SVR)":comp["β예측(SVR)"],"2단계":comp["2단계"]}
    ranked=sorted(show.items(),key=lambda kv:kv[1]["MAPE"])
    rec=ranked[0][0]
    gap=ranked[1][1]["MAPE"]-ranked[0][1]["MAPE"]
    conf = "높음 — 권장안이 뚜렷하게 우수" if gap>3 else ("보통 — 다소 우수, 대안도 고려" if gap>1 else "낮음 — 방법 간 차이 작음")

    for name,m in ranked:
        is_rec = name==rec
        icon = "✅ 권장" if is_rec else ""
        bar_color = GREEN if is_rec else STEEL
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:6px 0;
     padding:10px 16px;border-radius:8px;background:{'#EAF5EF' if is_rec else '#F4F6F9'};
     border-left:4px solid {bar_color}">
  <div style="font-size:15px;font-weight:bold;color:#13233A;width:130px">{name}</div>
  <div style="color:#24303F">MAPE <b>{m['MAPE']:.2f}%</b></div>
  <div style="color:#5B7A99">방향성 {m['Dstat']:.0f}%</div>
  <div style="margin-left:auto;color:{GREEN};font-weight:bold">{icon}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<div class="keybox">권장: <b>{rec}</b>  &nbsp;|&nbsp;  확신도: <b>{conf}</b></div>',unsafe_allow_html=True)

    st.markdown("""
<div class="warn">
⚠ <b>핵심 원칙</b> — RSM 단독이 대부분 상황에서 우수하지만 <b>'항상'은 아닙니다.</b>
방법을 맹신하지 말고, 상황마다 검증한 뒤 선택하는 것이 올바른 사용법입니다.
확신도가 '낮음/보통'일 때는 특히 대안 방법을 함께 검토해야 합니다.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ⑥ 구간 분리 분석
# ══════════════════════════════════════════
elif page.startswith("⑥"):
    st.markdown("## ⑥ 구간 분리 분석 (논문 방식)")
    st.markdown("""
<div class="keybox">
참고 논문은 데이터를 <b>전환점(turning point) 기준으로 여러 구간으로 나눠</b> 분석합니다.
전체로만 보면 놓치는 <b>시장 국면별 특성</b>을 드러내기 위함입니다.
우리도 LME 전환점(2020 코로나 저점 → 2022 전쟁 고점)으로 3구간을 나눠 재분석했습니다.
</div>
""", unsafe_allow_html=True)

    if "periods" not in D:
        st.warning("구간 데이터가 없습니다. precompute를 다시 실행하세요.")
        st.stop()

    P = D["periods"]

    # 구간별 요약 카드
    st.markdown("### 구간별 최적 방법")
    cols = st.columns(3)
    period_names = ["구간1 안정기","구간2 급등기","구간3 조정기"]
    period_colors = [GREEN, RED, COPPER]
    for col, pn, pc in zip(cols, period_names, period_colors):
        pdata = P[pn]
        with col:
            st.markdown(f"""
<div style="background:#F4F6F9;border-radius:10px;padding:16px;border-top:4px solid {pc}">
<div style="font-size:16px;font-weight:bold;color:{NAVY}">{pn}</div>
<div style="font-size:12px;color:#5B7A99;margin:4px 0 10px">{pdata['desc']} · {pdata['n']}개월</div>
<div style="font-size:13px;color:#24303F">최적 방법</div>
<div style="font-size:18px;font-weight:bold;color:{pc}">{pdata.get('best','-')}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="finding">📌 <b>구간마다 최적 방법이 다릅니다</b> — 안정기는 RSM 단독, 급등기는 2단계, 조정기는 직접 예측(SVR). 단일 방법이 모든 국면을 지배하지 않습니다.</div>', unsafe_allow_html=True)

    # 상세 비교표 (구간 선택)
    st.markdown("### 구간별 상세 성능")
    sel_p = st.radio("구간 선택", list(P.keys()), horizontal=True)
    pdata = P[sel_p]
    st.caption(f"{pdata['desc']} · {pdata['n']}개월")

    rows=[]
    for split_label, methods in pdata["splits"].items():
        row={"학습:검증": split_label}
        for m,v in methods.items():
            row[m] = v if v is not None else None
        rows.append(row)
    if rows:
        dfp = pd.DataFrame(rows)
        st.dataframe(dfp, use_container_width=True, hide_index=True)
        st.caption("숫자는 MAPE(%) — 낮을수록 정확")

    # 국면별 막대 비교 (70:30 기준)
    st.markdown("### 국면별 방법 성능 비교 (70:30 기준)")
    methods_all = ["RSM 단독","직접(SVR)","2단계"]
    fig = go.Figure()
    colors_m = {"RSM 단독":NAVY, "직접(SVR)":COPPER, "2단계":STEEL}
    for m in methods_all:
        vals=[]
        for pn in period_names:
            sp = P[pn]["splits"].get("70:30",{})
            vals.append(sp.get(m))
        fig.add_trace(go.Bar(name=m, x=period_names, y=vals, marker_color=colors_m[m]))
    fig.update_layout(barmode="group", height=360, margin=dict(t=20,b=20),
        yaxis_title="MAPE (%)", legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="warn">
💡 <b>연구 발견</b> — "최적 예측 방법은 <b>예측 기간 × 시장 국면</b>의 함수다."<br>
급등기엔 중간변수(구리)를 경유하는 2단계 예측이 안정적이고, 조정기엔 ML의 유연성이,
안정기엔 단순 RSM이 유리합니다. 이는 참고 논문의 "결합모델이 항상 최고"와 달리,
<b>데이터 국면에 따라 방법을 달리해야 함</b>을 보여줍니다.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ⑦ 데이터 증강
# ══════════════════════════════════════════
elif page.startswith("⑦"):
    st.markdown("## ⑦ 데이터 증강 (수업 내용 연계)")
    st.markdown("""
<div class="keybox">
논문·본 연구 공통의 한계는 <b>소표본(108개월)</b>입니다.
수업에서 학습한 데이터 증강 기법을 시계열에 적용해, 학습 데이터를 늘리면
예측이 개선되는지 검증했습니다. <b>증강은 학습셋에만</b> 적용하고
검증셋(실제 미래)은 원본을 유지해 데이터 누수를 막았습니다.
</div>
""", unsafe_allow_html=True)

    if "augmentation" not in D:
        st.warning("증강 데이터가 없습니다. precompute를 다시 실행하세요.")
        st.stop()

    A = D["augmentation"]

    # 증강 기법 설명
    st.markdown("### 적용한 증강 기법 (변수 관계 유지)")
    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="finding"><b>지터링</b><br>x + ε<br>작은 랜덤 노이즈 추가</div>', unsafe_allow_html=True)
    c2.markdown('<div class="finding"><b>부트스트랩</b><br>복원추출<br>기존 샘플 재추출</div>', unsafe_allow_html=True)
    c3.markdown('<div class="finding"><b>분포기반</b><br>다변량 정규분포<br>공분산 유지하며 생성</div>', unsafe_allow_html=True)

    # 증강 전후 비교 막대그래프
    st.markdown("### 증강 전후 성능 (108 → 324개, 70:30)")
    aug_labels = ["증강 없음","지터링","부트스트랩","분포기반"]
    fig = go.Figure()
    model_colors = {"SVR":NAVY, "XGBoost":COPPER, "MLP":STEEL}
    for mdl in ["SVR","XGBoost","MLP"]:
        vals = [A[mdl].get(a) for a in aug_labels]
        fig.add_trace(go.Bar(name=mdl, x=aug_labels, y=vals, marker_color=model_colors[mdl]))
    fig.update_layout(barmode="group", height=380, margin=dict(t=20,b=20),
        yaxis_title="MAPE (%)", legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    # 개선폭 요약
    st.markdown("### 모델별 최적 증강 기법")
    cols = st.columns(3)
    for col, mdl in zip(cols, ["SVR","XGBoost","MLP"]):
        base = A[mdl]["증강 없음"]
        best_aug = min([a for a in aug_labels if a!="증강 없음"], key=lambda a: A[mdl][a])
        best_val = A[mdl][best_aug]
        improve = base - best_val
        color = GREEN if improve>0 else RED
        with col:
            st.markdown(f"""
<div style="background:#F4F6F9;border-radius:10px;padding:16px;border-top:4px solid {color}">
<div style="font-size:16px;font-weight:bold;color:{NAVY}">{mdl}</div>
<div style="font-size:12px;color:#5B7A99;margin:4px 0">증강 없음 {base:.2f}% → {best_aug} {best_val:.2f}%</div>
<div style="font-size:20px;font-weight:bold;color:{color}">{'▼' if improve>0 else '▲'} {abs(improve):.2f}%p</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="warn">
💡 <b>연구 발견</b><br>
• <b>분포기반 증강</b>이 가장 효과적 (XGBoost 8.23%→5.82%). 변수 간 공분산을
  유지하며 샘플을 생성해, 변수들이 함께 움직이는 원자재 데이터에 적합합니다.<br>
• 지터링·부트스트랩은 모델에 따라 오히려 악화 → <b>증강 기법도 데이터·모델에 맞게 선택</b>해야 합니다.<br>
• 이는 본 연구의 일관된 주제("방법에 절대 강자는 없다")와 맞닿아 있습니다.
</div>
""", unsafe_allow_html=True)
