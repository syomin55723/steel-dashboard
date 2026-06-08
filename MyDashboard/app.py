import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64, os
from theme import THEME_CSS, render_landing, show_loading, CHART_THEME

st.set_page_config(page_title="AegisCore", layout="wide", page_icon="👁️", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── 全域圖表色盤 ──────────────────────────────────────────
CHART_BG      = "#ffffff"
CHART_GRID    = "#e2e8f0"
CHART_TEXT    = "#1e293b"
CHART_AXIS    = "#cbd5e1"
CHART_AVG     = "#10b981"   # 翠綠：平均線
CHART_UCL     = "#ef4444"   # 紅：管制上下限
CHART_NORMAL  = "rgba(96,165,250,0.65)"   # 藍：直方圖正常範圍
CHART_OUTLIER = "rgba(239,68,68,0.70)"    # 紅：規格外
CHART_CURVE   = "#0ea5e9"   # 天藍：常態曲線

# ══════════════════════════════════════════════════════
#  資料讀取
# ══════════════════════════════════════════════════════
@st.cache_data
def load_and_clean_data(file_bytes: bytes, file_name: str):
    import io
    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='big5')
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    df.columns = df.columns.astype(str).str.replace(r'\s+', '', regex=True).str.upper()

    rename_mapping = [
        (['鋼捲號碼', 'COIL_NO'],    '產出鋼捲號碼'),
        (['PRODUCTION_DATE'],         '生產日期'),
        (['QUALITY_CLASS'],           '試驗等級'),
        (['BASE_METAL_THICK'],        '訂單厚度'),
        (['REAL_WIDTH'],              '訂單寬度'),
    ]
    rename_dict = {}
    for source_cols, target_name in rename_mapping:
        for col in source_cols:
            if col in df.columns:
                rename_dict[col] = target_name
                break
    df.rename(columns=rename_dict, inplace=True)

    if df.columns.duplicated().any():
        for col in df.columns[df.columns.duplicated(keep=False)].unique():
            indices = [i for i, x in enumerate(df.columns) if x == col]
            for idx in indices[1:]:
                df = df.drop(df.columns[idx], axis=1)

    xray_sets = [
        ['XRAY_A_T_N','XRAY_A_T_C','XRAY_A_T_S','XRAY_A_B_N','XRAY_A_B_C','XRAY_A_B_S'],
        ['NORTH_TOP_COAT_WEIGHT','CENTER_TOP_COAT_WEIGHT','SOUTH_TOP_COAT_WEIGHT',
         'NORTH_BACK_COAT_WEIGHT','CENTER_BACK_COAT_WEIGHT','SOUTH_BACK_COAT_WEIGHT'],
        ['北正面鍍層','中正面鍍層','南正面鍍層','北背面鍍層','中背面鍍層','南背面鍍層'],
    ]
    best_set, max_valid = None, -1
    for cols in xray_sets:
        if all(c in df.columns for c in cols):
            v = df[cols].apply(pd.to_numeric, errors='coerce').notna().sum().sum()
            if v > max_valid:
                max_valid, best_set = v, cols
    if best_set and max_valid > 0:
        for c in best_set:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['雙面總鍍層量(AVG)'] = (
            (df[best_set[0]] + df[best_set[1]] + df[best_set[2]]) / 3 +
            (df[best_set[3]] + df[best_set[4]] + df[best_set[5]]) / 3
        )

    if '生產日期' in df.columns:
        def _ym(v):
            try:
                dt = pd.to_datetime(v)
                return f"{dt.year}年{dt.month:02d}月"
            except Exception:
                return str(v)
        df['生產年月'] = df['生產日期'].apply(_ym)
    else:
        df['生產年月'] = '全區間'

    return df


# ══════════════════════════════════════════════════════
#  側邊欄
# ══════════════════════════════════════════════════════
_logo_candidates = ["logo_zheng.svg", "MyDashboard/logo_zheng.svg"]
_logo_path = next((p for p in _logo_candidates if os.path.exists(p)), None)
if _logo_path:
    try:
        with open(_logo_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        mime = "image/svg+xml" if _logo_path.endswith(".svg") else "image/png"
        st.markdown(f"""
<style>
.glowing-logo {{
    position:fixed; bottom:80px; right:28px; width:64px; z-index:9999;
    opacity:0.60; filter:drop-shadow(0 2px 10px rgba(14,165,233,0.35));
    transition:all 0.3s ease; cursor:pointer;
}}
.glowing-logo:hover {{ opacity:1; transform:scale(1.12) translateY(-4px); }}
</style>
<img src="data:{mime};base64,{img_b64}" class="glowing-logo">
""", unsafe_allow_html=True)
    except Exception:
        pass

with st.sidebar:
    st.header("⚙️ 儀表板控制中心")
    uploaded_file = st.file_uploader("📂 上傳 RAW DATA", type=["xlsx", "csv"])
    st.markdown("---")
    if uploaded_file:
        st.success("✅ 文件已加載")
        st.caption(f"文件名：{uploaded_file.name}")


# ══════════════════════════════════════════════════════
#  未上傳：封面頁
# ══════════════════════════════════════════════════════
if uploaded_file is None:
    render_landing()
    st.stop()


# ══════════════════════════════════════════════════════
#  已上傳：資料處理
# ══════════════════════════════════════════════════════
file_id = uploaded_file.name + str(uploaded_file.size)

if st.session_state.get("loaded_file_id") != file_id:
    show_loading()
    st.session_state["raw_df"] = load_and_clean_data(
        uploaded_file.read(), uploaded_file.name
    )
    st.session_state["loaded_file_id"] = file_id
    st.rerun()

raw_df = st.session_state["raw_df"]
df = raw_df.copy()

if '試驗等級' in df.columns:
    df = df.dropna(subset=['試驗等級'])
    df['試驗等級'] = df['試驗等級'].astype(str).str.strip()
    df = df[df['試驗等級'] != '']
    df = df[~df['試驗等級'].str.lower().isin(['nan','null','none','na'])]
    df["比對群組"] = df["生產年月"] + " - " + df["試驗等級"]
else:
    df["比對群組"] = "全批次數據"

# ── 側邊欄篩選器 ─────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:8px;">
      <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">
        🎯 篩選器
      </div>
      <div style="font-size:13px;color:#64748b;">💡 條件即時連動，支援跨月多選</div>
    </div>
    """, unsafe_allow_html=True)

    file_key = uploaded_file.name

    def cascading_filter(col, cur_df, label):
        if col not in cur_df.columns:
            return []
        opts = sorted(cur_df[col].dropna().astype(str).unique())
        if not opts:
            return []
        k = f"filter_{file_key}_{col}"
        if k in st.session_state:
            st.session_state[k] = [x for x in st.session_state[k] if x in opts]
        # label 用 markdown 顯示，multiselect 本身隱藏 label
        st.markdown(
            f'<div style="font-size:14px;font-weight:600;color:#1e293b;'
            f'margin-bottom:4px;margin-top:8px;">{label}</div>',
            unsafe_allow_html=True
        )
        return st.multiselect("", options=opts, key=k,
                              placeholder="ALL",
                              label_visibility="collapsed")

    f_month  = cascading_filter('生產年月',      df,    "🗓️ 生產年月")
    df_f1 = df[df['生產年月'].astype(str).isin(f_month)] if f_month else df.copy()

    f_order  = cascading_filter('訂單號碼',      df_f1, "📝 訂單號碼")
    df_f2 = df_f1[df_f1['訂單號碼'].astype(str).isin(f_order)] if f_order else df_f1.copy()

    f_thick  = cascading_filter('訂單厚度',      df_f2, "📏 訂單厚度")
    df_f3 = df_f2[df_f2['訂單厚度'].astype(str).isin(f_thick)] if f_thick else df_f2.copy()

    f_width  = cascading_filter('訂單寬度',      df_f3, "↔️ 訂單寬度")
    df_f4 = df_f3[df_f3['訂單寬度'].astype(str).isin(f_width)] if f_width else df_f3.copy()

    f_mat    = cascading_filter('熱軋材質',      df_f4, "🪨 熱軋材質")
    df_f5 = df_f4[df_f4['熱軋材質'].astype(str).isin(f_mat)] if f_mat else df_f4.copy()

    f_spec   = cascading_filter('產品規格代碼',  df_f5, "📋 產品規格代碼")
    df_f6 = df_f5[df_f5['產品規格代碼'].astype(str).isin(f_spec)] if f_spec else df_f5.copy()

    f_rthick = cascading_filter('原料厚度',      df_f6, "🔲 原料厚度")
    df_f6b = df_f6[df_f6['原料厚度'].astype(str).isin(f_rthick)] if f_rthick else df_f6.copy()

    f_maker  = cascading_filter('原料製造廠商',  df_f6b, "🏭 原料製造廠商")
    df_f7 = df_f6b[df_f6b['原料製造廠商'].astype(str).isin(f_maker)] if f_maker else df_f6b.copy()

    f_plate  = cascading_filter('取板位置',      df_f7, "📍 取板位置")
    df_f8 = df_f7[df_f7['取板位置'].astype(str).isin(f_plate)] if f_plate else df_f7.copy()

    f_coat   = cascading_filter('上鍍層',        df_f8, "🔩 上鍍層")
    df_f9 = df_f8[df_f8['上鍍層'].astype(str).isin(f_coat)] if f_coat else df_f8.copy()

    f_usage  = cascading_filter('用途中文說明',  df_f9, "📌 用途中文說明")
    filtered_df = df_f9[df_f9['用途中文說明'].astype(str).isin(f_usage)] if f_usage else df_f9.copy()

if filtered_df.empty:
    st.warning("⚠️ 目前篩選條件下沒有找到任何數據，請放寬左側的篩選條件！")
    st.stop()

# ── 可分析參數清單 ────────────────────────────────────
_EXCLUDE = {
    '產出鋼捲號碼','試驗等級','投入等級','生產日期','比對群組','生產年月','SHIFT_NO',
    '鍍層下限管制值','北正面鍍層','中正面鍍層','南正面鍍層','北背面鍍層','中背面鍍層','南背面鍍層',
    'XRAY_A_T_N','XRAY_A_T_C','XRAY_A_T_S','XRAY_A_B_N','XRAY_A_B_C','XRAY_A_B_S',
    'NORTH_TOP_COAT_WEIGHT','CENTER_TOP_COAT_WEIGHT','SOUTH_TOP_COAT_WEIGHT',
    'NORTH_BACK_COAT_WEIGHT','CENTER_BACK_COAT_WEIGHT','SOUTH_BACK_COAT_WEIGHT',
    '訂單厚度','訂單寬度','原料厚度','原料寬度','投入厚度','投入寬度',
    '投入重量','實測重量','實測厚度','實測寬度','實測長度',
    '開始時間','排程單號','結束時間','班次','產出內徑','上粗糙度','下粗糙度',
    '化成','切除米數','收捲方向','AIM符號','引帶號碼',
}
numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
available_params = [c for c in numeric_cols if c not in _EXCLUDE]
if '雙面總鍍層量(AVG)' in available_params:
    available_params = ['雙面總鍍層量(AVG)'] + [x for x in available_params if x != '雙面總鍍層量(AVG)']

if not available_params:
    st.warning("⚠️ 找不到可分析的數值欄位，請確認上傳的檔案內容。")
    st.stop()


# ══════════════════════════════════════════════════════
#  參數選擇
# ══════════════════════════════════════════════════════
selected_param = st.selectbox(
    "🔍 選擇分析參數",
    available_params,
    key=f"param_{file_key}",
    label_visibility="collapsed"
)

plot_df = filtered_df.dropna(subset=[selected_param])
if plot_df.empty:
    st.warning(f"⚠️ 篩選後的鋼捲中，沒有任何一顆擁有【{selected_param}】的有效數據！")
    st.stop()

avg_val    = float(plot_df[selected_param].mean())
std_val    = float(plot_df[selected_param].std())
median_val = float(plot_df[selected_param].median())

is_7b = plot_df['試驗等級'].astype(str).str.upper().str.replace(' ','').str.contains('7B', na=False) \
        if '試驗等級' in plot_df.columns else pd.Series([False]*len(plot_df), index=plot_df.index)
abnormal_count = int(is_7b.sum())
yield_rate = (len(plot_df) - abnormal_count) / len(plot_df) * 100 if len(plot_df) > 0 else 100.0
months_count = plot_df['生產年月'].nunique() if '生產年月' in plot_df.columns else 0

# ── 頂部資訊列 ─────────────────────────────────────
st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
    padding:16px 24px;margin-bottom:14px;
    display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:10px;height:10px;border-radius:50%;background:#10b981;
        box-shadow:0 0 0 3px #d1fae5;flex-shrink:0;"></div>
    <div>
      <div style="font-size:17px;font-weight:700;color:#0f172a;">{uploaded_file.name}</div>
      <div style="font-size:14px;color:#94a3b8;margin-top:2px;">共 {len(raw_df):,} 筆原始資料</div>
    </div>
  </div>
  <div style="display:flex;gap:28px;align-items:center;">
    <div style="text-align:center;">
      <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
          letter-spacing:.8px;margin-bottom:4px;">月份涵蓋</div>
      <div style="font-size:20px;font-weight:800;color:#0ea5e9;">
          {df['生產年月'].nunique() if '生產年月' in df.columns else '—'} 個月</div>
    </div>
    <div style="width:1px;height:36px;background:#e2e8f0;"></div>
    <div style="text-align:center;">
      <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
          letter-spacing:.8px;margin-bottom:4px;">可分析參數</div>
      <div style="font-size:20px;font-weight:800;color:#0ea5e9;">{len(available_params)} 項</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 5 指標卡片（統一字體）───────────────────────────
abnormal_pct = abnormal_count / len(plot_df) * 100 if len(plot_df) > 0 else 0
yield_bar    = min(yield_rate, 100)

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;
    padding:16px 18px;border-left:4px solid #0ea5e9;">
  <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
      letter-spacing:.8px;margin-bottom:8px;">總鋼捲數</div>
  <div style="font-size:26px;font-weight:800;color:#0f172a;line-height:1.1;">{len(plot_df):,}</div>
  <div style="font-size:14px;color:#64748b;margin-top:6px;">顆</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;
    padding:16px 18px;border-left:4px solid #0ea5e9;">
  <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
      letter-spacing:.8px;margin-bottom:8px;">平均值</div>
  <div style="font-size:26px;font-weight:800;color:#0f172a;line-height:1.1;">{avg_val:.3f}</div>
  <div style="font-size:14px;color:#64748b;margin-top:6px;">{selected_param}</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
    padding:16px 18px;border-left:4px solid #10b981;">
  <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
      letter-spacing:.8px;margin-bottom:8px;">良品率</div>
  <div style="font-size:26px;font-weight:800;color:#10b981;line-height:1.1;">{yield_rate:.1f}%</div>
  <div style="font-size:14px;color:#10b981;font-weight:600;margin-top:6px;">{"✓ 全數合格" if yield_rate == 100 else f"良品 {len(plot_df) - abnormal_count:,} 顆"}</div>
</div>""", unsafe_allow_html=True)

_ab_bg     = "#fffafa" if abnormal_count > 0 else "#fff"
_ab_border = "#fecaca" if abnormal_count > 0 else "#e2e8f0"
_ab_left   = "#ef4444" if abnormal_count > 0 else "#0ea5e9"
_ab_numclr = "#ef4444" if abnormal_count > 0 else "#0f172a"
_ab_sub    = f'↑ {abnormal_pct:.1f}%' if abnormal_count > 0 else "✓ 無異常"
_ab_subclr = "#ef4444" if abnormal_count > 0 else "#10b981"

c4.markdown(f"""
<div style="background:{_ab_bg};border:1px solid {_ab_border};border-radius:12px;
    padding:16px 18px;border-left:4px solid {_ab_left};">
  <div style="font-size:12px;color:#94a3b8;font-weight:700;text-transform:uppercase;
      letter-spacing:.8px;margin-bottom:8px;">異常品數量</div>
  <div style="font-size:26px;font-weight:800;color:{_ab_numclr};line-height:1.1;">{abnormal_count:,}</div>
  <div style="font-size:14px;color:{_ab_subclr};font-weight:600;margin-top:6px;">{_ab_sub}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════
#  主分頁
# ══════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊 數據總覽 & 趨勢分析", "📐 製程能力分析 (Ca · Cp · Cpk)"])


# ────────────────────────────────────────────────────
# TAB 1：趨勢 + 月份高亮
# ────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <div style="width:4px;height:22px;background:#0ea5e9;border-radius:2px;"></div>
        <span style="font-size:20px;font-weight:700;color:#0f172a;">生產順序異常監控圖</span>
        <span style="font-size:13px;color:#64748b;margin-left:4px;">SPC ±3σ 管制界限</span>
    </div>
    """, unsafe_allow_html=True)

    months_list = sorted(plot_df['生產年月'].unique().tolist(), key=str) \
                  if '生產年月' in plot_df.columns else []

    hl_key = f"hl_month_{file_key}_{selected_param}"
    if hl_key not in st.session_state:
        st.session_state[hl_key] = months_list[-1] if months_list else "全部"

    if months_list:
        btn_cols = st.columns(len(months_list) + 1)
        for i, m in enumerate(months_list):
            is_active = st.session_state[hl_key] == m
            if btn_cols[i].button(m, key=f"hl_btn_{file_key}_{selected_param}_{m}",
                                   type="primary" if is_active else "secondary"):
                st.session_state[hl_key] = m
                st.rerun()
        is_all = st.session_state[hl_key] == "全部"
        if btn_cols[-1].button("全部", key=f"hl_btn_all_{file_key}_{selected_param}",
                                type="primary" if is_all else "secondary"):
            st.session_state[hl_key] = "全部"
            st.rerun()

    selected_month = st.session_state.get(hl_key, "全部")
    x_col = "產出鋼捲號碼" if "產出鋼捲號碼" in plot_df.columns else None

    fig_line = go.Figure()
    month_palette = px.colors.qualitative.Bold

    # ── 月份色帶背景 ────────────────────────────────
    if months_list and x_col:
        band_colors = ["rgba(186,230,253,0.65)", "rgba(255,255,255,0)"]
        for i, month in enumerate(months_list):
            m_df = plot_df[plot_df['生產年月'] == month]
            if m_df.empty:
                continue
            x_vals = m_df[x_col].tolist()
            x0, x1 = x_vals[0], x_vals[-1]
            fig_line.add_vrect(
                x0=x0, x1=x1,
                fillcolor=band_colors[i % 2],
                line_width=0, layer="below"
            )
            # 月份分隔虛線（除第一個月）
            if i > 0:
                fig_line.add_vline(
                    x=x0, line_color="#cbd5e1",
                    line_width=1, line_dash="dot"
                )
            # 月份標籤（頂部）
            is_active = (selected_month == month or selected_month == "全部")
            fig_line.add_annotation(
                x=x_vals[len(x_vals)//2],
                y=1.0, yref="paper",
                text=f"<b>{month}</b>" if is_active and selected_month == month else month,
                font=dict(
                    color="#0ea5e9" if selected_month == month else "#94a3b8",
                    size=12,
                    weight=700 if selected_month == month else 400
                ),
                showarrow=False, yanchor="bottom", xanchor="center"
            )

    for i, month in enumerate(months_list if months_list else ["全部"]):
        m_df = plot_df[plot_df['生產年月'] == month] if months_list else plot_df
        if m_df.empty:
            continue
        x_data = m_df[x_col] if x_col else m_df.index
        is_highlighted = (selected_month == "全部") or (selected_month == month)
        opacity = 1.0 if is_highlighted else 0.12
        color   = month_palette[i % len(month_palette)]

        # hover 加入日期資訊
        if '生產日期' in m_df.columns:
            custom = m_df['生產日期'].astype(str).tolist()
            hover = "<b>鋼捲號碼：%{x}</b><br>數值：%{y:.3f}<br>日期：%{customdata}<extra></extra>"
        else:
            custom = None
            hover = "<b>鋼捲號碼：%{x}</b><br>數值：%{y:.3f}<extra></extra>"

        fig_line.add_trace(go.Scatter(
            x=x_data, y=m_df[selected_param],
            mode='lines+markers', name=month,
            line=dict(color=color, width=2.5 if is_highlighted else 1),
            marker=dict(size=7 if is_highlighted else 4, color=color, opacity=opacity),
            opacity=opacity, connectgaps=True,
            customdata=custom,
            hovertemplate=hover
        ))

    if '試驗等級' in plot_df.columns:
        ab_df = plot_df[is_7b]
        if not ab_df.empty:
            x_ab = ab_df[x_col] if x_col else ab_df.index
            if '生產日期' in ab_df.columns:
                ab_custom = ab_df['生產日期'].astype(str).tolist()
                ab_hover = "<b>鋼捲號碼：%{x}</b><br>7B 異常：%{y:.3f}<br>日期：%{customdata}<extra></extra>"
            else:
                ab_custom = None
                ab_hover = "<b>鋼捲號碼：%{x}</b><br>7B 異常：%{y:.3f}<extra></extra>"
            fig_line.add_trace(go.Scatter(
                x=x_ab, y=ab_df[selected_param],
                mode='markers', name='異常 (7B)',
                marker=dict(color='#FFD700', size=14, symbol='circle',
                            line=dict(color='#1e293b', width=1.5)),
                customdata=ab_custom,
                hovertemplate=ab_hover
            ))

    ucl = avg_val + 3 * std_val
    lcl = avg_val - 3 * std_val

    # 管制帶背景
    fig_line.add_hrect(y0=lcl, y1=ucl, fillcolor="rgba(14,165,233,0.04)",
                       line_width=0)

    fig_line.add_hline(y=avg_val, line_dash="dash", line_color=CHART_AVG, line_width=1.8,
                       annotation_text=f"均值 {avg_val:.3f}", annotation_position="bottom right",
                       annotation_font=dict(color=CHART_AVG, size=13))
    fig_line.add_hline(y=ucl, line_dash="dot", line_color=CHART_UCL, line_width=1.5,
                       annotation_text=f"+3σ  {ucl:.3f}", annotation_position="top right",
                       annotation_font=dict(color=CHART_UCL, size=13))
    fig_line.add_hline(y=lcl, line_dash="dot", line_color=CHART_UCL, line_width=1.5,
                       annotation_text=f"−3σ  {lcl:.3f}", annotation_position="bottom right",
                       annotation_font=dict(color=CHART_UCL, size=13))

    fig_line.update_xaxes(showticklabels=False, title_text="生產順序（依照時間 / 鋼捲號碼）",
                          title_font=dict(size=14))
    fig_line.update_layout(
        template="simple_white",
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        title=dict(text=f"【{selected_param}】 SPC 趨勢管制圖",
                   font=dict(color=CHART_TEXT, size=17), x=0),
        height=500, hovermode="closest",
        font=dict(color=CHART_TEXT, size=14),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            title=dict(text="生產順序（依照時間 / 鋼捲號碼）",
                       font=dict(color="#64748b", size=14))
        ),
        yaxis=dict(gridcolor=CHART_GRID, tickfont=dict(color=CHART_TEXT, size=14),
                   linecolor=CHART_AXIS, showgrid=True),
        legend=dict(bgcolor=CHART_BG, bordercolor=CHART_GRID, borderwidth=1,
                    font=dict(color=CHART_TEXT, size=13),
                    orientation="h", yanchor="top", y=-0.12, xanchor="right", x=1.0),
        margin=dict(t=50, b=80, l=60, r=80)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    if abnormal_count > 0:
        st.warning(f"⚠️ 趨勢圖中共標示了 **{abnormal_count} 顆** 7B 異常鋼捲（黃色點），請重點追蹤。")
    else:
        st.success("✅ 目前顯示的鋼捲中，沒有出現 7B 等級。")

    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <div style="width:4px;height:22px;background:#0ea5e9;border-radius:2px;"></div>
        <span style="font-size:20px;font-weight:700;color:#0f172a;">群組數據分佈箱型圖</span>
        <span style="font-size:13px;color:#64748b;margin-left:4px;">月份 × 等級 分群比對</span>
    </div>
    """, unsafe_allow_html=True)

    group_palette = {g: month_palette[i % len(month_palette)]
                     for i, g in enumerate(plot_df['比對群組'].unique())}
    fig_box = px.box(
        plot_df, x="比對群組", y=selected_param, color="比對群組",
        color_discrete_map=group_palette,
        title=f"【{selected_param}】 群組箱型圖對比",
        points="all", template="simple_white"
    )
    fig_box.add_hline(y=avg_val, line_dash="dash", line_color=CHART_AVG, line_width=1.8,
                      annotation_text=f"均值: {avg_val:.3f}",
                      annotation_font=dict(color=CHART_AVG, size=13))
    fig_box.update_layout(
        template="simple_white",
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        height=500, showlegend=False,
        font=dict(color=CHART_TEXT, size=14),
        title=dict(font=dict(color=CHART_TEXT, size=17)),
        xaxis=dict(title=dict(text="群組分類", font=dict(color="#64748b", size=14)),
                   gridcolor=CHART_GRID, tickfont=dict(color=CHART_TEXT, size=14),
                   linecolor=CHART_AXIS),
        yaxis=dict(gridcolor=CHART_GRID, tickfont=dict(color=CHART_TEXT, size=14),
                   linecolor=CHART_AXIS),
        margin=dict(t=60, b=60, l=60, r=30)
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <div style="width:4px;height:22px;background:#0ea5e9;border-radius:2px;"></div>
        <span style="font-size:20px;font-weight:700;color:#0f172a;">數據匯出</span>
    </div>
    """, unsafe_allow_html=True)
    st.download_button(
        "📥 下載目前篩選數據 (CSV)",
        data=filtered_df.to_csv(index=False).encode('utf-8-sig'),
        file_name='AegisCore_品質分析資料.csv',
        mime='text/csv'
    )


# ────────────────────────────────────────────────────
# TAB 2：製程能力分析（60/40 排版）
# ────────────────────────────────────────────────────
with tab2:

    spc_data = plot_df[selected_param].dropna()
    if len(spc_data) < 2:
        st.warning("⚠️ 數據不足，請放寬篩選條件。")
        st.stop()

    spc_n      = len(spc_data)
    spc_mean   = float(spc_data.mean())
    spc_median = float(spc_data.median())
    spc_std    = float(spc_data.std())
    spc_min    = float(spc_data.min())
    spc_max    = float(spc_data.max())
    spc_cv     = spc_std / spc_mean * 100 if spc_mean != 0 else 0

    # ── 設定區：兩張卡片並排 ────────────────────────
    set_col1, set_col2 = st.columns(2)

    with set_col1:
        st.markdown("""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin-bottom:2px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
          <div style="width:3px;height:18px;background:#0ea5e9;"></div>
          <span style="font-size:15px;font-weight:700;color:#0f172a;letter-spacing:.5px;">顯示設定</span>
        </div>
        """, unsafe_allow_html=True)

        spec_type = st.selectbox(
            "📐 規格類型",
            ["雙邊 (LSL & USL)", "單邊上限 (USL only)", "單邊下限 (LSL only)"],
            key=f"spc_spectype_{file_key}"
        )

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("<div style='font-size:15px;font-weight:600;color:#475569;margin-bottom:3px;'>組距 Bins</div>", unsafe_allow_html=True)
            spc_bins = st.slider("", min_value=5, max_value=30, value=12,
                                 key=f"spc_bins_{selected_param}", label_visibility="collapsed")
        with b2:
            st.markdown("<div style='font-size:14px;font-weight:600;color:#475569;margin-bottom:3px;'>小數位數</div>", unsafe_allow_html=True)
            spc_prec = st.slider("", min_value=0, max_value=6, value=3,
                                 key=f"spc_prec_{selected_param}", label_visibility="collapsed")

        st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;'>", unsafe_allow_html=True)
        tg1, tg2 = st.columns(2)
        tg3, tg4 = st.columns(2)
        show_mean2   = tg1.toggle("平均值線", value=True,  key=f"spc_mean_{selected_param}")
        show_curve2  = tg2.toggle("常態曲線", value=True,  key=f"spc_curve_{selected_param}")
        show_median2 = tg3.toggle("中位數線", value=False, key=f"spc_med_{selected_param}")
        show_target2 = tg4.toggle("目標值線", value=True,  key=f"spc_tgt_{selected_param}")
        st.markdown("</div>", unsafe_allow_html=True)

    with set_col2:
        st.markdown("""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin-bottom:2px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
          <div style="width:3px;height:18px;background:#ef4444;"></div>
          <span style="font-size:15px;font-weight:700;color:#0f172a;letter-spacing:.5px;">規格設定</span>
        </div>
        """, unsafe_allow_html=True)

        is_both  = "雙邊" in spec_type
        is_upper = "上限" in spec_type
        is_lower = "下限" in spec_type

        # LSL
        st.markdown("""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="font-size:15px;font-weight:700;color:#1e293b;">LSL　規格下限</span>
          <span style="font-size:12px;background:#fee2e2;color:#991b1b;border-radius:4px;padding:2px 10px;font-weight:700;">下限</span>
        </div>""", unsafe_allow_html=True)
        lsl2 = st.number_input("", value=float(spc_mean - 4*spc_std),
            key=f"spc_lsl_{selected_param}", disabled=is_upper,
            format="%.3f", label_visibility="collapsed")

        st.markdown("<div style='margin-top:4px;'>", unsafe_allow_html=True)
        st.markdown("""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="font-size:15px;font-weight:700;color:#1e293b;">USL　規格上限</span>
          <span style="font-size:12px;background:#fee2e2;color:#991b1b;border-radius:4px;padding:2px 10px;font-weight:700;">上限</span>
        </div>""", unsafe_allow_html=True)
        usl2 = st.number_input("", value=float(spc_mean + 4*spc_std),
            key=f"spc_usl_{selected_param}", disabled=is_lower,
            format="%.3f", label_visibility="collapsed")

        st.markdown("<div style='margin-top:4px;'>", unsafe_allow_html=True)
        st.markdown("""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="font-size:15px;font-weight:700;color:#1e293b;">目標值</span>
          <span style="font-size:12px;background:#ede9fe;color:#5b21b6;border-radius:4px;padding:2px 10px;font-weight:700;">目標</span>
        </div>""", unsafe_allow_html=True)
        target2 = st.number_input("", value=float(spc_mean),
            key=f"spc_target_{selected_param}",
            format="%.3f", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

        # ── 計算 ─────────────────────────────────────────
    if is_both and (usl2 - lsl2) != 0:
        spec_mid = (usl2 + lsl2) / 2          # 規格中心，與目標值完全無關
        ca2 = (spc_mean - spec_mid) / ((usl2 - lsl2) / 2) * 100
    else:
        ca2 = None

    if spc_std > 0:
        if is_both:    cp2 = (usl2 - lsl2) / (6 * spc_std)
        elif is_upper: cp2 = (usl2 - spc_mean) / (3 * spc_std)
        else:          cp2 = (spc_mean - lsl2) / (3 * spc_std)
    else:
        cp2 = 0.0

    cpk2 = cp2 * (1 - abs(ca2) / 100) if (is_both and ca2 is not None) else cp2

    out_usl2 = int((spc_data > usl2).sum()) if not is_lower else 0
    out_lsl2 = int((spc_data < lsl2).sum()) if not is_upper else 0
    in2      = spc_n - out_usl2 - out_lsl2
    yield2   = in2 / spc_n * 100

    def _grade_ca(v):
        if v is None: return "—", "#64748b", "需雙邊規格"
        a = abs(v)
        if a < 6.25:  return "A+", "#059669", "準確度極佳"
        if a < 12.5:  return "A",  "#10b981", "準確度良好"
        if a < 25.0:  return "B",  "#f59e0b", "建議調整 Offset"
        if a < 50.0:  return "C",  "#f97316", "能力不足"
        return             "D",  "#ef4444", "能力極差"

    def _grade_cp(v):
        if v >= 1.67: return "A+", "#059669", "精密度極佳"
        if v >= 1.33: return "A",  "#10b981", "精密度良好"
        if v >= 1.00: return "B",  "#f59e0b", "尚可，加強管制"
        if v >= 0.67: return "C",  "#f97316", "能力不足"
        return             "D",  "#ef4444", "能力極差"

    ca_g, ca_c, ca_d   = _grade_ca(ca2)
    cp_g, cp_c, cp_d   = _grade_cp(cp2)
    cpk_g, cpk_c, cpk_d = _grade_cp(cpk2)

    def _light_bg(color):
        """根據等級顏色回傳淡色背景"""
        m = {
            "#059669": "#d1fae5",  # A+ 淡綠
            "#10b981": "#d1fae5",  # A  淡綠
            "#f59e0b": "#fef9c3",  # B  淡黃
            "#f97316": "#ffedd5",  # C  淡橙
            "#ef4444": "#fee2e2",  # D  淡紅
            "#64748b": "#f1f5f9",  # N/A 淡灰
        }
        return m.get(color, "#f8fafc")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1：統計摘要 6 欄（全寬）──────────────────
    p = int(spc_prec)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);background:#fff;
        border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;margin-bottom:14px;">
      <div style="padding:14px 16px;text-align:center;border-right:1px solid #e2e8f0;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">樣本數 (N)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_n:,}</div>
      </div>
      <div style="padding:14px 16px;text-align:center;border-right:1px solid #e2e8f0;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">平均值 (MEAN)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_mean:.{p}f}</div>
      </div>
      <div style="padding:14px 16px;text-align:center;border-right:1px solid #e2e8f0;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">中位數 (MED)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_median:.{p}f}</div>
      </div>
      <div style="padding:14px 16px;text-align:center;border-right:1px solid #e2e8f0;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">標準差 (STD)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_std:.{p}f}</div>
      </div>
      <div style="padding:14px 16px;text-align:center;border-right:1px solid #e2e8f0;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">最小值 (MIN)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_min:.{p}f}</div>
      </div>
      <div style="padding:14px 16px;text-align:center;">
        <div style="font-size:14px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">最大值 (MAX)</div>
        <div style="font-size:24px;font-weight:700;color:#0f172a;">{spc_max:.{p}f}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Row 2：3 大卡片 Ca / Cp / Cpk（全寬）─────────
    _ca_bg  = _light_bg(ca_c)
    _cp_bg  = _light_bg(cp_c)
    _cpk_bg = _light_bg(cpk_c)

    k2, k3, k4 = st.columns(3)

    k2.markdown(f"""
    <div style="background:{_ca_bg};border:1px solid {ca_c}40;border-radius:14px;
        padding:22px 24px;border-top:5px solid {ca_c};text-align:center;">
      <div style="font-size:15px;font-weight:700;color:{ca_c};letter-spacing:1px;
          text-transform:uppercase;margin-bottom:10px;">Ca（準確度）</div>
      <div style="font-size:42px;font-weight:800;color:{ca_c};line-height:1.1;margin-bottom:10px;">
          {("N/A" if ca2 is None else f"{abs(ca2):.2f}%")}</div>
      <div style="display:inline-block;font-size:14px;font-weight:700;color:#fff;
          background:{ca_c};border-radius:20px;padding:4px 18px;">{ca_g}　{ca_d}</div>
    </div>""", unsafe_allow_html=True)

    k3.markdown(f"""
    <div style="background:{_cp_bg};border:1px solid {cp_c}40;border-radius:14px;
        padding:22px 24px;border-top:5px solid {cp_c};text-align:center;">
      <div style="font-size:15px;font-weight:700;color:{cp_c};letter-spacing:1px;
          text-transform:uppercase;margin-bottom:10px;">Cp（精密度）</div>
      <div style="font-size:42px;font-weight:800;color:{cp_c};line-height:1.1;margin-bottom:10px;">{cp2:.3f}</div>
      <div style="display:inline-block;font-size:14px;font-weight:700;color:#fff;
          background:{cp_c};border-radius:20px;padding:4px 18px;">{cp_g}　{cp_d}</div>
    </div>""", unsafe_allow_html=True)

    k4.markdown(f"""
    <div style="background:{_cpk_bg};border:1px solid {cpk_c}40;border-radius:14px;
        padding:22px 24px;border-top:5px solid {cpk_c};text-align:center;">
      <div style="font-size:15px;font-weight:700;color:{cpk_c};letter-spacing:1px;
          text-transform:uppercase;margin-bottom:10px;">Cpk（製程能力）</div>
      <div style="font-size:42px;font-weight:800;color:{cpk_c};line-height:1.1;margin-bottom:10px;">{cpk2:.3f}</div>
      <div style="display:inline-block;font-size:14px;font-weight:700;color:#fff;
          background:{cpk_c};border-radius:20px;padding:4px 18px;">{cpk_g}　{cpk_d}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 3：60/40 主圖表區 ────────────────────────
    col_main, col_side = st.columns([1.5, 1])

    with col_main:
        # 直方圖（主視覺 60%）
        arr    = spc_data.values
        bins   = int(spc_bins)
        p      = int(spc_prec)
        pad    = spc_std * 1.5
        amin   = min(arr.min(), lsl2) - pad
        amax   = max(arr.max(), usl2) + pad
        step_h = (amax - amin) / bins
        edges  = [amin + i * step_h for i in range(bins + 1)]
        counts = [0] * bins
        for v in arr:
            idx = max(0, min(bins-1, int((v - amin) / step_h)))
            counts[idx] += 1

        # 柱子顏色（加深 + 外框）
        bar_colors, bar_borders = [], []
        for i in range(bins):
            out = False
            if is_both or is_lower: out = out or (edges[i+1] <= lsl2)
            if is_both or is_upper: out = out or (edges[i]   >= usl2)
            if out:
                bar_colors.append("rgba(239,68,68,0.80)")
                bar_borders.append("#dc2626")
            else:
                bar_colors.append("rgba(59,130,246,0.75)")
                bar_borders.append("#2563eb")

        bar_x = [(edges[i]+edges[i+1])/2 for i in range(bins)]
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=bar_x, y=counts, width=[step_h*0.92]*bins,
            marker=dict(
                color=bar_colors,
                line=dict(width=1.2, color=bar_borders)
            ),
            name="分布",
            hovertemplate=f"區間: %{{x:.{p}f}}<br>次數: %{{y}}<extra></extra>"
        ))

        if spc_std > 0 and show_curve2:
            xc = np.linspace(amin, amax, 300)
            yc = (1/(spc_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((xc-spc_mean)/spc_std)**2)
            fig_h.add_trace(go.Scatter(
                x=xc, y=yc*spc_n*step_h, mode='lines',
                line=dict(color="#1e293b", width=2.5), name='常態曲線'
            ))

        # ── 規格線：線條畫在圖內，標籤顯示在圖表下方說明區 ──
        y_max = max(counts) if counts else 1

        # 右上角統計資訊框（報告用）- 統一字體，避免截圖時被裁切
        stats_text = (
            f"<b>樣本數　{spc_n:,}</b>"
            f"　｜　平均值　{spc_mean:.{p}f}"
            f"　｜　標準差　{spc_std:.{p}f}"
        )
        fig_h.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.97,
            text=stats_text,
            font=dict(color="#1e293b", size=13, weight=700),
            bgcolor="#f8fafc", bordercolor="#cbd5e1", borderwidth=1.5,
            borderpad=8, showarrow=False, align="left",
            xanchor="right", yanchor="top"
        )

        # 規格線：scatter 垂直線 + 頂部永久標籤框，hover 自動浮到最上層
        y_top = max(counts) if counts else 1
        label_y = y_top * 1.06   # 標籤固定位置（柱頂上方）

        # 線條設定：(x值, 顏色, dash樣式, 線寬, 標籤文字, 背景色)
        # 用 paper 座標讓線延伸到圖表頂部，標籤固定在最上方
        lines_to_draw = []
        if is_both or is_lower:
            lines_to_draw.append((lsl2,      "#ef4444", "solid",   2.5, f"LSL {lsl2:.{p}f}",      "#fee2e2"))
        if show_mean2:
            lines_to_draw.append((spc_mean,  "#059669", "dot",     2.5, f"平均 {spc_mean:.{p}f}",  "#d1fae5"))
        if show_target2:
            lines_to_draw.append((target2,   "#7c3aed", "dash",    2.5, f"目標值 {target2:.{p}f}", "#ede9fe"))
        if show_median2:
            lines_to_draw.append((spc_median,"#0284c7", "dashdot", 2.5, f"中位 {spc_median:.{p}f}","#e0f2fe"))
        if is_both or is_upper:
            lines_to_draw.append((usl2,      "#ef4444", "solid",   2.5, f"USL {usl2:.{p}f}",      "#fee2e2"))

        for x_val, color, dash, width, label, bg in lines_to_draw:
            # 密集多點讓整條線都可以 hover
            n_pts = 300
            ys = [y_top * 1.38 * i / (n_pts - 1) for i in range(n_pts)]
            fig_h.add_trace(go.Scatter(
                x=[x_val] * n_pts,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=label,
                hovertemplate=f"<b>{label}</b><extra></extra>",
                hoverlabel=dict(
                    bgcolor=color,
                    font=dict(color="#fff", size=14),
                    bordercolor=color
                ),
                showlegend=False
            ))
            # 標籤固定在圖表最頂端（paper 座標）
            fig_h.add_annotation(
                x=x_val, y=1.01,
                xref="x", yref="paper",
                text=f"<b>{label}</b>",
                font=dict(color=color, size=11),
                bgcolor=bg,
                bordercolor=color, borderwidth=1.5, borderpad=4,
                showarrow=False,
                yanchor="bottom", xanchor="center"
            )



        fig_h.update_layout(
            template="simple_white",
            plot_bgcolor="#fafafa", paper_bgcolor=CHART_BG,
            height=620, font=dict(color=CHART_TEXT, size=15),
            xaxis=dict(
                gridcolor="#e2e8f0", tickfont=dict(color=CHART_TEXT, size=14),
                title=dict(text=selected_param, font=dict(color="#64748b", size=15)),
                linecolor="#94a3b8", linewidth=1.5, showgrid=True,
                gridwidth=0.8, zeroline=False
            ),
            yaxis=dict(
                gridcolor="#e2e8f0", tickfont=dict(color=CHART_TEXT, size=14),
                title=dict(text="次數 (Frequency)", font=dict(color="#64748b", size=15)),
                linecolor="#94a3b8", linewidth=1.5, gridwidth=0.8,
                range=[0, y_top * 1.42]
            ),
            showlegend=False,
            bargap=0.04, margin=dict(t=70, b=55, l=65, r=30)
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_side:
        # 良品率大數字卡（右上）
        yield_color = "#10b981" if yield2 >= 99 else ("#f59e0b" if yield2 >= 95 else "#ef4444")
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;
            padding:22px;text-align:center;margin-bottom:12px;">
          <div style="font-size:16px;color:#94a3b8;font-weight:700;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:12px;">規格符合率</div>
          <div style="font-size:52px;font-weight:800;color:{yield_color};line-height:1;">{yield2:.1f}%</div>
          <div style="font-size:16px;color:#94a3b8;margin:8px 0 18px;">良品率</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:18px;">
              <div style="font-size:30px;font-weight:800;color:#10b981;">{in2:,}</div>
              <div style="font-size:15px;font-weight:600;color:#64748b;margin-top:6px;">符合規格</div>
            </div>
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:18px;">
              <div style="font-size:30px;font-weight:800;color:#ef4444;">{out_usl2+out_lsl2:,}</div>
              <div style="font-size:15px;font-weight:600;color:#64748b;margin-top:6px;">規格外</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # 等級說明卡（右下）
        st.markdown("""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px;">
          <div style="font-size:14px;color:#64748b;font-weight:700;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:14px;">等級說明</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:36px;height:36px;border-radius:50%;background:#d1fae5;
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:#059669;flex-shrink:0;">A+</div>
              <div style="font-size:17px;font-weight:600;color:#1e293b;">Cpk ≥ 1.67　<span style="font-weight:400;font-size:16px;color:#64748b;">精密度極佳</span></div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:36px;height:36px;border-radius:50%;background:#d1fae5;
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:#10b981;flex-shrink:0;">A</div>
              <div style="font-size:17px;font-weight:600;color:#1e293b;">1.33 – 1.67　<span style="font-weight:400;font-size:16px;color:#64748b;">精密度良好</span></div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:36px;height:36px;border-radius:50%;background:#fef9c3;
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:#b45309;flex-shrink:0;">B</div>
              <div style="font-size:17px;font-weight:600;color:#1e293b;">1.00 – 1.33　<span style="font-weight:400;font-size:16px;color:#64748b;">尚可管制</span></div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:36px;height:36px;border-radius:50%;background:#ffedd5;
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:#c2410c;flex-shrink:0;">C</div>
              <div style="font-size:17px;font-weight:600;color:#1e293b;">0.67 – 1.00　<span style="font-weight:400;font-size:16px;color:#64748b;">能力不足</span></div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:36px;height:36px;border-radius:50%;background:#fee2e2;
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:700;color:#b91c1c;flex-shrink:0;">D</div>
              <div style="font-size:17px;font-weight:600;color:#1e293b;">&lt; 0.67　<span style="font-weight:400;font-size:16px;color:#64748b;">能力極差</span></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4：診斷結論（全寬）───────────────────────
    diags = []
    _actions = {
        "A+": "繼續維持，可考慮每季降低抽樣頻率以節省管制成本。",
        "A":  "繼續維持，持續監控趨勢變化。",
        "B":  "加強製程管制，目標達到 A 級。",
        "C":  "立即啟動改善，排查機台老化與原材料穩定性。",
        "D":  "全面停機檢討，需系統性改善後方可復產。",
    }

    if spc_n < 30:
        diags.append(("warning", "⚠", f"樣本數僅 {spc_n} 筆（建議 ≥30），分析結果可信度有限。",
                      "建議：增加取樣數量後重新分析。"))
    else:
        diags.append(("ok", "✓", f"樣本數 {spc_n} 筆，符合統計基本需求。", ""))

    if ca2 is not None:
        a = abs(ca2)
        diags.append((
            "ok" if a < 12.5 else "warning" if a < 25 else "error",
            "✓" if a < 12.5 else "△" if a < 25 else "✕",
            f"Ca = {a:.1f}%（{ca_g} 級）：{ca_d}。",
            _actions.get(ca_g, "")
        ))

    diags.append((
        "ok" if cp2 >= 1.33 else "warning" if cp2 >= 1.0 else "error",
        "✓" if cp2 >= 1.33 else "△" if cp2 >= 1.0 else "✕",
        f"Cp = {cp2:.3f}（{cp_g} 級）：{cp_d}。",
        _actions.get(cp_g, "")
    ))

    diags.append((
        "ok" if cpk2 >= 1.33 else "warning" if cpk2 >= 1.0 else "error",
        "★" if cpk2 >= 1.33 else "△" if cpk2 >= 1.0 else "✕",
        f"Cpk = {cpk2:.3f}（{cpk_g} 級）：{cpk_d}。",
        _actions.get(cpk_g, "")
    ))

    if out_usl2 + out_lsl2 > 0:
        diags.append(("error", "!",
            f"規格外品：超過 USL {out_usl2} 顆、低於 LSL {out_lsl2} 顆，共 {out_usl2+out_lsl2} 顆不良。",
            "建議：立即隔離不良品，追蹤生產批次根因。"))

    has_error   = any(d[0] == "error"   for d in diags)
    has_warning = any(d[0] == "warning" for d in diags)

    if has_error:
        sb, sbd, sib, si, stc, sac = "#fef2f2","#fecaca","#fee2e2","✕","#7f1d1d","#991b1b"
        worst = next(d for d in diags if d[0]=="error")
    elif has_warning:
        sb, sbd, sib, si, stc, sac = "#fffbeb","#fde68a","#fef9c3","△","#78350f","#92400e"
        worst = next(d for d in diags if d[0]=="warning")
    else:
        sb, sbd, sib, si, stc, sac = "#f0fdf4","#bbf7d0","#dcfce7","✓","#14532d","#166534"
        worst = diags[-1]

    st.markdown(f"""
    <div style="background:{sb};border:1px solid {sbd};border-radius:12px;
        padding:18px 22px;margin-bottom:12px;">
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="width:36px;height:36px;border-radius:50%;background:{sib};
            display:flex;align-items:center;justify-content:center;
            font-size:17px;flex-shrink:0;">{si}</div>
        <div style="flex:1;">
          <div style="font-size:17px;font-weight:600;color:{stc};margin-bottom:6px;">{worst[2]}</div>
          <div style="font-size:16px;color:{sac};">建議行動：{worst[3]}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _color_map = {"ok": "#16a34a", "warning": "#d97706", "error": "#ef4444"}
    with st.expander("📋 查看完整診斷報告", expanded=False):
        for lvl, icon, msg, action in diags:
            clr = _color_map[lvl]
            st.markdown(f"""
            <div style="border-left:4px solid {clr};background:#f8fafc;
                border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px;">
              <span style="color:{clr};font-weight:700;margin-right:10px;font-size:15px;">{icon}</span>
              <span style="font-size:16px;color:#1e293b;">{msg}</span>
              {"<div style='font-size:15px;color:#64748b;margin-top:6px;padding-left:26px;'>→ " + action + "</div>" if action else ""}
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📊 評價基準對照表", expanded=False):
        st.dataframe(pd.DataFrame({
            "等級":   ["A+",    "A",         "B",         "C",        "D"],
            "Cp/Cpk": ["≥1.67", "1.33–1.67", "1.00–1.33", "0.67–1.00","<0.67"],
            "|Ca|":   ["<6.25%","6.25–12.5%","12.5–25%",  "25–50%",   ">50%"],
            "判斷":   ["製程極佳","製程良好","製程尚可",   "能力不足", "能力極差"],
            "建議":   ["可降低管制成本","繼續維持","加強管制","加強訓練","全面停機檢討"],
        }), use_container_width=True, hide_index=True)
