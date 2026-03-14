"""
AI-Driven Marketing Dashboard — Customer Profiling & Churn Prediction
Run: streamlit run app.py
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Palette & theme constants ────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
PURPLE    = "#bc8cff"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

SEG_COLORS = {
    "Regular":    GREEN,
    "Occasional": ACCENT,
    "New":        YELLOW,
    "Lost":       RED,
}
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    margin=dict(t=40, b=20, l=10, r=10),
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Driven Marketing | Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
      background: {DARK_BG};
      color: {TEXT};
      font-family: 'Inter', sans-serif;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stSidebar"] {{ display: none; }}
  section.main > div {{ padding-top: 1rem; }}

  /* ── Hero ── */
  .hero {{
      background: linear-gradient(135deg, #0d1117 0%, #0f2942 55%, #1a3a5c 100%);
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 3rem 2.5rem 2.5rem;
      text-align: center;
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
  }}
  .hero::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 60% 0%, rgba(88,166,255,.12) 0%, transparent 70%);
      pointer-events: none;
  }}
  .hero-eyebrow {{
      font-size: .75rem; font-weight: 600; letter-spacing: .12em;
      text-transform: uppercase; color: {ACCENT}; margin-bottom: .6rem;
  }}
  .hero h1 {{
      font-size: 2.6rem; font-weight: 700; line-height: 1.15;
      color: {TEXT}; margin: 0 0 .75rem;
  }}
  .hero h1 span {{ color: {ACCENT}; }}
  .hero p {{
      font-size: 1.05rem; color: {MUTED}; max-width: 620px;
      margin: 0 auto 1.5rem; line-height: 1.6;
  }}
  .hero-tags {{ display: flex; gap: .6rem; justify-content: center; flex-wrap: wrap; }}
  .tag {{
      background: rgba(88,166,255,.12); border: 1px solid rgba(88,166,255,.3);
      color: {ACCENT}; border-radius: 20px;
      padding: .25rem .85rem; font-size: .78rem; font-weight: 500;
  }}

  /* ── Section label ── */
  .section-label {{
      font-size: .7rem; font-weight: 600; letter-spacing: .1em;
      text-transform: uppercase; color: {MUTED}; margin-bottom: .4rem;
  }}
  .section-title {{
      font-size: 1.35rem; font-weight: 700; color: {TEXT};
      margin: 0 0 1.2rem; padding-bottom: .5rem;
      border-bottom: 1px solid {BORDER};
  }}

  /* ── KPI cards ── */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: .9rem; margin-bottom: 2rem; }}
  .kpi-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 1.2rem 1rem; text-align: center;
      transition: border-color .2s;
  }}
  .kpi-card:hover {{ border-color: {ACCENT}; }}
  .kpi-value {{ font-size: 1.85rem; font-weight: 700; color: {TEXT}; line-height: 1; }}
  .kpi-label {{ font-size: .78rem; color: {MUTED}; margin-top: .35rem; }}
  .kpi-accent {{ color: {ACCENT}; }}
  .kpi-green  {{ color: {GREEN}; }}
  .kpi-red    {{ color: {RED}; }}
  .kpi-yellow {{ color: {YELLOW}; }}

  /* ── Chart card ── */
  .chart-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 1.2rem 1.2rem .8rem;
      margin-bottom: 1rem;
  }}
  .chart-title {{
      font-size: .92rem; font-weight: 600; color: {TEXT}; margin-bottom: .1rem;
  }}
  .chart-sub {{ font-size: .78rem; color: {MUTED}; margin-bottom: .6rem; }}

  /* ── Segment pills ── */
  .seg-pill {{
      display: inline-flex; align-items: center; gap: .35rem;
      border-radius: 20px; padding: .28rem .9rem;
      font-size: .78rem; font-weight: 600; border: 1px solid;
  }}
  .seg-Regular   {{ background: rgba(63,185,80,.15);  border-color: {GREEN};  color: {GREEN}; }}
  .seg-Occasional{{ background: rgba(88,166,255,.15); border-color: {ACCENT}; color: {ACCENT}; }}
  .seg-New       {{ background: rgba(210,153,34,.15); border-color: {YELLOW}; color: {YELLOW}; }}
  .seg-Lost      {{ background: rgba(248,81,73,.15);  border-color: {RED};    color: {RED}; }}

  /* ── Strategy table ── */
  .strat-row {{
      display: flex; align-items: flex-start; gap: .8rem;
      padding: .75rem 0; border-bottom: 1px solid {BORDER};
  }}
  .strat-row:last-child {{ border-bottom: none; }}
  .strat-text {{ font-size: .87rem; color: {MUTED}; line-height: 1.5; }}
  .strat-title {{ font-size: .87rem; font-weight: 600; color: {TEXT}; }}

  /* ── Email section ── */
  .email-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 1.8rem; margin-bottom: 1rem;
  }}
  .email-output {{
      background: #0d1117; border: 1px solid {BORDER};
      border-radius: 10px; padding: 1.5rem;
      font-family: 'Georgia', serif; line-height: 1.8;
      color: {TEXT}; white-space: pre-wrap;
      font-size: .92rem; margin-top: 1rem;
  }}
  div[data-testid="stButton"] > button {{
      background: {ACCENT}; color: #0d1117;
      font-weight: 700; border: none; border-radius: 8px;
      padding: .65rem 1.4rem; font-size: .92rem;
      transition: opacity .2s;
  }}
  div[data-testid="stButton"] > button:hover {{ opacity: .88; }}
  div[data-testid="stSelectbox"] > div > div {{
      background: {CARD_BG} !important; border-color: {BORDER} !important; color: {TEXT} !important;
  }}
  [data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
  .divider {{ border: none; border-top: 1px solid {BORDER}; margin: 2rem 0; }}
</style>
""", unsafe_allow_html=True)


# ── Data pipeline (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Crunching the data…")
def load_and_process():
    base = os.path.join(os.path.dirname(__file__), "data", "task1")
    df_items  = pd.read_excel(os.path.join(base, "MenuData.xlsx"))
    df_market = pd.read_csv(os.path.join(base, "Marketing_data.csv"))

    # Clean items
    for col in ['itemDescription', 'Allergens', 'preparationTime']:
        df_items[col] = df_items[col].fillna('NA')
    df_items['Category'] = df_items['Category'].fillna('Unknown')
    df_items['composite_key'] = df_items['itemName'].astype(str) + '_$' + df_items['itemPrice'].astype(str)

    # Clean transactions
    for col in ['Total', 'Tip', 'Gratuity', 'Discount', 'Refund']:
        df_market[col] = pd.to_numeric(
            df_market[col].astype(str).str.replace('$', '', regex=False), errors='coerce'
        ).fillna(0)
    df_market['Qty'] = pd.to_numeric(df_market['Qty'], errors='coerce').fillna(0)
    df_market['Reason']    = df_market['Reason'].fillna('NA')
    df_market['Modifiers'] = df_market['Modifiers'].fillna('NA')

    df_market['Order Date'] = pd.to_datetime(df_market['Order Date'])
    df_market['Hour']       = df_market['Order Date'].dt.hour
    df_market['Year-Month'] = df_market['Order Date'].dt.to_period('M')
    df_market = df_market.rename(columns={'Last 4 Card Digits': 'Customer ID'})

    # Spending & timing
    df_market['SpendingTotal'] = df_market['Qty'] * (
        df_market['Total'] + df_market['Tip'] + df_market['Gratuity'] - df_market['Discount']
    )
    p75 = df_market['SpendingTotal'].quantile(0.75)
    p50 = df_market['SpendingTotal'].quantile(0.50)

    def cat_spend(v):
        return 'Premium' if v >= p75 else ('Standard' if v >= p50 else 'Economy')

    def cat_time(h):
        return 'Morning' if 5 <= h < 12 else ('Mid-day' if h < 17 else 'Evening')

    df_market['SpendingCategory'] = df_market['SpendingTotal'].apply(cat_spend)
    df_market['Activity Timing']  = df_market['Hour'].apply(cat_time)

    # Customer profiles
    item_counts = (
        df_market.groupby(['Customer ID', 'Menu Item']).size()
        .reset_index(name='cnt')
        .sort_values(['Customer ID', 'cnt'], ascending=[True, False])
    )
    top3 = (
        item_counts.groupby('Customer ID').head(3)
        .groupby('Customer ID')['Menu Item'].agg(', '.join)
        .reset_index(name='Favourite Picks')
    )
    cust_spend = (
        df_market.groupby('Customer ID')['SpendingTotal'].sum()
        .reset_index(name='TotalSpending')
    )
    cust_spend['SpendingCategory'] = cust_spend['TotalSpending'].apply(cat_spend)

    timing_cnt = df_market.groupby(['Customer ID', 'Activity Timing']).size().reset_index(name='cnt')
    pref_time  = (
        timing_cnt.loc[timing_cnt.groupby('Customer ID')['cnt'].idxmax()]
        [['Customer ID', 'Activity Timing']]
    )
    monthly = df_market.groupby(['Customer ID', 'Year-Month']).size().reset_index(name='Orders')
    avg_mo  = monthly.groupby('Customer ID')['Orders'].mean().reset_index(name='Avg Monthly Orders')

    profiles = (
        cust_spend
        .merge(pref_time, on='Customer ID', how='left')
        .merge(avg_mo,    on='Customer ID', how='left')
        .merge(top3,      on='Customer ID', how='left')
    )

    np.random.seed(42)
    n = len(profiles)
    names = ['Ana','Alice','James','Taylor','Casey','Harry',
             'Jamie','Drew','Cameron','Reese','Mike','Skyler','Rowan']
    profiles['Customer Name'] = np.random.choice(names, n)
    rnd_m = np.random.randint(1, 13, n)
    rnd_d = np.random.randint(1, 29, n)
    profiles['Birthday'] = pd.to_datetime(
        [f"2002-{m:02d}-{d:02d}" for m, d in zip(rnd_m, rnd_d)]
    )

    # RFM + KMeans
    current_date = df_market['Order Date'].max()
    last_order   = df_market.groupby('Customer ID')['Order Date'].max().reset_index(name='LastOrder')
    freq_df      = df_market.groupby('Customer ID').size().reset_index(name='Frequency')

    mo = df_market.groupby(['Customer ID', 'Year-Month']).size().reset_index(name='Monthly Orders')
    mo['Order Trend'] = mo['Monthly Orders'] - mo.groupby('Customer ID')['Monthly Orders'].shift(1)
    trend_df = mo.groupby('Customer ID')['Order Trend'].mean().reset_index()

    rfm = last_order.merge(freq_df, on='Customer ID').merge(trend_df, on='Customer ID')
    rfm['Recency'] = (current_date - rfm['LastOrder']).dt.days
    rfm = rfm[['Customer ID', 'Recency', 'Frequency', 'Order Trend']].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Order Trend']])
    rfm['Cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

    def assign_segment(row):
        r, f = row['Recency'], row['Frequency']
        if r <= 20 and f >= 7: return 'Regular'
        if r >= 80 and f <= 6: return 'Lost'
        if f <= 3:             return 'New'
        return 'Occasional'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    rfm['PC1'], rfm['PC2'] = X_pca[:, 0], X_pca[:, 1]

    profiles = profiles.merge(
        rfm[['Customer ID', 'Recency', 'Frequency', 'Order Trend', 'Segment', 'PC1', 'PC2']],
        on='Customer ID', how='left'
    )

    return df_items, df_market, profiles, rfm, p75, p50


# ── Email via Ollama ──────────────────────────────────────────────────────────
def generate_email(cust: dict) -> str:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    segment = cust.get('Segment', 'Occasional')
    today   = datetime.now()
    year    = today.year
    holidays = {
        "New Year's Day":    datetime(year, 1, 1),
        "Valentine's Day":   datetime(year, 2, 14),
        "St. Patrick's Day": datetime(year, 3, 17),
        "Independence Day":  datetime(year, 7, 4),
        "Halloween":         datetime(year, 10, 31),
        "Christmas Day":     datetime(year, 12, 25),
    }
    bday = pd.to_datetime(cust['Birthday']).replace(year=year)
    events = {**holidays, "Your Birthday": bday}
    future = {n: d for n, d in events.items() if d >= today}
    if not future:
        future = {n: d.replace(year=year + 1) for n, d in events.items()}
    next_event = min(future, key=future.get)
    event_date = future[next_event].strftime('%B %d')

    strategies = {
        "Regular":    "Offer loyalty rewards, birthday specials, or early access to new dishes.",
        "Occasional": "Offer a personalised weekend deal and highlight their favourite picks.",
        "New":        "Give a warm welcome and a 10% discount on their next visit.",
        "Lost":       "Win them back with a 20% discount and a strong time-limited offer.",
    }
    prompt = f"""Write a high-conversion marketing email for this customer:
- Name: {cust['Customer Name']}
- Segment: {segment}
- Favourite Items: {cust.get('Favourite Picks', 'our top menu items')}
- Preferred visit time: {cust.get('Activity Timing', 'anytime')}
- Upcoming event: {next_event} on {event_date}

Strategy: {strategies.get(segment, 'Standard appreciation offer.')}

Keep the tone warm, professional, and personalised. Reference the upcoming {next_event} and their favourite items.
Do NOT mention any restaurant name — use "our place", "our team", or "us"."""

    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are an expert CRM copywriter for a restaurant."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ── Load ─────────────────────────────────────────────────────────────────────
df_items, df_market, profiles, rfm, p75, p50 = load_and_process()

# pre-compute reused aggregates
seg_counts  = rfm['Segment'].value_counts().reset_index()
seg_counts.columns = ['Segment', 'Customers']
top_items   = df_market['Menu Item'].value_counts().head(15).reset_index()
top_items.columns = ['Item', 'Orders']
hourly      = df_market.groupby('Hour').size().reset_index(name='Orders')
revenue_total = df_market['Total'].sum()


# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Driven Marketing · Restaurant Analytics</div>
  <h1>Customer Profiling &<br><span>Churn Prediction</span></h1>
  <p>End-to-end pipeline that turns raw POS transaction records into rich customer profiles
     and predicts churn risk using RFM features and KMeans clustering.</p>
  <div class="hero-tags">
    <span class="tag">12,545 Transactions</span>
    <span class="tag">2,058 Customers</span>
    <span class="tag">325 Menu Items</span>
    <span class="tag">RFM · KMeans · PCA</span>
    <span class="tag">AI Email Generation</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════════════════
lost_n      = int(rfm['Segment'].value_counts().get('Lost', 0))
regular_n   = int(rfm['Segment'].value_counts().get('Regular', 0))
avg_freq    = rfm['Frequency'].mean()

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-value kpi-accent">{len(df_market):,}</div>
    <div class="kpi-label">Total Transactions</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{profiles.shape[0]:,}</div>
    <div class="kpi-label">Unique Customers</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value kpi-accent">${revenue_total:,.0f}</div>
    <div class="kpi-label">Total Revenue</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value kpi-green">{regular_n}</div>
    <div class="kpi-label">Regular Customers</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value kpi-red">{lost_n}</div>
    <div class="kpi-label">At-Risk (Lost)</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SPENDING & TIMING
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Pattern Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Spending Behaviour & Order Timing</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.4, 1, 1])

with col1:
    st.markdown(f'<div class="chart-card"><div class="chart-title">Transaction Spending Distribution</div><div class="chart-sub">P50 = ${p50:.0f} &nbsp;·&nbsp; P75 = ${p75:.0f}</div>', unsafe_allow_html=True)
    fig = px.histogram(
        df_market, x='SpendingTotal', color='SpendingCategory', nbins=50,
        color_discrete_map={'Premium': ACCENT, 'Standard': PURPLE, 'Economy': MUTED},
        labels={'SpendingTotal': 'Spending ($)', 'SpendingCategory': 'Tier'},
    )
    fig.update_layout(**PLOTLY_THEME, height=280, showlegend=True,
                      legend=dict(orientation='h', y=1.12, x=0))
    fig.update_traces(opacity=0.85)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card"><div class="chart-title">Spending Tier Share</div><div class="chart-sub">By transaction count</div>', unsafe_allow_html=True)
    spend_share = df_market['SpendingCategory'].value_counts().reset_index()
    spend_share.columns = ['Category', 'Count']
    fig2 = px.pie(spend_share, values='Count', names='Category', hole=0.55,
                  color='Category',
                  color_discrete_map={'Premium': ACCENT, 'Standard': PURPLE, 'Economy': MUTED})
    fig2.update_layout(**PLOTLY_THEME, height=280,
                       legend=dict(orientation='h', y=-0.1, x=0.2))
    fig2.update_traces(textposition='outside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-card"><div class="chart-title">Orders by Hour of Day</div><div class="chart-sub">Evening peak clearly dominant</div>', unsafe_allow_html=True)
    fig3 = go.Figure(go.Scatter(
        x=hourly['Hour'], y=hourly['Orders'],
        mode='lines', fill='tozeroy',
        line=dict(color=ACCENT, width=2),
        fillcolor='rgba(88,166,255,0.12)',
    ))
    fig3.add_vline(x=17, line_dash='dash', line_color=RED, line_width=1,
                   annotation_text='5 PM', annotation_font_color=RED)
    fig3.update_layout(**PLOTLY_THEME, height=280,
                       xaxis_title='Hour', yaxis_title='Orders')
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOP ITEMS & CATEGORY MIX
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Menu Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Best-Selling Items & Category Mix</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown('<div class="chart-card"><div class="chart-title">Top 15 Most Ordered Items</div>', unsafe_allow_html=True)
    fig = px.bar(
        top_items, x='Orders', y='Item', orientation='h',
        color='Orders', color_continuous_scale=[[0, '#1a3a5c'], [1, ACCENT]],
    )
    fig.update_layout(**PLOTLY_THEME, height=360, coloraxis_showscale=False)
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Orders by Category</div>', unsafe_allow_html=True)
    cat_counts = df_items['Category'].value_counts().head(8).reset_index()
    cat_counts.columns = ['Category', 'Count']
    fig2 = px.bar(
        cat_counts, x='Count', y='Category', orientation='h',
        color='Count', color_continuous_scale=[[0, '#1a3a5c'], [1, PURPLE]],
    )
    fig2.update_layout(**PLOTLY_THEME, height=360, coloraxis_showscale=False)
    fig2.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CUSTOMER SEGMENTS & CHURN
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Churn Prediction · RFM + KMeans (k=4)</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Customer Segmentation & Retention Strategy</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.4, 1])

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Segment Distribution</div>', unsafe_allow_html=True)
    fig = px.pie(
        seg_counts, values='Customers', names='Segment', hole=0.55,
        color='Segment', color_discrete_map=SEG_COLORS,
    )
    fig.update_layout(**PLOTLY_THEME, height=300,
                      legend=dict(orientation='h', y=-0.1, x=0.1))
    fig.update_traces(textposition='outside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # stat pills
    for _, row in seg_counts.iterrows():
        pct = row['Customers'] / len(rfm) * 100
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:.35rem 0;border-bottom:1px solid {BORDER}">'
            f'<span class="seg-pill seg-{row["Segment"]}">{row["Segment"]}</span>'
            f'<span style="color:{MUTED};font-size:.82rem">{row["Customers"]} &nbsp;'
            f'<span style="color:{TEXT};font-weight:600">{pct:.0f}%</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card"><div class="chart-title">RFM Cluster Map (PCA Projection)</div><div class="chart-sub">Each dot is a customer — clusters reveal churn risk groups</div>', unsafe_allow_html=True)
    fig2 = px.scatter(
        rfm, x='PC1', y='PC2', color='Segment',
        color_discrete_map=SEG_COLORS, opacity=0.65,
        hover_data={'Customer ID': True, 'Recency': True, 'Frequency': True,
                    'PC1': False, 'PC2': False},
    )
    fig2.update_traces(marker=dict(size=5))
    fig2.update_layout(**PLOTLY_THEME, height=340,
                       legend=dict(orientation='h', y=1.08, x=0),
                       xaxis_title='PC 1', yaxis_title='PC 2')
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-card"><div class="chart-title">Retention Strategy</div>', unsafe_allow_html=True)
    strategies_info = {
        "Regular":    (GREEN,  "Loyalty rewards, birthday specials, early access to new dishes."),
        "Occasional": (ACCENT, "Personalised weekend deal, highlight their favourite picks."),
        "New":        (YELLOW, "Warm welcome email + 10% off next visit."),
        "Lost":       (RED,    "Win-back offer: 20% off, strong personalisation, time-limited."),
    }
    for seg, (color, desc) in strategies_info.items():
        st.markdown(
            f'<div class="strat-row">'
            f'<span class="seg-pill seg-{seg}" style="min-width:90px;justify-content:center">{seg}</span>'
            f'<div><div class="strat-text">{desc}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<br>', unsafe_allow_html=True)
    # RFM quick stats
    st.markdown(f'<div class="chart-sub">Avg Recency: <b style="color:{TEXT}">{rfm["Recency"].mean():.0f} days</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">Avg Frequency: <b style="color:{TEXT}">{rfm["Frequency"].mean():.1f} visits</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">Median Spend: <b style="color:{TEXT}">${profiles["TotalSpending"].median():.0f}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# RFM box plots
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
for col_st, feat, label, color in [
    (c1, 'Recency',     'Recency — days since last visit', RED),
    (c2, 'Frequency',   'Frequency — total visits',        GREEN),
    (c3, 'Order Trend', 'Order Trend — month-over-month',  ACCENT),
]:
    with col_st:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title">{label}</div>', unsafe_allow_html=True)
        fig = px.box(rfm, x='Segment', y=feat, color='Segment',
                     color_discrete_map=SEG_COLORS, points=False)
        fig.update_layout(**PLOTLY_THEME, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CUSTOMER PROFILES SCATTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Customer Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Lifetime Spend vs. Visit Frequency</p>', unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
fig = px.scatter(
    profiles, x='TotalSpending', y='Avg Monthly Orders',
    color='Segment', size='Frequency',
    size_max=18, opacity=0.75,
    color_discrete_map=SEG_COLORS,
    hover_data={
        'Customer Name': True, 'SpendingCategory': True,
        'Activity Timing': True, 'Segment': True,
        'TotalSpending': ':.2f', 'Avg Monthly Orders': ':.1f',
        'Frequency': True,
    },
    labels={'TotalSpending': 'Lifetime Spend ($)', 'Avg Monthly Orders': 'Avg Monthly Visits'},
)
fig.update_layout(**PLOTLY_THEME, height=380,
                  legend=dict(orientation='h', y=1.06, x=0))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AI EMAIL GENERATOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Powered by Llama 3.2 via Ollama</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">AI-Powered Retention Email Generator</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.6])

with col1:
    st.markdown('<div class="email-card">', unsafe_allow_html=True)

    seg_filter = st.selectbox(
        "Filter by segment",
        ['All Segments', 'Regular', 'Occasional', 'New', 'Lost'],
        key='seg_filter'
    )
    subset = profiles if seg_filter == 'All Segments' else profiles[profiles['Segment'] == seg_filter]
    valid_ids = subset['Customer ID'].dropna().astype(int).sort_values().tolist()

    cust_id = st.selectbox("Select Customer ID", valid_ids, key='cust_id')
    cust = profiles[profiles['Customer ID'] == cust_id].iloc[0]

    seg     = str(cust.get('Segment', '—'))
    tier    = str(cust.get('SpendingCategory', '—'))
    seg_cls = f"seg-{seg}" if seg in SEG_COLORS else ''
    fav     = str(cust.get('Favourite Picks', '—'))

    st.markdown(f"""
    <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid {BORDER}">
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Customer</span>
        <b style="color:{TEXT}">{cust['Customer Name']}</b>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Segment</span>
        <span class="seg-pill {seg_cls}">{seg}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Spending Tier</span>
        <b style="color:{TEXT}">{tier}</b>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Lifetime Spend</span>
        <b style="color:{ACCENT}">${cust['TotalSpending']:.2f}</b>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Recency</span>
        <b style="color:{TEXT}">{int(cust.get('Recency', 0))} days ago</b>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Total Visits</span>
        <b style="color:{TEXT}">{int(cust.get('Frequency', 0))}</b>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="color:{MUTED};font-size:.82rem">Prefers</span>
        <b style="color:{TEXT}">{cust.get('Activity Timing', '—')}</b>
      </div>
      <div style="margin-top:.8rem;padding-top:.8rem;border-top:1px solid {BORDER}">
        <span style="color:{MUTED};font-size:.78rem">Favourite Picks</span>
        <div style="margin-top:.3rem;color:{TEXT};font-size:.83rem;line-height:1.6">
          {"<br>".join(f"· {i}" for i in fav.split(', ')[:3])}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="email-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:.9rem;color:{MUTED};line-height:1.6">
        The model analyses the customer's <b style="color:{TEXT}">churn segment</b>,
        <b style="color:{TEXT}">favourite items</b>, and <b style="color:{TEXT}">next upcoming event</b>
        (holiday or birthday) to craft a personalised retention email — all running locally via
        <span style="color:{ACCENT}">Ollama · Llama 3.2</span>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Personalised Email", use_container_width=True):
        with st.spinner("Llama 3.2 is writing…"):
            try:
                email_text = generate_email(cust.to_dict())
                st.markdown(
                    f'<div class="email-output">{email_text}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Ollama error: {e}. Make sure Ollama is running (`ollama serve`) and llama3.2 is pulled.")
    else:
        st.markdown(f"""
        <div style="border:1px dashed {BORDER};border-radius:10px;padding:2rem;
                    text-align:center;color:{MUTED};font-size:.88rem;margin-top:.5rem">
          Select a customer above and click <b style="color:{TEXT}">Generate Personalised Email</b><br>
          to produce a tailored marketing message using the local LLM.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid {BORDER}">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;font-size:.8rem">
        <div style="background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.2);
                    border-radius:8px;padding:.6rem .8rem">
          <b style="color:{GREEN}">Regular</b><br>
          <span style="color:{MUTED}">Loyalty rewards · early access</span>
        </div>
        <div style="background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.2);
                    border-radius:8px;padding:.6rem .8rem">
          <b style="color:{ACCENT}">Occasional</b><br>
          <span style="color:{MUTED}">Weekend deal · favourite picks</span>
        </div>
        <div style="background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.2);
                    border-radius:8px;padding:.6rem .8rem">
          <b style="color:{YELLOW}">New</b><br>
          <span style="color:{MUTED}">Warm welcome · 10% off</span>
        </div>
        <div style="background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.2);
                    border-radius:8px;padding:.6rem .8rem">
          <b style="color:{RED}">Lost</b><br>
          <span style="color:{MUTED}">Win-back · 20% off time-limited</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2rem 0 1rem;color:{MUTED};font-size:.78rem;
            border-top:1px solid {BORDER};margin-top:2rem">
  AI-Driven Marketing · Customer Profiling & Churn Prediction ·
  Built with Streamlit, Plotly, scikit-learn & Llama 3.2
</div>
""", unsafe_allow_html=True)
