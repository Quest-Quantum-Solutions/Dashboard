import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64

# --- Page setup ---
st.set_page_config(page_title="AdaptiveShield-VT18 Dashboard", layout="wide")

# --- Set Background Image ---
def set_png_as_page_bg(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    /* Force all text to white for visibility on Windows & Mac */
    html, body, [class*="st-"], .stMarkdown, .stText, .stTitle, .stSubheader, .stRadio, .stSlider, .stExpander {{
        color: #FFFFFF !important;
    }}
    table.dataframe, .dataframe th, .dataframe td {{
        background-color: black !important;
        color: #FFFFFF !important;
        border: 1px solid white !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg("QQS_background.png")

# --- Load Data ---
df = pd.read_pickle("Str_Bench_RET.pkl")  # Replace with your file
df.index = pd.to_datetime(df["Date"])
df = df[["Strat_Ret", "Bench_Ret"]]

backtest_STR_Weights = pd.read_pickle("backtest_STR_Weights.pkl")
backtest_STR_Weights.index = pd.to_datetime(backtest_STR_Weights.index)

# --- Ticker Descriptions ---
ticker_descriptions = {
    "VOO": "Vanguard S&P 500 ETF",
    "HYG": "High Yield Corporate Bond ETF",
    "IEF": "7-10 Year Treasury ETF",
    "LQD": "Investment Grade Corporate Bond ETF",
    "TLT": "20+ Year Treasury ETF",
    "VIG": "Dividend Appreciation ETF",
    "XLU": "Utilities Sector ETF",
    "GLD": "Gold ETF",
    "GSG": "Commodity ETF",
    "SH": "Short S&P 500 ETF",
    "UNG": "Natural Gas ETF",
    "USO": "Oil Fund ETF",
    "XLB": "Materials Sector ETF",
    "XLE": "Energy Sector ETF",
    "XLF": "Financial Sector ETF",
    "XLP": "Consumer Staples ETF",
}

# --- Inception date ---
inception_date = pd.Timestamp("2025-05-15")

# --- Cumulative returns ---
df_cum = (1 + df).cumprod()
df_cum = df_cum / df_cum.iloc[0]

# --- Header ---
st.title("📊 AdaptiveShield-VT18 Performance Dashboard")

# --- Highlights and period buttons ---
col_h, col_r = st.columns([3, 1])
with col_h:
    st.markdown("""
    **Highlights**
    - ⚡ Volatility Targeted (18% annual)
    - 🌡 Inflation-Resilient
    - 📈 Equity-Enhanced (VOO-based)
    """)

with col_r:
    period = st.radio(
        "",  
        ["1D", "1M", "3M", "6M", "1Y", "5Y", "All"],
        index=0,
        horizontal=True
    )

    # Map period to start date for relative return
    latest_date = df.index.max()
    if period == "1D":
        start = latest_date - pd.Timedelta(days=1)
    elif period == "1M":
        start = latest_date - pd.DateOffset(months=1)
    elif period == "3M":
        start = latest_date - pd.DateOffset(months=3)
    elif period == "6M":
        start = latest_date - pd.DateOffset(months=6)
    elif period == "1Y":
        start = latest_date - pd.DateOffset(years=1)
    elif period == "5Y":
        start = latest_date - pd.DateOffset(years=5)
    elif period == "All":
        start = df.index.min()

    df_period = df.loc[start:latest_date]
    if not df_period.empty:
        rel_return = (1 + df_period["Strat_Ret"]).prod() - 1
        arrow = "▲" if rel_return >= 0 else "▼"
        color = "green" if rel_return >= 0 else "red"
        st.markdown(
            f"<h2 style='text-align:right; color:{color};'>{arrow} {rel_return:.2%}</h2>"
            f"<p style='text-align:right; color:white;'>Latest update: {latest_date.date()}</p>",
            unsafe_allow_html=True
        )

st.markdown("---")

# --- Performance Metrics Function ---
def compute_metrics(series, benchmark):
    series = series.dropna()
    benchmark = benchmark.loc[series.index].dropna()

    total_return = (1 + series).prod() - 1
    num_days = len(series)
    ann_return = (1 + total_return)**(252 / num_days) - 1 if num_days > 0 else np.nan
    vol = series.std() * np.sqrt(252) if num_days > 1 else np.nan
    sharpe = ann_return / vol if vol != 0 else np.nan

    cumulative = (1 + series).cumprod()
    cummax = cumulative.cummax()
    drawdowns = cumulative / cummax - 1
    max_drawdown = drawdowns.min()
    end_date = drawdowns.idxmin()
    start_date = cumulative.loc[:end_date].idxmax()
    post_dd = cumulative.loc[end_date:]
    recovery_date = post_dd[post_dd >= cumulative[start_date]].first_valid_index()
    recovery_days = (recovery_date - end_date).days if recovery_date is not None else np.nan

    monthly = series.resample('M').sum()
    benchmark_monthly = benchmark.resample('M').sum()
    benchmark_monthly = benchmark_monthly.reindex(monthly.index)
    monthly_hit = (monthly > benchmark_monthly).sum() / len(monthly) if len(monthly) > 0 else np.nan

    quarterly = series.resample('Q').sum()
    benchmark_quarterly = benchmark.resample('Q').sum()
    quarterly_hit = (quarterly > benchmark_quarterly).sum() / len(quarterly) if len(quarterly) > 0 else np.nan

    annual = series.resample('Y').sum()
    benchmark_annual = benchmark.resample('Y').sum()
    annual_hit = (annual > benchmark_annual).sum() / len(annual) if len(annual) > 0 else np.nan

    return {
        "Total Return (Since Inception)": f"{total_return:.2%}",
        "CAGR (Annualized)": f"{ann_return:.2%}",
        "Volatility (Annualized)": f"{vol:.2%}",
        "Sharpe Ratio (CAGR / Vol)": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Max Drawdown Period": f"{start_date.date()} → {end_date.date()}",
        "Drawdown Recovery Time (Days)": f"{int(recovery_days) if not np.isnan(recovery_days) else 'Not Recovered'}",
        "Monthly Hit Ratio vs Benchmark": f"{monthly_hit:.1%}",
        "Quarterly Hit Ratio vs Benchmark": f"{quarterly_hit:.1%}",
        "Annual Hit Ratio vs Benchmark": f"{annual_hit:.1%}"
    }

# --- Date slider ---
start_date, end_date = st.slider(
    "📅 Select Date Range",
    min_value=df_cum.index.min().to_pydatetime(),
    max_value=df_cum.index.max().to_pydatetime(),
    value=(df_cum.index.min().to_pydatetime(), df_cum.index.max().to_pydatetime())
)

if (end_date - start_date).days < 30:
    st.warning("Please select at least a 1-month window.")
    st.stop()

# --- Filtered data ---
df_filtered = df.loc[start_date:end_date]
filtered_weights = backtest_STR_Weights.loc[start_date:end_date]

# --- Metrics ---
strategy_metrics = compute_metrics(df_filtered["Strat_Ret"], df_filtered["Bench_Ret"])
benchmark_metrics = compute_metrics(df_filtered["Bench_Ret"], df_filtered["Strat_Ret"])

st.subheader("📐 Performance Metrics: Strategy vs Benchmark")
metrics_df = pd.DataFrame({
    "Strategy": strategy_metrics,
    "Benchmark": benchmark_metrics
})

short_df = metrics_df.loc[: "Max Drawdown"]
extra_df = metrics_df.loc["Max Drawdown":].iloc[1:]

# --- Render tables as black HTML tables ---
def render_black_table(df):
    html = df.to_html(escape=False)
    html = html.replace('<table border="1" class="dataframe">', 
                        '<table border="1" class="dataframe" style="background-color:black;color:white;border-color:white;">')
    html = html.replace('<th>', '<th style="background-color:#111;color:white;">')
    html = html.replace('<td>', '<td style="background-color:black;color:white;">')
    st.markdown(html, unsafe_allow_html=True)

render_black_table(short_df)
with st.expander("🔎 Click for More Stats"):
    render_black_table(extra_df)

st.markdown("---")

# --- Cumulative Return Charts ---
filtered = df_cum.loc[start_date:end_date]
backtest = filtered[filtered.index < inception_date]
realtime = filtered[filtered.index >= inception_date]

col1, col2 = st.columns(2)
with col1:
    fig_orig = go.Figure()
    fig_orig.add_trace(go.Scatter(x=backtest.index, y=backtest["Strat_Ret"], mode="lines", name="Strategy (Backtest)", line=dict(color="blue")))
    if not realtime.empty:
        fig_orig.add_trace(go.Scatter(x=realtime.index, y=realtime["Strat_Ret"], mode="lines", name="Strategy (Real-Time)", line=dict(color="red")))
    fig_orig.add_trace(go.Scatter(x=filtered.index, y=filtered["Bench_Ret"], mode="lines", name="Benchmark (VOO)", line=dict(color="lightgray")))
    if start_date <= inception_date <= end_date:
        fig_orig.add_shape(type="line", x0=inception_date, y0=filtered.min().min(), x1=inception_date, y1=filtered.max().max(), line=dict(color="red", width=2, dash="dash"))
        fig_orig.add_annotation(x=inception_date, y=filtered.max().max(), text="Inception", showarrow=False, yanchor="bottom", yshift=20, font=dict(color="red", size=12))
    fig_orig.update_layout(
        title="Cumulative Return: Full History",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")  # Force white font for Windows
    )
    st.plotly_chart(fig_orig, use_container_width=True)

with col2:
    filtered_norm = filtered / filtered.iloc[0]
    backtest_norm = filtered_norm[filtered_norm.index < inception_date]
    realtime_norm = filtered_norm[filtered_norm.index >= inception_date]
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(x=backtest_norm.index, y=backtest_norm["Strat_Ret"], mode="lines", name="Strategy (Backtest)", line=dict(color="blue")))
    if not realtime_norm.empty:
        fig_norm.add_trace(go.Scatter(x=realtime_norm.index, y=realtime_norm["Strat_Ret"], mode="lines", name="Strategy (Real-Time)", line=dict(color="red")))
    fig_norm.add_trace(go.Scatter(x=filtered_norm.index, y=filtered_norm["Bench_Ret"], mode="lines", name="Benchmark (VOO)", line=dict(color="lightgray")))
    if start_date <= inception_date <= end_date:
        fig_norm.add_shape(type="line", x0=inception_date, y0=filtered_norm.min().min(), x1=inception_date, y1=filtered_norm.max().max(), line=dict(color="red", width=2, dash="dash"))
        fig_norm.add_annotation(x=inception_date, y=filtered_norm.max().max(), text="Inception", showarrow=False, yanchor="bottom", yshift=20, font=dict(color="red", size=12))
    fig_norm.update_layout(
        title="Cumulative Return: Normalized to Slider Start",
        xaxis_title="Date",
        yaxis_title="Normalized Return",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(fig_norm, use_container_width=True)

# (rest of your code remains unchanged — I only added `font=dict(color="white")` for charts and CSS for all text)


st.markdown("---")

# --- Monthly, Quarterly, Annual Returns & Volatility ---
periods = {"Monthly": "M", "Quarterly": "Q", "Annual": "Y"}
for name, freq in periods.items():
    ret = df_filtered.resample(freq).apply(lambda x: (1 + x).prod() - 1)
    vol = df_filtered.resample(freq).std() * np.sqrt(252)
    ret = ret.loc[ret.index <= df_filtered.index.max()]
    vol = vol.loc[vol.index <= df_filtered.index.max()]

    col1, col2 = st.columns(2)
    with col1:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(x=ret.index, y=ret["Strat_Ret"], name="Strategy", marker_color="blue"))
        fig_ret.add_trace(go.Bar(x=ret.index, y=ret["Bench_Ret"], name="Benchmark (VOO)", marker_color="gray"))
        fig_ret.update_layout(
            title=f"{name} Returns: Strategy vs Benchmark",
            xaxis_title=name,
            yaxis_title="Return",
            barmode='group',
            yaxis_tickformat=".1%",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ret, use_container_width=True)
    with col2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=vol.index, y=vol["Strat_Ret"], name="Strategy", marker_color="blue"))
        fig_vol.add_trace(go.Bar(x=vol.index, y=vol["Bench_Ret"], name="Benchmark (VOO)", marker_color="gray"))
        fig_vol.update_layout(
            title=f"{name} Volatility: Strategy vs Benchmark",
            xaxis_title=name,
            yaxis_title="Volatility",
            barmode='group',
            yaxis_tickformat=".1%",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# --- Weights Distribution & Descriptions ---
st.subheader("📊 Portfolio Weights Distribution")
col_left, col_right = st.columns([2, 1])

with col_left:
    filtered_df = filtered_weights[filtered_weights['Rescaled_Weights'] > 0].copy()
    unique_tickers = sorted(filtered_df['Ticker'].unique())
    num_tickers = len(unique_tickers)
    colors = cm.tab20(np.linspace(0, 1, num_tickers))
    colors_hex = [plt.cm.colors.to_hex(c) for c in colors]
    color_map = dict(zip(unique_tickers, colors_hex))
    color_map['VOO'] = '#BBBBBB'

    date_groups = filtered_df.groupby(filtered_df.index)
    data, dates = [], []
    for date, group in date_groups:
        weights = dict(zip(group['Ticker'], group['Rescaled_Weights']))
        row = [weights.get(ticker, 0) for ticker in unique_tickers]
        data.append(row)
        dates.append(date)

    if data:
        stack_df = pd.DataFrame(data, columns=unique_tickers, index=dates)
        fig, (ax_w, ax_pie) = plt.subplots(2, 1, figsize=(10, 12))
        fig.patch.set_alpha(0.0)  # Transparent figure background
        ax_w.set_facecolor((0, 0, 0, 0))  # Transparent axes
        ax_pie.set_facecolor((0, 0, 0, 0))

        ax_w.stackplot(stack_df.index, (stack_df.T * 100).values, labels=unique_tickers,
                       colors=[color_map[ticker] for ticker in unique_tickers], alpha=0.8)
        ax_w.axhline(100, color='white', linestyle='--', linewidth=1)
        ax_w.set_title('Weights Over Time', color='white')
        ax_w.set_ylabel('Weight (%)', color='white')
        ax_w.tick_params(colors='white')
        ax_w.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
        ax_w.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left',
                    facecolor=(0, 0, 0, 0), edgecolor='white', labelcolor='white')

        avg_weights = stack_df.mean()
        wedges, _ = ax_pie.pie(avg_weights, startangle=90,
                               colors=[color_map[ticker] for ticker in avg_weights.index],
                               labels=avg_weights.index,
                               textprops={'color': 'white', 'fontsize': 10})
        ax_pie.set_title('Average Portfolio Weights', color='white')
        st.pyplot(fig)
    else:
        st.info("No weights available in this date range.")

#with col_right:
#    if data:
#        avg_weights = stack_df.mean() * 100
#        summary_table = pd.DataFrame({
#            "Description": [ticker_descriptions.get(t, "N/A") for t in avg_weights.index],
#            "Weight (%)": avg_weights.round(2).astype(str) + "%",
#            "": [color_map[t] for t in avg_weights.index]
#        }, index=avg_weights.index)
#
#        def color_square(color):
#            return f'<div style="width:18px;height:18px;background-color:{color};border-radius:3px;margin:auto"></div>'
#
#        summary_table[""] = summary_table[""].apply(color_square)
#        st.markdown(summary_table.to_html(escape=False), unsafe_allow_html=True)
#

with col_right:
    if data:
        avg_weights = stack_df.mean() * 100
        summary_table = pd.DataFrame({
            "Description": [ticker_descriptions.get(t, "N/A") for t in avg_weights.index],
            "Weight (%)": avg_weights.round(2).astype(str) + "%",
            "": [color_map[t] for t in avg_weights.index]
        }, index=avg_weights.index)

        def color_square(color):
            return f'<div style="width:18px;height:18px;background-color:{color};border-radius:3px;margin:auto"></div>'

        summary_table[""] = summary_table[""].apply(color_square)

        # --- Render black table for summary ---
        def render_black_summary_table(df):
            html = df.to_html(escape=False)
            html = html.replace('<table border="1" class="dataframe">', 
                                '<table border="1" class="dataframe" style="background-color:black;color:white;border-color:white;">')
            html = html.replace('<th>', '<th style="background-color:#111;color:white;">')
            html = html.replace('<td>', '<td style="background-color:black;color:white;">')
            st.markdown(html, unsafe_allow_html=True)

        render_black_summary_table(summary_table)
