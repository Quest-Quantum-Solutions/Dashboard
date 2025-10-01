import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import matplotlib.colors as mcolors


# --- Page setup ---
st.set_page_config(page_title="AdaptiveShield-VT18 Dashboard", layout="wide")

# --- Set Background Image --
def set_png_as_page_bg(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stDataFrame, .stMarkdown, .stRadio, .stSlider, .stSubheader, .stTitle, .stText, .stExpander {{
            background: transparent !important;
        }}
        .stApp, .stMarkdown, .stText, .stSubheader, .stTitle, .stCaption, .stRadio, .stSlider, .stDataFrame, .stExpander {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    
    

    
    

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








# --- Highlights (daily percentage only) ---
latest_date = df.index.max()
daily_return = df["Strat_Ret"].iloc[-1]
arrow = "▲" if daily_return >= 0 else "▼"
color = "green" if daily_return >= 0 else "red"

col_h, col_r = st.columns([3, 1])
with col_h:
    st.markdown("""
    **Highlights**
    - ⚡ Volatility Targeted (18% annual)
    - 🌡 Inflation-Resilient
    - 📈 Equity-Enhanced (VOO-based)
    """)

with col_r:
    st.markdown(
        f"<h2 style='text-align:right; color:{color};'>{arrow} {daily_return:.2%}</h2>"
        f"<p style='text-align:right; color:gray;'>Latest update: {latest_date.date()}</p>",
        unsafe_allow_html=True
    )

# --- Period selector for metrics and charts (move above date slider) ---
st.sidebar.markdown("### Select Performance Period")
period = st.sidebar.radio(
    "",
    ["1M", "3M", "6M", "1Y", "5Y", "All"],
    index=0
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

# --- Date slider + Period buttons in one row ---
col_slider, col_period = st.columns([4, 1])

with col_slider:
    start_date, end_date = st.slider(
        "📅 Select Date Range",
        min_value=df_cum.index.min().to_pydatetime(),
        max_value=df_cum.index.max().to_pydatetime(),
        value=(df_cum.index.min().to_pydatetime(), df_cum.index.max().to_pydatetime())
    )

with col_period:
    period = st.radio(
        "",
        ["1M", "3M", "6M", "1Y", "5Y", "All"],
        index=0,
        horizontal=True
    )

# --- Adjust slider based on period buttons ---
if period != "All":
    if period.endswith("M"):
        months = int(period[:-1])
        start_date = end_date - pd.DateOffset(months=months)
    elif period.endswith("Y"):
        years = int(period[:-1])
        start_date = end_date - pd.DateOffset(years=years)

# --- Filter data after selection ---
df_filtered = df.loc[start_date:end_date]
filtered_weights = backtest_STR_Weights.loc[start_date:end_date]


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
        plot_bgcolor="rgba(0,0,0,0)"
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
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_norm, use_container_width=True)

st.markdown("---")


#show_more = st.button("📈 Click for More Performance")
#show_more = st.checkbox("📈 Click for More Performance")

#if show_more:
    # --- Monthly, Quarterly, Annual Returns & Volatility ---
with st.expander("📈 Click for More Performance"):
    view_option = st.radio("Select View for Monthly & Quarterly Charts", ["Last 1Y", "Full Sample"], index=0, horizontal=True)
    
    periods = {"Monthly": "M", "Quarterly": "Q", "Annual": "Y"}
    for name, freq in periods.items():
        ret = df_filtered.resample(freq).apply(lambda x: (1 + x).prod() - 1)
        vol = df_filtered.resample(freq).std() * np.sqrt(252)
        ret = ret.loc[ret.index <= df_filtered.index.max()]
        vol = vol.loc[vol.index <= df_filtered.index.max()]
    
        # Limit Monthly and Quarterly to last 12 months if selected
        if name in ["Monthly", "Quarterly"] and view_option == "Last 1Y":
            cutoff = df_filtered.index.max() - pd.DateOffset(years=1)
            ret = ret.loc[ret.index >= cutoff]
            vol = vol.loc[vol.index >= cutoff]
    
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
    
        weights_matrix = pd.DataFrame(data, columns=unique_tickers, index=dates)
    
        # --- Dynamic Weights ---
        fig_w = go.Figure()
        prev = np.zeros(len(weights_matrix))
        
        for i, ticker in enumerate(unique_tickers):
            fig_w.add_trace(go.Scatter(
                x=weights_matrix.index,
                y=prev + weights_matrix[ticker],
                mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'),  # fully transparent line
                fill='tozeroy' if i == 0 else 'tonexty',   # first trace from zero, rest stack
                fillcolor=f'rgba{(*mcolors.to_rgb(color_map[ticker]), 0.7)}', # area color
                name=ticker,
                hoverinfo='x+y+name'
            ))
            prev += weights_matrix[ticker].values
        
        fig_w.update_layout(
            title="Portfolio Weights Over Time",
            xaxis_title="Date",
            yaxis_title="Weight",
            yaxis=dict(tickformat=".0%"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True
        )
        
        st.plotly_chart(fig_w, use_container_width=True)
    
    
        # --- Pie Chart for Average Weights ---
        avg_weights = weights_matrix.mean()
        fig_pie = go.Figure(go.Pie(
            labels=avg_weights.index,
            values=avg_weights.values,
            marker_colors=[
                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{0.8})"
                for t in avg_weights.index
                for r, g, b in [mcolors.to_rgb(color_map[t])]
            ],
            hole=0.3
        ))
        fig_pie.update_layout(
            title="Average Portfolio Weights",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("Ticker Descriptions")
        for ticker, desc in ticker_descriptions.items():
            st.markdown(f"**{ticker}**: {desc}")
    
    st.markdown("---")
    
    # --- Final Note ---
    st.caption("AdaptiveShield-VT18 Dashboard © Quest Quantum Solutions")
    