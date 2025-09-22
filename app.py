import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Adaptive Shield VT18", layout="wide")

# Background customization
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: url("QQS_background.png") no-repeat center center fixed;
    background-size: cover;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stToolbar"] {right: 2rem;}
table {
    background: transparent !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("Adaptive Shield VT18 Dashboard")

# Load data (placeholder paths)
monthly_returns = pd.read_csv("monthly_returns.csv", parse_dates=["Date"])
quarterly_returns = pd.read_csv("quarterly_returns.csv", parse_dates=["Date"])
annual_returns = pd.read_csv("annual_returns.csv", parse_dates=["Date"])
weights = pd.read_csv("weights.csv", parse_dates=["Date"])

# Sidebar options
st.sidebar.header("Options")

# Toggle between full sample or last year
show_last_year = st.sidebar.radio(
    "Select data range (for Monthly & Quarterly):",
    ("Last 12 Months", "Full Sample")
)

# Filter datasets for Monthly & Quarterly
if show_last_year == "Last 12 Months":
    monthly_returns = monthly_returns[monthly_returns["Date"] >= (monthly_returns["Date"].max() - pd.DateOffset(months=12))]
    quarterly_returns = quarterly_returns[quarterly_returns["Date"] >= (quarterly_returns["Date"].max() - pd.DateOffset(months=12))]

# ---- Charts ----

# Monthly and Quarterly in an expander
with st.expander("📊 Show Monthly & Quarterly Charts", expanded=False):
    st.subheader("Monthly Returns")
    fig, ax = plt.subplots()
    ax.plot(monthly_returns["Date"], monthly_returns["Return"], label="Monthly Return", color="cyan")
    ax.axhline(0, color="white", linestyle="--")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Quarterly Returns")
    fig, ax = plt.subplots()
    ax.plot(quarterly_returns["Date"], quarterly_returns["Return"], label="Quarterly Return", color="orange")
    ax.axhline(0, color="white", linestyle="--")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.legend()
    st.pyplot(fig)

# Annual always visible
st.subheader("Annual Returns")
fig, ax = plt.subplots()
ax.plot(annual_returns["Date"], annual_returns["Return"], label="Annual Return", color="lime")
ax.axhline(0, color="white", linestyle="--")
ax.set_facecolor("none")
fig.patch.set_alpha(0.0)
ax.legend()
st.pyplot(fig)


# --- Portfolio Weights Distribution ---
st.subheader("📊 Portfolio Weights Distribution")
col_left, col_right = st.columns([2, 1])

with col_left:
    filtered_df = weights[weights['Rescaled_Weights'] > 0].copy()
    unique_tickers = sorted(filtered_df['Ticker'].unique())
    num_tickers = len(unique_tickers)
    colors = plt.cm.tab20(np.linspace(0, 1, num_tickers))
    colors_hex = [plt.cm.colors.to_hex(c) for c in colors]
    color_map = dict(zip(unique_tickers, colors_hex))
    color_map['VOO'] = '#BBBBBB'

    date_groups = filtered_df.groupby(filtered_df.index)
    data, dates = [], []
    for date, group in date_groups:
        weights_row = dict(zip(group['Ticker'], group['Rescaled_Weights']))
        data.append([weights_row.get(t, 0) for t in unique_tickers])
        dates.append(date)

    if data:
        stack_df = pd.DataFrame(data, columns=unique_tickers, index=dates)
        fig, (ax_stack, ax_pie) = plt.subplots(2, 1, figsize=(10, 12))
        fig.patch.set_alpha(0.0)
        ax_stack.set_facecolor((0, 0, 0, 0))
        ax_pie.set_facecolor((0, 0, 0, 0))

        ax_stack.stackplot(stack_df.index, (stack_df.T * 100).values, labels=unique_tickers,
                           colors=[color_map[ticker] for ticker in unique_tickers], alpha=0.8)
        ax_stack.axhline(100, color='white', linestyle='--', linewidth=1)
        ax_stack.set_title('Weights Over Time', color='white')
        ax_stack.set_ylabel('Weight (%)', color='white')
        ax_stack.tick_params(colors='white')
        ax_stack.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
        ax_stack.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left',
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

with col_right:
    if data:
        avg_weights = stack_df.mean() * 100
        ticker_descriptions = {t: t for t in unique_tickers}  # Replace with actual descriptions
        summary_table = pd.DataFrame({
            "Description": [ticker_descriptions.get(t, "N/A") for t in avg_weights.index],
            "Weight (%)": avg_weights.round(2).astype(str) + "%",
            "": [color_map[t] for t in avg_weights.index]
        }, index=avg_weights.index)

        def color_square(color):
            return f'<div style="width:18px;height:18px;background-color:{color};border-radius:3px;margin:auto"></div>'

        summary_table[""] = summary_table[""].apply(color_square)

        # Render black-styled table
        def render_black_summary_table(df):
            html = df.to_html(escape=False)
            html = html.replace('<table border="1" class="dataframe">', 
                                '<table border="1" class="dataframe" style="background-color:black;color:white;border-color:white;">')
            html = html.replace('<th>', '<th style="background-color:#111;color:white;">')
            html = html.replace('<td>', '<td style="background-color:black;color:white;">')
            st.markdown(html, unsafe_allow_html=True)

        render_black_summary_table(summary_table)
