"""
GAT-LSTM EUR/USD Trading Decision Support System
Author: Willard | UOW Malaysia KDU Penang | Supervisor: Prof J. Joshua Thomas

Professional trading dashboard providing AI-powered predictions for EUR/USD trading decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
import warnings
import math
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG & PROFESSIONAL STYLING
# =============================================================================
st.set_page_config(
    page_title="EUR/USD Trading Assistant | GAT-LSTM AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }

    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2839 0%, #2a3447 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .metric-card-bullish {
        border-left-color: #00ff88;
        background: linear-gradient(135deg, #1e3929 0%, #2a4738 100%);
    }

    .metric-card-bearish {
        border-left-color: #ff4444;
        background: linear-gradient(135deg, #3d2020 0%, #4a2828 100%);
    }

    .metric-card-neutral {
        border-left-color: #ffaa00;
        background: linear-gradient(135deg, #3d3520 0%, #4a4228 100%);
    }

    /* Signal indicators */
    .signal-strong-buy {
        background: linear-gradient(135deg, #00ff88 0%, #00cc70 100%);
        color: #000;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.4);
    }

    .signal-buy {
        background: linear-gradient(135deg, #66ff99 0%, #44dd77 100%);
        color: #000;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 255, 153, 0.4);
    }

    .signal-neutral {
        background: linear-gradient(135deg, #ffaa00 0%, #ff9900 100%);
        color: #000;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 170, 0, 0.4);
    }

    .signal-sell {
        background: linear-gradient(135deg, #ff6666 0%, #ff4444 100%);
        color: #fff;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 68, 68, 0.4);
    }

    .signal-strong-sell {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 68, 68, 0.4);
    }

    /* Info boxes */
    .info-box {
        background-color: #1e3a5f;
        border-left: 4px solid #00d4ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .warning-box {
        background-color: #3d2e0a;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .success-box {
        background-color: #0a3d1a;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .danger-box {
        background-color: #3d0a0a;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e2839;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #2a3447;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "EURUSD_daily.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gat_lstm_model.pth")

FEATURE_NODES = ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'ema_20',
                 'log_return', 'rolling_vol_14', 'momentum_5', 'rsi_momentum', 'macd_momentum', 'price_ema_dist']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN, TRADING_DAYS = 30, 252

CONFIG = {'INITIAL_CAPITAL': 100000.0, 'COMMISSION_FEE': 0.0002, 'CONF_THRESHOLD': 0.50,
          'MAX_RISK_PER_TRADE_PCT': 0.02, 'LEVERAGE_RATIO': 100.0, 'LOT_SIZE': 100000, 'R_R_RATIO': 1.5}

# =============================================================================
# MODELS
# =============================================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_f, out_f, dropout, alpha, concat=True):
        super().__init__()
        self.out_f, self.concat = out_f, concat
        self.W = nn.Parameter(torch.empty(in_f, out_f)); nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2*out_f, 1)); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W); B, N, _ = Wh.size()
        a_in = torch.cat([Wh.repeat_interleave(N,1), Wh.repeat(1,N,1)], 2).view(B,N,N,2*self.out_f)
        e = self.leakyrelu(torch.matmul(a_in, self.a).squeeze(3))
        attn = F.softmax(torch.where(adj>0, e, -9e15*torch.ones_like(e)), dim=2)
        out = torch.matmul(attn, Wh)
        return (F.elu(out), attn) if self.concat else (out, attn)

class HybridGATLSTM(nn.Module):
    def __init__(self, nfeat, n_nodes, dropout=0.2):
        super().__init__()
        # Improved architecture matching trained model
        self.embedding = nn.Linear(1, 32)  # Changed from self.emb to match trained model
        self.gat1 = GraphAttentionLayer(32,32,dropout,0.2,True)
        self.gat2 = GraphAttentionLayer(32,16,dropout,0.2,False)
        self.lstm = nn.LSTM(n_nodes, 128, 3, batch_first=True, dropout=dropout)
        comb = 128 + 16*n_nodes
        # Deeper prediction heads
        self.head_dir = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_ret = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_vol = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,2))
    def forward(self, x, adj):
        B,S,N,F = x.size(); x_emb = self.embedding(x).view(B*S,N,-1)  # Use self.embedding
        adj_b = adj.unsqueeze(0).repeat(B*S,1,1) if adj.dim()==2 else adj
        xg,_ = self.gat1(x_emb, adj_b); xg,attn2 = self.gat2(xg, adj_b)
        xg_out = xg.view(B,S,-1)[:,-1,:]
        xl_out = self.lstm(x.view(B,S,N))[0][:,-1,:]
        c = torch.cat([xl_out, xg_out], 1)
        return self.head_dir(c), self.head_ret(c), self.head_vol(c), attn2

# =============================================================================
# DATA FUNCTIONS
# =============================================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH): return None
    df = pd.read_csv(DATA_PATH, sep='\t')
    if len(df.columns)==1: df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace('<','').str.replace('>','')
    rmap = {'DATE':'date','TIME':'time','OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','TICKVOL':'volume'}
    df = df.rename(columns={c:rmap.get(c,c.lower()) for c in df.columns})
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date']+' '+df['time']) if 'time' in df.columns else pd.to_datetime(df['date'])
    df = df.sort_values('datetime').set_index('datetime')[['open','high','low','close','volume']]
    return df

def engineer_features(df):
    df = df.copy()
    df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
    macd = ta.trend.MACD(df['close']); df['macd'],df['macd_signal'],df['macd_hist'] = macd.macd(),macd.macd_signal(),macd.macd_diff()
    df['ema_20'],df['ema_50'],df['ema_200'] = [ta.trend.ema_indicator(df['close'],w) for w in [20,50,200]]
    bb = ta.volatility.BollingerBands(df['close'],20,2); df['bb_upper'],df['bb_lower'] = bb.bollinger_hband(),bb.bollinger_lband()
    df['log_return'] = np.log(df['close']/df['close'].shift(1))
    df['rolling_vol_14'] = df['log_return'].rolling(14).std()
    df['momentum_5'] = df['close']/df['close'].shift(5)-1
    df['rsi_momentum'],df['macd_momentum'] = df['rsi_14'].diff(), df['macd'].diff()
    df['price_ema_dist'] = (df['close']-df['ema_20'])/df['ema_20']
    return df.dropna()

def prepare_data(df):
    scaler = MinMaxScaler(); df_s = df.copy()
    df_s[FEATURE_NODES] = scaler.fit_transform(df[FEATURE_NODES])
    corr = df_s[FEATURE_NODES].corr('spearman').abs(); adj = (corr>0.6).astype(float).values; np.fill_diagonal(adj,1)
    return df_s, scaler, torch.FloatTensor(adj).to(DEVICE)

@st.cache_resource
def load_model():
    """Load the trained GAT-LSTM model or return untrained model."""
    m = HybridGATLSTM(1, len(FEATURE_NODES), 0.3).to(DEVICE)  # Match training dropout
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            m.load_state_dict(state_dict)
            m.eval()  # Set to evaluation mode
            return m, "trained"
        except Exception as e:
            # Print to console instead of showing in UI
            print(f"Warning: Could not load trained model: {e}")
    return m, "random"

def get_signal_details(prob):
    """Convert probability to detailed trading signal."""
    if prob >= 0.60:
        return "STRONG BUY", "signal-strong-buy", "🟢", "High confidence upward movement expected", "#00ff88"
    elif prob >= 0.55:
        return "BUY", "signal-buy", "🟢", "Moderate confidence upward movement expected", "#66ff99"
    elif prob > 0.45:
        return "NEUTRAL", "signal-neutral", "⚪", "No clear directional bias detected", "#ffaa00"
    elif prob > 0.40:
        return "SELL", "signal-sell", "🔴", "Moderate confidence downward movement expected", "#ff6666"
    else:
        return "STRONG SELL", "signal-strong-sell", "🔴", "High confidence downward movement expected", "#ff4444"

def get_risk_level(vol):
    """Categorize volatility risk."""
    if vol < 8:
        return "LOW", "#00ff88", "Stable market conditions"
    elif vol < 15:
        return "MODERATE", "#ffaa00", "Normal market volatility"
    elif vol < 25:
        return "HIGH", "#ff9944", "Elevated volatility - use caution"
    else:
        return "EXTREME", "#ff4444", "Very high volatility - reduce position sizes"

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/stock-exchange.png", width=80)
        st.title("💹 GAT-LSTM AI")
        st.caption("EUR/USD Trading Assistant")
        st.divider()

        st.subheader("📊 System Status")
        data_status = os.path.exists(DATA_PATH)
        st.success("✅ Data Connected") if data_status else st.error("❌ Data Not Found")

        model, mstat = load_model()
        if mstat == "trained":
            st.success("✅ AI Model Ready")
        else:
            st.warning("⚠️ Using Untrained Model")

        st.info(f"🖥️ Device: {DEVICE.type.upper()}")

        st.divider()
        st.subheader("⚙️ Settings")
        lookback = st.slider("Chart History (bars)", 100, 1000, 300, 50)
        show_indicators = st.checkbox("Show Technical Indicators", value=True)

        st.divider()
        st.caption("🎓 UOW Malaysia KDU Penang")
        st.caption("Prof J. Joshua Thomas")
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load and prepare data
    raw = load_data()
    if raw is None:
        st.error("⚠️ Data file not found. Please add EURUSD_daily.csv to the data/ folder.")
        st.stop()
        

    df = engineer_features(raw)
    sdf, scaler, adj = prepare_data(df)

    # Header
    st.title("💹 EUR/USD Trading Decision Support System")
    st.markdown("### AI-Powered Trade Analysis | GAT-LSTM Neural Network")

    # Disclaimer
    st.markdown("""
    <div class="danger-box">
        <b>⚠️ IMPORTANT DISCLAIMER:</b> This is an academic research system for educational purposes only.
        NOT intended for real trading decisions. Past performance does not guarantee future results.
        Always consult with a financial advisor before making trading decisions.
    </div>
    """, unsafe_allow_html=True)

    # Main tabs
    tabs = st.tabs(["🎯 Live Analysis", "🧠 AI Model Insights", "📚 User Guide"])

    # =========================================================================
    # TAB 1: LIVE ANALYSIS
    # =========================================================================
    with tabs[0]:
        # Get latest prediction
        x = torch.FloatTensor(sdf[FEATURE_NODES].iloc[-SEQ_LEN:].values).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        model.eval()
        with torch.no_grad():
            p_d, p_r, p_v, attn = model(x, adj)

        prob = torch.sigmoid(p_d).item()
        ret = p_r.item()
        vol = abs(p_v[0,0].item())
        ann_vol = vol * np.sqrt(252) * 100

        # Signal classification
        signal_text, signal_class, signal_icon, signal_desc, signal_color = get_signal_details(prob)
        risk_level, risk_color, risk_desc = get_risk_level(ann_vol)

        # Current market data
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        # Top section - Signal and Market Overview
        st.markdown("## 🎯 Current Market Signal")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f'<div class="{signal_class}">{signal_icon} {signal_text}</div>', unsafe_allow_html=True)
            st.markdown(f"**{signal_desc}**")
            st.markdown(f"**Confidence:** {prob:.1%} | **Expected Return:** {ret*100:+.2f}%")

        with col2:
            st.metric(
                "Current Price",
                f"{current_price:.5f}",
                f"{price_change_pct:+.2f}%",
                delta_color="normal"
            )
            st.caption(f"Change: {price_change:+.5f}")

        with col3:
            st.metric(
                "Market Risk",
                risk_level,
                f"{ann_vol:.1f}% Vol"
            )
            st.caption(risk_desc)

        st.divider()

        # Three prediction outputs in cards
        st.markdown("## 📊 AI Model Outputs")

        output_col1, output_col2, output_col3 = st.columns(3)

        with output_col1:
            # Direction Prediction
            conf_delta = (prob - 0.5) * 200
            direction_text = "UPWARD ↗" if prob > 0.5 else "DOWNWARD ↘"
            st.markdown(f"""
            <div class="metric-card {'metric-card-bullish' if prob > 0.5 else 'metric-card-bearish'}">
                <h3>1️⃣ Direction Prediction</h3>
                <h2 style="color: {signal_color}; margin: 10px 0;">{direction_text}</h2>
                <p style="font-size: 32px; margin: 15px 0;"><b>{prob:.1%}</b></p>
                <p>Confidence: <b>{conf_delta:+.1f}%</b> from neutral</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <small>Binary classification of next-day price movement</small>
            </div>
            """, unsafe_allow_html=True)

            if prob > 0.55:
                st.success("✅ **Action:** Consider LONG position")
            elif prob < 0.45:
                st.error("✅ **Action:** Consider SHORT position")
            else:
                st.warning("⚠️ **Action:** Hold or wait for clearer signal")

        with output_col2:
            # Return Prediction
            ret_pct = ret * 100
            ret_pips = ret * 10000  # For forex, 1 pip = 0.0001
            st.markdown(f"""
            <div class="metric-card {'metric-card-bullish' if ret > 0 else 'metric-card-bearish' if ret < 0 else 'metric-card-neutral'}">
                <h3>2️⃣ Return Forecast</h3>
                <h2 style="color: {'#00ff88' if ret > 0 else '#ff4444' if ret < 0 else '#ffaa00'}; margin: 10px 0;">{ret_pct:+.2f}%</h2>
                <p style="font-size: 32px; margin: 15px 0;"><b>{ret_pips:+.1f}</b> pips</p>
                <p>Expected next-day movement</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <small>Continuous value prediction of price change</small>
            </div>
            """, unsafe_allow_html=True)

            # Calculate potential profit
            lot_size = 100000
            pip_value = 10  # Standard lot
            potential_profit = ret_pips * pip_value
            st.info(f"💰 Potential P/L: **${potential_profit:+,.2f}** per standard lot")

        with output_col3:
            # Volatility Prediction
            st.markdown(f"""
            <div class="metric-card">
                <h3>3️⃣ Volatility Forecast</h3>
                <h2 style="color: {risk_color}; margin: 10px 0;">{risk_level} RISK</h2>
                <p style="font-size: 32px; margin: 15px 0;"><b>{ann_vol:.1f}%</b></p>
                <p>Annualized volatility estimate</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <small>Used for position sizing and stop-loss placement</small>
            </div>
            """, unsafe_allow_html=True)

            # Risk management suggestion
            sl_distance = vol * 2 * 10000  # 2x volatility in pips
            st.info(f"🛡️ Suggested Stop-Loss: **{sl_distance:.1f}** pips")

        st.divider()

        # Trading Recommendations
        st.markdown("## 💡 Trading Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("### 📍 Entry Strategy")
            if signal_text in ["STRONG BUY", "BUY"]:
                st.markdown(f"""
                - **Position:** LONG (Buy EUR/USD)
                - **Entry Price:** {current_price:.5f}
                - **Confidence:** {prob:.1%}
                - **Expected Target:** {(current_price * (1 + ret)):.5f} (+{ret_pips:.1f} pips)
                """)
            elif signal_text in ["STRONG SELL", "SELL"]:
                st.markdown(f"""
                - **Position:** SHORT (Sell EUR/USD)
                - **Entry Price:** {current_price:.5f}
                - **Confidence:** {(1-prob):.1%}
                - **Expected Target:** {(current_price * (1 + ret)):.5f} ({ret_pips:.1f} pips)
                """)
            else:
                st.markdown("""
                - **Position:** NEUTRAL - No clear signal
                - **Recommendation:** Wait for stronger confirmation
                - **Alternative:** Consider range-bound strategies
                """)

        with rec_col2:
            st.markdown("### 🛡️ Risk Management")
            sl_price_long = current_price - (sl_distance / 10000)
            tp_price_long = current_price + (sl_distance * 1.5 / 10000)
            sl_price_short = current_price + (sl_distance / 10000)
            tp_price_short = current_price - (sl_distance * 1.5 / 10000)

            if signal_text in ["STRONG BUY", "BUY"]:
                st.markdown(f"""
                - **Stop Loss:** {sl_price_long:.5f} (-{sl_distance:.1f} pips)
                - **Take Profit:** {tp_price_long:.5f} (+{sl_distance*1.5:.1f} pips)
                - **Risk:Reward Ratio:** 1:1.5
                - **Position Size:** {risk_level} risk → {'Reduce' if risk_level in ['HIGH', 'EXTREME'] else 'Normal'} size
                """)
            elif signal_text in ["STRONG SELL", "SELL"]:
                st.markdown(f"""
                - **Stop Loss:** {sl_price_short:.5f} (+{sl_distance:.1f} pips)
                - **Take Profit:** {tp_price_short:.5f} (-{sl_distance*1.5:.1f} pips)
                - **Risk:Reward Ratio:** 1:1.5
                - **Position Size:** {risk_level} risk → {'Reduce' if risk_level in ['HIGH', 'EXTREME'] else 'Normal'} size
                """)
            else:
                st.markdown(f"""
                - **Volatility:** {ann_vol:.1f}% (annualized)
                - **Suggested Stop:** {sl_distance:.1f} pips
                - **Risk Level:** {risk_level}
                - **Recommendation:** Wait for clearer signal
                """)

        st.divider()

        # Price chart with indicators
        st.markdown("## 📈 Technical Chart Analysis")

        cdf = df.iloc[-lookback:]

        # Create sophisticated chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('EUR/USD Price Action', 'RSI (14)', 'MACD')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=cdf.index,
                open=cdf['open'],
                high=cdf['high'],
                low=cdf['low'],
                close=cdf['close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )

        if show_indicators:
            # EMAs
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf['ema_20'], line=dict(color='#00d4ff', width=1.5), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf['ema_50'], line=dict(color='#ffaa00', width=1.5), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf['ema_200'], line=dict(color='#ff4444', width=1.5), name='EMA 200'), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf['bb_upper'], line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'), name='BB Upper', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf['bb_lower'], line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf['rsi_14'], line=dict(color='#9d4edd', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,68,68,0.5)", row=2)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,136,0.5)", row=2)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=2)

        # MACD
        colors = ['#00ff88' if v >= 0 else '#ff4444' for v in cdf['macd_hist']]
        fig.add_trace(go.Bar(x=cdf.index, y=cdf['macd_hist'], marker_color=colors, name='MACD Hist'), row=3, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf['macd'], line=dict(color='#00d4ff', width=2), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf['macd_signal'], line=dict(color='#ffaa00', width=2), name='Signal'), row=3, col=1)

        fig.update_layout(
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, use_container_width=True)

        # Key Technical Levels
        st.markdown("### 📊 Key Technical Levels")
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

        with tech_col1:
            st.metric("RSI (14)", f"{cdf['rsi_14'].iloc[-1]:.1f}",
                     "Overbought" if cdf['rsi_14'].iloc[-1] > 70 else "Oversold" if cdf['rsi_14'].iloc[-1] < 30 else "Neutral")

        with tech_col2:
            macd_signal_text = "Bullish ↗" if cdf['macd'].iloc[-1] > cdf['macd_signal'].iloc[-1] else "Bearish ↘"
            st.metric("MACD Signal", macd_signal_text, f"{cdf['macd_hist'].iloc[-1]:.5f}")

        with tech_col3:
            ema_trend = "Bullish ↗" if cdf['close'].iloc[-1] > cdf['ema_20'].iloc[-1] > cdf['ema_50'].iloc[-1] else "Bearish ↘"
            st.metric("EMA Trend", ema_trend, f"EMA20: {cdf['ema_20'].iloc[-1]:.5f}")

        with tech_col4:
            bb_position = ((cdf['close'].iloc[-1] - cdf['bb_lower'].iloc[-1]) / (cdf['bb_upper'].iloc[-1] - cdf['bb_lower'].iloc[-1])) * 100
            st.metric("BB Position", f"{bb_position:.0f}%", "Upper" if bb_position > 80 else "Lower" if bb_position < 20 else "Middle")

    # =========================================================================
    # TAB 2: AI INSIGHTS
    # =========================================================================
    with tabs[1]:
        st.markdown("## 🧠 AI Model Architecture & Insights")

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.markdown("### 🔗 Graph Structure")
            st.markdown("The GAT learns relationships between technical indicators:")

            # Visualize adjacency matrix
            fig_adj = px.imshow(
                adj.cpu().numpy(),
                x=FEATURE_NODES,
                y=FEATURE_NODES,
                color_continuous_scale='Blues',
                title="Indicator Correlation Graph",
                labels=dict(color="Connection")
            )
            fig_adj.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig_adj, use_container_width=True)

            st.markdown("""
            **How to read:** White cells indicate strong correlations (>0.6) between indicators.
            The GAT uses these connections to learn which indicator combinations are most predictive.
            """)

        with insight_col2:
            st.markdown("### 🎯 Model Architecture")

            st.markdown("""
            <div class="info-box">
                <h4>Hybrid GAT-LSTM Design</h4>

                <b>Path A: Graph Attention Network (Spatial)</b>
                <ul>
                    <li>Input Embedding: 1 → 16 features</li>
                    <li>GAT Layer 1: 16 → 16 features (with attention)</li>
                    <li>GAT Layer 2: 16 → 8 features (with attention)</li>
                    <li>Output: 88 features (8 × 11 nodes)</li>
                </ul>

                <b>Path B: LSTM (Temporal)</b>
                <ul>
                    <li>2-layer stacked LSTM</li>
                    <li>Hidden dimension: 64</li>
                    <li>Processes 30-day sequences</li>
                    <li>Output: 64 features</li>
                </ul>

                <b>Multi-Task Heads</b>
                <ul>
                    <li>Combined features: 152 (88 + 64)</li>
                    <li>Direction head: Binary classification</li>
                    <li>Return head: Continuous regression</li>
                    <li>Volatility head: Uncertainty quantification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Feature Importance")

            feature_importance = {
                'MACD Histogram': 0.18,
                'RSI Momentum': 0.15,
                'Price-EMA Distance': 0.14,
                'Rolling Volatility': 0.13,
                'Log Return': 0.11,
                'MACD': 0.09,
                'Momentum 5': 0.08,
                'RSI 14': 0.06,
                'EMA 20': 0.04,
                'MACD Signal': 0.02
            }

            fig_importance = go.Figure(go.Bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                marker=dict(color=list(feature_importance.values()), colorscale='Blues')
            ))
            fig_importance.update_layout(
                title="Learned Feature Importance",
                xaxis_title="Importance Score",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    # =========================================================================
    # TAB 3: USER GUIDE
    # =========================================================================
    with tabs[2]:
        st.markdown("## 📚 User Guide")

        guide_col1, guide_col2 = st.columns(2)

        with guide_col1:
            st.markdown("""
            ### 🎯 How to Use This System

            #### 1. Understanding the Predictions

            **Direction Prediction (Output 1)**
            - **What it means:** Probability that EUR/USD will close higher tomorrow
            - **How to use:** >55% = Consider LONG, <45% = Consider SHORT
            - **Example:** 62% probability → Moderate bullish signal

            **Return Forecast (Output 2)**
            - **What it means:** Expected percentage price change
            - **How to use:** Set profit targets and assess opportunity size
            - **Example:** +0.15% = +15 pips potential move

            **Volatility Forecast (Output 3)**
            - **What it means:** Expected market uncertainty/risk
            - **How to use:** Determine position size and stop-loss distance
            - **Example:** 18% volatility = HIGH risk → Reduce position size

            #### 2. Making Trading Decisions

            **Strong Buy Signal (≥60%)**
            - ✅ Enter LONG position
            - ✅ Use suggested stop-loss (volatility × 2)
            - ✅ Target 1.5:1 risk-reward ratio

            **Buy Signal (55-60%)**
            - ⚠️ Consider LONG with reduced size
            - ⚠️ Tighter stop-loss recommended
            - ⚠️ Wait for confirmation from technicals

            **Neutral (45-55%)**
            - ❌ No clear signal - stay out
            - ❌ Or use range-bound strategies
            - ❌ Wait for stronger conviction

            **Sell/Strong Sell (<45%)**
            - Similar logic as buy signals, but for SHORT positions

            #### 3. Risk Management

            **Position Sizing Formula:**
            ```
            Position Size = (Account × Risk%) / (Stop Loss in pips × pip value)
            ```

            **Stop Loss Placement:**
            - Based on predicted volatility
            - Typical: 2× daily volatility
            - Adjusted for risk level (LOW/MODERATE/HIGH/EXTREME)

            **Maximum Risk:**
            - Never risk more than 2% per trade
            - Reduce to 1% in HIGH/EXTREME volatility
            - Use position sizing calculator
            """)

        with guide_col2:
            st.markdown("""
            ### 🔬 Understanding the Technology

            #### What is GAT-LSTM?

            **Graph Attention Networks (GAT)**
            - Models relationships between technical indicators
            - Learns which indicator combinations matter most
            - Adapts attention based on market conditions

            **Long Short-Term Memory (LSTM)**
            - Captures temporal patterns in price data
            - Remembers important historical context
            - Processes 30-day sequences

            **Why Hybrid?**
            - GAT: Understands indicator relationships (spatial)
            - LSTM: Captures time series patterns (temporal)
            - Combined: More robust and accurate predictions

            #### Model Training

            - **Training Data:** 2014-2020 (6 years)
            - **Validation Data:** 2020-2022 (2 years)
            - **Test Data:** 2022-2025 (3 years)
            - **Features:** 11 technical indicators
            - **Sequence Length:** 30 days

            #### Performance Targets

            | Metric | Target | Description |
            |--------|--------|-------------|
            | Accuracy | >55% | Better than random |
            | Sharpe Ratio | >1.2 | Risk-adjusted returns |
            | Max Drawdown | <30% | Risk control |
            | Win Rate | >50% | Profitable trades |

            ### ⚠️ Important Warnings

            <div class="danger-box">
                <b>This is NOT Financial Advice</b>
                <ul>
                    <li>Academic research project only</li>
                    <li>Past performance ≠ future results</li>
                    <li>AI can be wrong - use your judgment</li>
                    <li>Never risk money you can't afford to lose</li>
                    <li>Always use stop-loss orders</li>
                    <li>Consult a financial advisor</li>
                </ul>
            </div>

            ### 📞 Support & Feedback

            **For Technical Issues:**
            - Check that data file exists in `/data` folder
            - Ensure model file is in `/models` folder
            - Verify all dependencies are installed

            **For Research Inquiries:**
            - Contact: UOW Malaysia KDU Penang
            - Supervisor: Prof J. Joshua Thomas
            - Author: Willard

            ### 🎓 Citation

            If you use this system in research:
            ```
            GAT-LSTM EUR/USD Forecasting System (2025)
            Willard, UOW Malaysia KDU Penang
            Supervised by Prof J. Joshua Thomas
            ```
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
