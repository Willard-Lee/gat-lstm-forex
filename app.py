"""
GAT-LSTM EUR/USD Forecasting System - Streamlit Dashboard
Author: Willard | UOW Malaysia KDU Penang | Supervisor: Prof J. Joshua Thomas
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

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(page_title="GAT-LSTM EUR/USD", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .info-box { background-color: #1e3a5f; border-left: 4px solid #00d4ff; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .warning-box { background-color: #3d2e0a; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .success-box { background-color: #0a3d1a; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
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
          'MAX_RISK_PER_TRADE_PCT': 0.02, 'LEVERAGE_RATIO': 100.0, 'LOT_SIZE': 100000, 'R_R_RATIO': 1.0}

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
        self.emb = nn.Linear(1, 16)
        self.gat1, self.gat2 = GraphAttentionLayer(16,16,dropout,0.2,True), GraphAttentionLayer(16,8,dropout,0.2,False)
        self.lstm = nn.LSTM(n_nodes, 64, 2, batch_first=True, dropout=dropout)
        comb = 64 + 8*n_nodes
        self.head_dir = nn.Sequential(nn.Linear(comb,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_ret = nn.Sequential(nn.Linear(comb,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_vol = nn.Sequential(nn.Linear(comb,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,2))
    def forward(self, x, adj):
        B,S,N,F = x.size(); x_emb = self.emb(x).view(B*S,N,-1)
        adj_b = adj.unsqueeze(0).repeat(B*S,1,1) if adj.dim()==2 else adj
        xg,_ = self.gat1(x_emb, adj_b); xg,_ = self.gat2(xg, adj_b)
        xg_out = xg.view(B,S,-1)[:,-1,:]
        xl_out = self.lstm(x.view(B,S,N))[0][:,-1,:]
        c = torch.cat([xl_out, xg_out], 1)
        return self.head_dir(c), self.head_ret(c), self.head_vol(c)

class SimpleLSTM(nn.Module):
    def __init__(self, n_nodes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_nodes, 64, 2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        if x.dim()==4: x=x.squeeze(-1)
        return self.head(self.lstm(x)[0][:,-1,:])

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
    m = HybridGATLSTM(1, len(FEATURE_NODES), 0.0).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try: m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)); return m, "✅ Trained"
        except: pass
    return m, "⚠️ Random weights"

# =============================================================================
# BACKTESTING STRATEGIES
# =============================================================================
def gen_preds(model, sdf, adj):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(SEQ_LEN, len(sdf)):
            x = torch.FloatTensor(sdf[FEATURE_NODES].iloc[i-SEQ_LEN:i].values).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            p_d,p_r,p_v = model(x,adj)
            preds.append({'prob':torch.sigmoid(p_d).item(),'ret':p_r.item(),'vol':abs(p_v[0,0].item())})
    return preds

def gen_lstm_preds(model, sdf):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(SEQ_LEN, len(sdf)):
            x = torch.FloatTensor(sdf[FEATURE_NODES].iloc[i-SEQ_LEN:i].values).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            preds.append(torch.sigmoid(model(x)).item())
    return preds

def run_ml_strategy(bt_df, probs, config, is_dict=True):
    df = bt_df.copy()
    df['Prob'] = [p['prob'] for p in probs] if is_dict else probs
    df['LogRet'] = np.log(df['close']/df['close'].shift(1))
    df['Vol'] = df['LogRet'].rolling(20).std().shift(1).bfill()
    eq, trades, in_t, pos, ot = config['INITIAL_CAPITAL'], [], False, 0, {}
    df['Equity'] = np.nan; df.iloc[0, df.columns.get_loc('Equity')] = eq
    for i in range(1, len(df)):
        c,p = df.iloc[i], df.iloc[i-1]
        if in_t:
            ex = None
            if ot['dir']==1:
                if c['low']<=ot['sl']: ex=ot['sl']
                elif c['high']>=ot['tp']: ex=ot['tp']
                pnl = (ex-ot['en'])*pos if ex else 0
            else:
                if c['high']>=ot['sl']: ex=ot['sl']
                elif c['low']<=ot['tp']: ex=ot['tp']
                pnl = (ot['en']-ex)*pos if ex else 0
            if ex:
                eq += pnl - ot['lots']*config['COMMISSION_FEE']
                trades.append({'pnl':pnl,'win':pnl>0})
                in_t, pos = False, 0
        if not in_t:
            prob, en, vol = p['Prob'], c['open'], c['Vol']
            d = 1 if prob>=config['CONF_THRESHOLD'] else (-1 if prob<=(1-config['CONF_THRESHOLD']) else 0)
            if d!=0 and not np.isnan(vol) and vol>0:
                sl_d = en*(np.exp(vol*2)-1); max_u = min(eq*config['MAX_RISK_PER_TRADE_PCT']/sl_d, eq*config['LEVERAGE_RATIO'])
                pos = math.floor(max_u/(config['LOT_SIZE']/10))*(config['LOT_SIZE']/10)
                if pos >= config['LOT_SIZE']/10:
                    sl,tp = (en-sl_d, en+sl_d*config['R_R_RATIO']) if d==1 else (en+sl_d, en-sl_d*config['R_R_RATIO'])
                    ot = {'en':en,'dir':d,'sl':sl,'tp':tp,'lots':pos/config['LOT_SIZE']}; in_t=True
        df.loc[c.name,'Equity'] = eq
    return df, pd.DataFrame(trades)

def run_ma_strategy(bt_df, config):
    df = bt_df.copy()
    df['EF'],df['ES'] = df['close'].ewm(50).mean(), df['close'].ewm(200).mean()
    df['LogRet'] = np.log(df['close']/df['close'].shift(1))
    df['Pos'] = np.where(df['EF']>df['ES'],1,-1).astype(float); df['Pos'] = df['Pos'].shift(1).fillna(0)
    df['Ret'] = df['Pos']*df['LogRet'] - df['Pos'].diff().abs().fillna(0)*0.0002
    df['Equity'] = config['INITIAL_CAPITAL']*np.exp(df['Ret'].cumsum())
    return df, pd.DataFrame({'pnl':df[df['Pos'].diff().abs()>0]['Ret']*config['INITIAL_CAPITAL'],'win':df[df['Pos'].diff().abs()>0]['Ret']>0})

def run_bh_strategy(bt_df, config):
    df = bt_df.copy()
    df['LogRet'] = np.log(df['close']/df['close'].shift(1))
    df['Equity'] = config['INITIAL_CAPITAL']*np.exp((df['LogRet']*config['LEVERAGE_RATIO']/100).cumsum())
    return df, pd.DataFrame()

def calc_metrics(df, trades, config, name):
    init,fin,days = config['INITIAL_CAPITAL'], df['Equity'].iloc[-1], len(df)
    tot_r = (fin-init)/init; ann_r = ((fin/init)**(TRADING_DAYS/days))-1
    vol = df['Equity'].pct_change().std()*np.sqrt(TRADING_DAYS); sharpe = ann_r/vol if vol>0 else 0
    dd = ((df['Equity']-df['Equity'].cummax())/df['Equity'].cummax()).min()*100
    wr = (len(trades[trades['pnl']>0])/len(trades)*100) if len(trades)>0 and 'pnl' in trades else 0
    return {'Strategy':name,'Final Equity':fin,'Total Return (%)':tot_r*100,'Annual Return (%)':ann_r*100,
            'Sharpe Ratio':sharpe,'Max Drawdown (%)':dd,'Total Trades':len(trades),'Win Rate (%)':wr}

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    with st.sidebar:
        st.title("🧠 GAT-LSTM"); st.caption("EUR/USD Forecasting")
        st.divider()
        st.subheader("Status")
        st.success("✅ Data") if os.path.exists(DATA_PATH) else st.error("❌ No data")
        model, mstat = load_model(); st.success(mstat) if "✅" in mstat else st.warning(mstat)
        st.divider()
        lookback = st.slider("Chart bars", 100, 1000, 300, 50)
        st.divider(); st.caption("UOW Malaysia KDU Penang"); st.caption("Prof J. Joshua Thomas")

    raw = load_data()
    if raw is None: st.error("Add EURUSD_daily.csv to data/ folder"); st.stop()
    df = engineer_features(raw); sdf, scaler, adj = prepare_data(df)

    st.title("🧠 GAT-LSTM EUR/USD Forecasting System")
    st.markdown('<div class="warning-box">⚠️ <b>Disclaimer:</b> Academic research only. NOT for real trading.</div>', unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["🎯 Prediction","📈 Backtest (4 Strategies)","🔬 Analysis","📚 Docs"])

    with t1:
        st.header("Live Prediction")
        x = torch.FloatTensor(sdf[FEATURE_NODES].iloc[-SEQ_LEN:].values).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        model.eval()
        with torch.no_grad(): p_d,p_r,p_v = model(x,adj)
        prob,ret,vol = torch.sigmoid(p_d).item(), p_r.item(), abs(p_v[0,0].item())
        ann_vol = vol*np.sqrt(252)*100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Price", f"{df['close'].iloc[-1]:.5f}", f"{(df['close'].iloc[-1]/df['close'].iloc[-2]-1)*100:+.2f}%")
        sig = "🟢 BUY" if prob>0.55 else "🔴 SELL" if prob<0.45 else "⚪ NEUTRAL"
        c2.metric("Signal", sig, f"{(prob-0.5)*200:+.1f}%", help="BUY >55%, SELL <45%")
        c3.metric("Probability", f"{prob:.1%}", help="Upward movement probability")
        vlvl = "Low" if ann_vol<10 else "Normal" if ann_vol<20 else "High"
        c4.metric("Volatility", f"{ann_vol:.1f}%", vlvl, help="Annualized. <10%=Low, 10-20%=Normal, >20%=High. Used for stop-loss sizing.")

        with st.expander("ℹ️ Volatility Explained"):
            st.markdown("""
**What is Predicted Volatility?** Next-day price uncertainty forecast.

| Use | Description |
|-----|-------------|
| **Position Sizing** | Higher vol → smaller positions |
| **Stop-Loss** | Vol × 2 = SL distance |
| **Risk Alert** | >20% = high risk |

*Sources: MACD histogram, RSI momentum, rolling std dev*
            """)

        st.subheader("📊 Chart")
        cdf = df.iloc[-lookback:]
        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=cdf.index,open=cdf['open'],high=cdf['high'],low=cdf['low'],close=cdf['close'],name='Price'),row=1,col=1)
        for e,c,n in [(cdf['ema_20'],'yellow','EMA20'),(cdf['ema_50'],'orange','EMA50'),(cdf['ema_200'],'red','EMA200')]:
            fig.add_trace(go.Scatter(x=cdf.index,y=e,line=dict(color=c,width=1),name=n),row=1,col=1)
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf['rsi_14'],line=dict(color='purple'),name='RSI'),row=2,col=1)
        fig.add_hline(y=70,line_dash="dot",line_color="red",row=2); fig.add_hline(y=30,line_dash="dot",line_color="green",row=2)
        fig.add_trace(go.Bar(x=cdf.index,y=cdf['macd_hist'],marker_color=['green' if v>=0 else 'red' for v in cdf['macd_hist']],name='MACD'),row=3,col=1)
        fig.update_layout(height=600,template="plotly_dark",xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

    with t2:
        st.header("📈 4-Strategy Backtest")
        st.markdown('<div class="info-box"><b>Strategies:</b> 1. GAT-LSTM 2. LSTM Baseline 3. MA Crossover (50/200) 4. Buy & Hold</div>', unsafe_allow_html=True)
        
        c1,c2,c3 = st.columns(3)
        cap = c1.number_input("Capital ($)",10000,1000000,100000,10000)
        lev = c1.slider("Leverage",10.0,100.0,100.0,10.0)
        risk = c2.slider("Risk/Trade %",1.0,5.0,2.0,0.5)
        rr = c2.slider("R:R Ratio",1.0,3.0,1.0,0.5)
        conf = c3.slider("Confidence",0.50,0.70,0.50,0.05)
        yrs = c3.slider("Test Years",1,5,2)
        run = st.button("🚀 Run Backtest",use_container_width=True)

        if run:
            cfg = {**CONFIG,'INITIAL_CAPITAL':float(cap),'LEVERAGE_RATIO':lev,'MAX_RISK_PER_TRADE_PCT':risk/100,'R_R_RATIO':rr,'CONF_THRESHOLD':conf}
            tdf,tsdf = df.iloc[-yrs*252:].copy(), sdf.iloc[-yrs*252:].copy()
            
            with st.spinner("GAT-LSTM..."): gp = gen_preds(model,tsdf,adj)
            with st.spinner("LSTM..."): lm = SimpleLSTM(len(FEATURE_NODES)).to(DEVICE); lp = gen_lstm_preds(lm,tsdf)
            
            bt = tdf.iloc[SEQ_LEN:].iloc[:len(gp)].copy(); lp = lp[:len(bt)]
            
            with st.spinner("Running strategies..."):
                g_df,g_tr = run_ml_strategy(bt,gp,cfg,True)
                l_df,l_tr = run_ml_strategy(bt,lp,cfg,False)
                m_df,m_tr = run_ma_strategy(bt,cfg)
                b_df,b_tr = run_bh_strategy(bt,cfg)

            res = pd.DataFrame([calc_metrics(g_df,g_tr,cfg,"GAT-LSTM"),calc_metrics(l_df,l_tr,cfg,"LSTM Baseline"),
                               calc_metrics(m_df,m_tr,cfg,"MA Crossover"),calc_metrics(b_df,b_tr,cfg,"Buy & Hold")]).set_index('Strategy')
            
            st.subheader("📊 Results")
            st.dataframe(res.style.format({'Final Equity':'${:,.0f}','Total Return (%)':'{:+.2f}%','Annual Return (%)':'{:+.2f}%',
                                           'Sharpe Ratio':'{:.3f}','Max Drawdown (%)':'{:.2f}%','Win Rate (%)':'{:.1f}%','Total Trades':'{:.0f}'}),use_container_width=True)
            
            fig = go.Figure()
            for d,n,c in [(g_df,"GAT-LSTM","#00ff00"),(l_df,"LSTM","#00ccff"),(m_df,"MA Cross","#ff9900"),(b_df,"Buy&Hold","gray")]:
                fig.add_trace(go.Scatter(x=d.index,y=d['Equity'],name=n,line=dict(color=c,width=2 if n=="GAT-LSTM" else 1)))
            fig.update_layout(title="Equity Curves",height=450,template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)

    with t3:
        st.header("🔬 Model Analysis")
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Graph Structure")
            fig = px.imshow(adj.cpu().numpy(),x=FEATURE_NODES,y=FEATURE_NODES,color_continuous_scale='Blues',title="Adjacency Matrix")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            st.subheader("Feature Nodes")
            for n in FEATURE_NODES: st.markdown(f"• **{n}**")

    with t4:
        st.header("📚 Documentation")
        st.markdown("""
## GAT-LSTM System
**Hybrid Graph Attention Network + LSTM** for EUR/USD forecasting.

### Architecture
- **GAT Path**: Spatial dependencies between 11 indicators
- **LSTM Path**: Temporal patterns over 30-day sequences  
- **Multi-task**: Direction, Return, Volatility prediction

### Research Targets
| Metric | Target |
|--------|--------|
| Accuracy | >55% |
| RMSE | <0.5 |
| Sharpe | >1.2 |

### Files
```
data/EURUSD_daily.csv   ← Your data
models/gat_lstm_model.pth ← Trained weights
```
        """)

if __name__ == "__main__": main()