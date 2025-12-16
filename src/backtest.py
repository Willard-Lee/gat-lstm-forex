"""
Backtesting engine for GAT-LSTM Forex Forecasting model.

Implements event-based backtesting with:
- Risk management (position sizing, stop-loss, take-profit)
- Leverage simulation
- Performance metrics calculation

Usage:
    python backtest.py --model models/gat_lstm_model.pth --data data/EURUSD_daily.csv
"""

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DEVICE, FEATURE_NODES, SEQ_LEN, BACKTEST_CONFIG
)
from src.data_loader import prepare_data_pipeline
from src.graph import build_adjacency_matrix
from src.dataset import ForexGraphDataset
from src.models import HybridGATLSTM

TRADING_DAYS_YEAR = 252


def generate_predictions(
    model: torch.nn.Module,
    dataset: ForexGraphDataset,
    adj_tensor: torch.Tensor,
    device: torch.device
) -> list:
    """
    Generate predictions for the entire dataset.
    
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _, _, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            p_d, p_r, p_v = model(x, adj_tensor)
            
            predictions.append({
                'prob': torch.sigmoid(p_d).item(),
                'pred_ret': p_r.item(),
                'pred_vol': abs(p_v[0, 0].item())
            })
    
    return predictions


def run_backtest(
    bt_df: pd.DataFrame,
    predictions: list,
    config: dict
) -> tuple:
    """
    Run event-based backtesting simulation.
    
    Args:
        bt_df: DataFrame with OHLCV data
        predictions: List of model predictions
        config: Backtest configuration dictionary
        
    Returns:
        Tuple of (results_df, trade_log_df, metrics_dict)
    """
    # Extract config
    INITIAL_CAPITAL = config['INITIAL_CAPITAL']
    COMMISSION_FEE = config['COMMISSION_FEE']
    CONF_THRESHOLD = config['CONF_THRESHOLD']
    MAX_RISK = config['MAX_RISK_PER_TRADE_PCT']
    LEVERAGE = config['LEVERAGE_RATIO']
    LOT_SIZE = config['LOT_SIZE']
    R_R_RATIO = config['R_R_RATIO']
    
    # Prepare data
    bt_df = bt_df.copy()
    bt_df['Model_Prob'] = [p['prob'] for p in predictions]
    bt_df['Log_Return'] = np.log(bt_df['close'] / bt_df['close'].shift(1))
    bt_df['Daily_Vol'] = bt_df['Log_Return'].rolling(window=20).std().shift(1).bfill()
    
    # Initialize state
    equity = INITIAL_CAPITAL
    trade_log = []
    in_trade = False
    position_size = 0
    open_trade = {}
    
    bt_df['Equity_Curve'] = np.nan
    bt_df['Benchmark_Curve'] = np.nan
    bt_df['Position'] = 0
    
    bt_df.iloc[0, bt_df.columns.get_loc('Equity_Curve')] = INITIAL_CAPITAL
    bt_df.iloc[0, bt_df.columns.get_loc('Benchmark_Curve')] = INITIAL_CAPITAL
    
    # Simulation loop
    for i in range(1, len(bt_df)):
        current_row = bt_df.iloc[i]
        prev_row = bt_df.iloc[i-1]
        
        # --- EXIT CHECK ---
        if in_trade:
            high, low = current_row['high'], current_row['low']
            exit_price = None
            
            if open_trade['direction'] == 1:  # LONG
                if low <= open_trade['sl_price']:
                    exit_price = open_trade['sl_price']
                elif high >= open_trade['tp_price']:
                    exit_price = open_trade['tp_price']
                pnl = (exit_price - open_trade['entry_price']) * position_size if exit_price else 0
            else:  # SHORT
                if high >= open_trade['sl_price']:
                    exit_price = open_trade['sl_price']
                elif low <= open_trade['tp_price']:
                    exit_price = open_trade['tp_price']
                pnl = (open_trade['entry_price'] - exit_price) * position_size if exit_price else 0
            
            if exit_price is not None:
                trade_cost = open_trade['lot_size'] * COMMISSION_FEE
                net_pnl = pnl - trade_cost
                equity += net_pnl
                
                trade_log.append({
                    'entry_date': open_trade['entry_date'],
                    'exit_date': current_row.name,
                    'direction': open_trade['direction'],
                    'entry_price': open_trade['entry_price'],
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl_gross': pnl,
                    'pnl_net': net_pnl,
                    'win': net_pnl > 0
                })
                
                in_trade = False
                position_size = 0
        
        # --- ENTRY CHECK ---
        if not in_trade:
            prob = prev_row['Model_Prob']
            entry_price = current_row['open']
            daily_vol = current_row['Daily_Vol']
            
            direction = 0
            if prob >= CONF_THRESHOLD:
                direction = 1
            elif prob <= (1 - CONF_THRESHOLD):
                direction = -1
            
            if direction != 0 and not np.isnan(daily_vol) and daily_vol > 0:
                # Calculate stop-loss distance based on volatility
                sl_distance = entry_price * (np.exp(daily_vol * 2.0) - 1)
                
                # Position sizing based on risk
                max_loss = equity * MAX_RISK
                max_units_risk = max_loss / sl_distance if sl_distance > 0 else 0
                max_units_leverage = equity * LEVERAGE
                
                position_size_raw = min(max_units_risk, max_units_leverage)
                lot_multiple = LOT_SIZE / 10  # Mini lot
                position_size = math.floor(position_size_raw / lot_multiple) * lot_multiple
                
                if position_size >= lot_multiple:
                    if direction == 1:  # LONG
                        sl_price = entry_price - sl_distance
                        tp_price = entry_price + (sl_distance * R_R_RATIO)
                    else:  # SHORT
                        sl_price = entry_price + sl_distance
                        tp_price = entry_price - (sl_distance * R_R_RATIO)
                    
                    open_trade = {
                        'entry_date': current_row.name,
                        'entry_price': entry_price,
                        'direction': direction,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'lot_size': position_size / LOT_SIZE
                    }
                    in_trade = True
                    bt_df.loc[current_row.name, 'Position'] = direction
        
        # Update curves
        bt_df.loc[current_row.name, 'Equity_Curve'] = equity
        benchmark_ret = current_row['close'] / prev_row['close'] - 1
        bt_df.loc[current_row.name, 'Benchmark_Curve'] = (
            bt_df.loc[prev_row.name, 'Benchmark_Curve'] * (1 + benchmark_ret)
        )
    
    # Create trade log DataFrame
    trade_log_df = pd.DataFrame(trade_log)
    
    # Calculate metrics
    metrics = calculate_backtest_metrics(bt_df, trade_log_df, config)
    
    return bt_df, trade_log_df, metrics


def calculate_backtest_metrics(
    bt_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    config: dict
) -> dict:
    """
    Calculate comprehensive backtest performance metrics.
    """
    INITIAL_CAPITAL = config['INITIAL_CAPITAL']
    
    total_days = len(bt_df)
    final_equity = bt_df['Equity_Curve'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Annualized return
    annual_return = ((final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_YEAR / total_days)) - 1
    
    # Volatility and Sharpe
    daily_returns = bt_df['Equity_Curve'].pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    max_equity = bt_df['Equity_Curve'].cummax()
    drawdown = (bt_df['Equity_Curve'] - max_equity) / max_equity
    max_dd = drawdown.min() * 100
    
    metrics = {
        'Final_Equity': final_equity,
        'Total_Return_Pct': total_return * 100,
        'Annual_Return_Pct': annual_return * 100,
        'Annual_Volatility_Pct': annual_vol * 100,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown_Pct': max_dd,
        'Total_Trades': len(trade_log)
    }
    
    if len(trade_log) > 0:
        wins = trade_log[trade_log['pnl_net'] > 0]
        losses = trade_log[trade_log['pnl_net'] <= 0]
        
        metrics['Win_Rate_Pct'] = (len(wins) / len(trade_log)) * 100
        metrics['Avg_Win'] = wins['pnl_net'].mean() if len(wins) > 0 else 0
        metrics['Avg_Loss'] = abs(losses['pnl_net'].mean()) if len(losses) > 0 else 0
        metrics['Achieved_RR'] = metrics['Avg_Win'] / metrics['Avg_Loss'] if metrics['Avg_Loss'] > 0 else 0
        metrics['Total_PnL'] = trade_log['pnl_net'].sum()
        
        # Long/Short breakdown
        longs = trade_log[trade_log['direction'] == 1]
        shorts = trade_log[trade_log['direction'] == -1]
        metrics['Long_PnL'] = longs['pnl_net'].sum() if len(longs) > 0 else 0
        metrics['Short_PnL'] = shorts['pnl_net'].sum() if len(shorts) > 0 else 0
    else:
        metrics['Win_Rate_Pct'] = 0
        metrics['Avg_Win'] = 0
        metrics['Avg_Loss'] = 0
        metrics['Achieved_RR'] = 0
        metrics['Total_PnL'] = 0
        metrics['Long_PnL'] = 0
        metrics['Short_PnL'] = 0
    
    return metrics


def print_backtest_report(metrics: dict, config: dict):
    """Print formatted backtest report."""
    print("\n" + "=" * 80)
    print("BACKTESTING PERFORMANCE REPORT")
    print("=" * 80)
    
    print(f"\n📋 Configuration")
    print("-" * 60)
    print(f"   Initial Capital:    ${config['INITIAL_CAPITAL']:,.2f}")
    print(f"   Leverage:           {config['LEVERAGE_RATIO']}:1")
    print(f"   Max Risk/Trade:     {config['MAX_RISK_PER_TRADE_PCT']*100:.1f}%")
    print(f"   R/R Ratio:          {config['R_R_RATIO']}")
    print(f"   Confidence Threshold: {config['CONF_THRESHOLD']}")
    
    print(f"\n📊 Performance Summary")
    print("-" * 60)
    print(f"   Final Equity:       ${metrics['Final_Equity']:,.2f}")
    print(f"   Total Return:       {metrics['Total_Return_Pct']:.2f}%")
    print(f"   Annual Return:      {metrics['Annual_Return_Pct']:.2f}%")
    print(f"   Sharpe Ratio:       {metrics['Sharpe_Ratio']:.3f}")
    print(f"   Max Drawdown:       {metrics['Max_Drawdown_Pct']:.2f}%")
    
    print(f"\n📈 Trade Statistics")
    print("-" * 60)
    print(f"   Total Trades:       {metrics['Total_Trades']}")
    print(f"   Win Rate:           {metrics['Win_Rate_Pct']:.1f}%")
    print(f"   Avg Win:            ${metrics['Avg_Win']:,.2f}")
    print(f"   Avg Loss:           ${metrics['Avg_Loss']:,.2f}")
    print(f"   Achieved R/R:       {metrics['Achieved_RR']:.2f}")
    print(f"   Long PnL:           ${metrics['Long_PnL']:,.2f}")
    print(f"   Short PnL:          ${metrics['Short_PnL']:,.2f}")
    
    print("=" * 80)


def plot_equity_curve(bt_df: pd.DataFrame, save_path: str = None):
    """Plot equity curve comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(bt_df.index, bt_df['Equity_Curve'], 
            label='GAT-LSTM Strategy', color='#00ff00', linewidth=2)
    ax.plot(bt_df.index, bt_df['Benchmark_Curve'], 
            label='Buy & Hold', color='gray', linewidth=1, linestyle='--')
    
    ax.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Equity curve saved to: {save_path}")
    
    plt.show()


def backtest(
    model_path: str,
    data_path: str,
    config: dict = None
):
    """
    Main backtesting function.
    """
    if config is None:
        config = BACKTEST_CONFIG
    
    print("Loading data...")
    train_df, val_df, test_df, scaler = prepare_data_pipeline(data_path, save_scaler=False)
    
    print("Building graph...")
    adj_tensor = build_adjacency_matrix(train_df, FEATURE_NODES)
    
    print("Loading model...")
    model = HybridGATLSTM(nfeat=1, n_nodes=len(FEATURE_NODES), dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print("Generating predictions...")
    test_dataset = ForexGraphDataset(test_df, FEATURE_NODES, SEQ_LEN)
    predictions = generate_predictions(model, test_dataset, adj_tensor, DEVICE)
    
    # Align data
    bt_df = test_df.iloc[SEQ_LEN:].copy()
    bt_df = bt_df.iloc[:len(predictions)]
    
    print("Running backtest...")
    results_df, trade_log, metrics = run_backtest(bt_df, predictions, config)
    
    print_backtest_report(metrics, config)
    plot_equity_curve(results_df)
    
    return results_df, trade_log, metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest GAT-LSTM model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--leverage', type=float, default=100.0,
                        help='Leverage ratio')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Max risk per trade')
    
    args = parser.parse_args()
    
    config = BACKTEST_CONFIG.copy()
    config['LEVERAGE_RATIO'] = args.leverage
    config['MAX_RISK_PER_TRADE_PCT'] = args.risk
    
    backtest(
        model_path=args.model,
        data_path=args.data,
        config=config
    )


if __name__ == "__main__":
    main()
