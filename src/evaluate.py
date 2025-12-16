"""
Evaluation script for GAT-LSTM Forex Forecasting model.

Computes comprehensive statistical and financial performance metrics
aligned with dissertation research objectives.

Usage:
    python evaluate.py --model models/gat_lstm_model.pth --data data/EURUSD_daily.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, roc_auc_score,
    confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DEVICE, FEATURE_NODES, SEQ_LEN,
    TARGET_ACCURACY, TARGET_RMSE, TARGET_SHARPE, TARGET_AUC,
    METRICS_OUTPUT_PATH
)
from src.data_loader import prepare_data_pipeline
from src.graph import build_adjacency_matrix
from src.dataset import create_data_loaders
from src.models import HybridGATLSTM


def calculate_statistical_metrics(
    model: nn.Module,
    test_loader,
    adj_tensor: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Calculate comprehensive statistical performance metrics.
    
    Returns:
        Dictionary containing all statistical metrics
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_actuals_dir = []
    all_preds_ret = []
    all_actuals_ret = []
    
    with torch.no_grad():
        for x, y_d, y_r, y_v in test_loader:
            x = x.to(device)
            
            p_d, p_r, p_v = model(x, adj_tensor)
            
            probs = torch.sigmoid(p_d).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_actuals_dir.extend(y_d.numpy().flatten())
            all_preds_ret.extend(p_r.cpu().numpy().flatten())
            all_actuals_ret.extend(y_r.numpy().flatten())
    
    # Convert to arrays
    y_true = np.array(all_actuals_dir)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_ret_true = np.array(all_actuals_ret)
    y_ret_pred = np.array(all_preds_ret)
    
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_ret_true, y_ret_pred))
    mae = mean_absolute_error(y_ret_true, y_ret_pred)
    
    # Directional accuracy for returns
    directional_acc = np.mean((y_ret_pred > 0) == (y_ret_true > 0))
    
    # Statistical significance test
    n_samples = len(y_true)
    n_correct = int(accuracy * n_samples)
    
    # Z-test against 50% random baseline
    z_score = (accuracy - 0.5) / np.sqrt(0.5 * 0.5 / n_samples)
    p_value = 1 - stats.norm.cdf(z_score)
    
    # Binomial test
    binom_result = stats.binomtest(n_correct, n_samples, 0.5, alternative='greater')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_acc,
        'n_samples': n_samples,
        'z_score': z_score,
        'p_value': p_value,
        'binom_p_value': binom_result.pvalue,
        'statistically_significant': p_value < 0.05,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def check_dissertation_targets(metrics: dict) -> dict:
    """
    Check if dissertation research targets are achieved.
    
    Returns:
        Dictionary with target achievement status
    """
    targets = {
        'Accuracy > 55%': {
            'achieved': metrics['accuracy'] > TARGET_ACCURACY,
            'value': f"{metrics['accuracy']*100:.2f}%",
            'target': f"> {TARGET_ACCURACY*100:.0f}%"
        },
        'RMSE < 0.5': {
            'achieved': metrics['rmse'] < TARGET_RMSE,
            'value': f"{metrics['rmse']:.6f}",
            'target': f"< {TARGET_RMSE}"
        },
        'AUC-ROC > 0.55': {
            'achieved': metrics['auc_roc'] > TARGET_AUC,
            'value': f"{metrics['auc_roc']:.4f}",
            'target': f"> {TARGET_AUC}"
        },
        'Statistical Significance (p < 0.05)': {
            'achieved': metrics['p_value'] < 0.05,
            'value': f"p = {metrics['p_value']:.6f}",
            'target': "p < 0.05"
        }
    }
    
    return targets


def print_evaluation_report(metrics: dict, targets: dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 80)
    print("GAT-LSTM MODEL EVALUATION REPORT")
    print("=" * 80)
    
    print("\n📊 STATISTICAL PERFORMANCE METRICS")
    print("-" * 60)
    print(f"   Direction Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"   F1 Score:               {metrics['f1_score']:.4f}")
    print(f"   Precision:              {metrics['precision']:.4f}")
    print(f"   Recall:                 {metrics['recall']:.4f}")
    print(f"   AUC-ROC:                {metrics['auc_roc']:.4f}")
    print(f"   RMSE (Returns):         {metrics['rmse']:.6f}")
    print(f"   MAE (Returns):          {metrics['mae']:.6f}")
    print(f"   Directional Accuracy:   {metrics['directional_accuracy']*100:.2f}%")
    
    print("\n📈 STATISTICAL SIGNIFICANCE")
    print("-" * 60)
    print(f"   Test Samples:           {metrics['n_samples']}")
    print(f"   Z-Score (vs 50%):       {metrics['z_score']:.4f}")
    print(f"   P-Value:                {metrics['p_value']:.6f}")
    print(f"   Significant (α=0.05):   {'Yes ✓' if metrics['statistically_significant'] else 'No ✗'}")
    
    print("\n🎯 DISSERTATION TARGET ACHIEVEMENT")
    print("-" * 60)
    
    n_achieved = 0
    for target_name, target_info in targets.items():
        status = "✓ PASS" if target_info['achieved'] else "✗ FAIL"
        n_achieved += target_info['achieved']
        print(f"   {target_name}")
        print(f"      Target: {target_info['target']}")
        print(f"      Achieved: {target_info['value']} [{status}]")
    
    print(f"\n   Overall: {n_achieved}/{len(targets)} targets achieved")
    
    print("\n📋 CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    print(f"                 Predicted")
    print(f"              DOWN    UP")
    print(f"   Actual DOWN  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"   Actual UP    {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    print("\n📋 CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(
        metrics['y_true'], metrics['y_pred'],
        target_names=['DOWN (0)', 'UP (1)']
    ))
    
    print("=" * 80)


def save_metrics(metrics: dict, targets: dict, output_path: str = METRICS_OUTPUT_PATH):
    """Save metrics to CSV file."""
    rows = []
    
    # Statistical metrics
    stat_metrics = [
        ('accuracy', 'Accuracy', metrics['accuracy']),
        ('f1_score', 'F1 Score', metrics['f1_score']),
        ('precision', 'Precision', metrics['precision']),
        ('recall', 'Recall', metrics['recall']),
        ('auc_roc', 'AUC-ROC', metrics['auc_roc']),
        ('rmse', 'RMSE', metrics['rmse']),
        ('mae', 'MAE', metrics['mae']),
        ('directional_accuracy', 'Directional Accuracy', metrics['directional_accuracy']),
        ('z_score', 'Z-Score', metrics['z_score']),
        ('p_value', 'P-Value', metrics['p_value']),
    ]
    
    for key, name, value in stat_metrics:
        rows.append({'category': 'Statistical', 'metric': name, 'value': value})
    
    # Target achievements
    for target_name, target_info in targets.items():
        rows.append({
            'category': 'Target',
            'metric': target_name,
            'value': 1 if target_info['achieved'] else 0
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Metrics saved to: {output_path}")


def evaluate(
    model_path: str,
    data_path: str,
    output_path: str = METRICS_OUTPUT_PATH
):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to trained model weights
        data_path: Path to CSV data file
        output_path: Path to save metrics CSV
    """
    print("Loading data...")
    train_df, val_df, test_df, scaler = prepare_data_pipeline(data_path, save_scaler=False)
    
    print("Building graph...")
    adj_tensor = build_adjacency_matrix(train_df, FEATURE_NODES)
    
    print("Creating data loaders...")
    _, _, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        shuffle_train=False
    )
    
    print("Loading model...")
    model = HybridGATLSTM(nfeat=1, n_nodes=len(FEATURE_NODES), dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print("Evaluating model...")
    metrics = calculate_statistical_metrics(model, test_loader, adj_tensor, DEVICE)
    targets = check_dissertation_targets(metrics)
    
    print_evaluation_report(metrics, targets)
    save_metrics(metrics, targets, output_path)
    
    return metrics, targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate GAT-LSTM model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--output', type=str, default=METRICS_OUTPUT_PATH,
                        help='Path to save metrics CSV')
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
