"""
评估指标计算模块

包含：
1. 回归指标：MSE, RMSE, MAE, R²
2. 分类指标：Accuracy, F1, AUC, Precision, Recall
3. 交易回测指标：Sharpe, 收益率, 回撤
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


class MetricsCalculator:
    """
    评估指标计算器
    
    支持的指标：
    1. 回归指标：MSE, RMSE, MAE, R²
    2. 分类指标：Accuracy, F1, AUC, Precision, Recall
    3. 交易回测指标：年化收益率、年化波动率、夏普比率、最大回撤
    """
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归评估指标
        
        参数：
            y_true: 真实值
            y_pred: 预测值
            
        返回：
            包含 MSE, RMSE, MAE, R² 的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算分类评估指标
        
        参数：
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（用于计算 AUC）
            
        返回：
            包含 Accuracy, F1, AUC, Precision, Recall 的字典
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['AUC'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_returns(returns: np.ndarray, annualize: bool = True) -> float:
        """
        计算收益率
        
        参数：
            returns: 收益率序列
            annualize: 是否年化
            
        返回：
            收益率
        """
        total_return = (1 + returns).prod() - 1
        
        if annualize:
            # 假设252个交易日/年
            n_years = len(returns) / 252
            annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
            return annualized_return
        
        return total_return
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """
        计算波动率
        
        参数：
            returns: 收益率序列
            annualize: 是否年化
            
        返回：
            波动率
        """
        volatility = returns.std()
        
        if annualize:
            # 年化波动率
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        计算夏普比率
        
        参数：
            returns: 收益率序列
            risk_free_rate: 无风险利率
            annualize: 是否年化
            
        返回：
            夏普比率
        """
        excess_returns = returns - risk_free_rate / 252 if annualize else returns - risk_free_rate
        
        if annualize:
            # 年化夏普比率
            sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)
        else:
            sharpe = excess_returns.mean() / (excess_returns.std() + 1e-8)
        
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        计算最大回撤
        
        参数：
            returns: 收益率序列
            
        返回：
            最大回撤
        """
        # 计算累计收益
        cumulative = (1 + returns).cumprod()
        
        # 计算滚动最大值（使用 numpy 的 cummax）
        rolling_max = np.maximum.accumulate(cumulative)
        
        # 计算回撤
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # 最大回撤
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    @staticmethod
    def backtest_metrics(
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        计算交易回测指标
        
        参数：
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        返回：
            包含年化收益率、年化波动率、夏普比率、最大回撤的字典
        """
        return {
            'Annualized_Return': MetricsCalculator.calculate_returns(returns, annualize=True),
            'Annualized_Volatility': MetricsCalculator.calculate_volatility(returns, annualize=True),
            'Sharpe_Ratio': MetricsCalculator.calculate_sharpe_ratio(returns, risk_free_rate, annualize=True),
            'Max_Drawdown': MetricsCalculator.calculate_max_drawdown(returns)
        }
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        计算所有评估指标
        
        参数：
            y_true: 真实值/标签
            y_pred: 预测值/标签
            y_prob: 预测概率
            returns: 收益率序列（用于回测指标）
            
        返回：
            包含所有指标的嵌套字典
        """
        results = {}
        
        # 分类指标
        results['classification'] = MetricsCalculator.classification_metrics(
            y_true, y_pred, y_prob
        )
        
        # 回测指标
        if returns is not None:
            results['backtest'] = MetricsCalculator.backtest_metrics(returns)
        
        return results
    
    @staticmethod
    def print_metrics_report(metrics: Dict[str, Dict[str, float]]) -> None:
        """
        打印评估指标报告
        
        参数：
            metrics: 评估指标字典
        """
        print("=" * 50)
        print("分类评估指标")
        print("=" * 50)
        
        if 'classification' in metrics:
            for metric, value in metrics['classification'].items():
                print(f"{metric}: {value:.4f}")
        
        print("\n" + "=" * 50)
        print("交易回测指标")
        print("=" * 50)
        
        if 'backtest' in metrics:
            for metric, value in metrics['backtest'].items():
                if metric == 'Max_Drawdown':
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.2%}")
        
        print("=" * 50)
