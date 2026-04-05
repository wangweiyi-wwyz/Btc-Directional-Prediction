"""
回测引擎模块

功能：
1. 执行交易回测
2. 计算回测指标：年化收益率、年化波动率、夏普比率、最大回撤
3. 生成回测报告和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.dates as mdates

from .signal_generator import SignalGenerator
from .strategy import TradingStrategy
from ..evaluation.metrics import MetricsCalculator


class Backtester:
    """
    交易回测引擎
    
    负责：
    1. 信号生成
    2. 策略执行
    3. 回测指标计算
    4. 回测报告生成
    """
    
    # 所有支持的策略列表
    ALL_STRATEGIES = ['buy_and_hold', 'long_only', 'short_only', 'long_short', 'equal_weighted']
    
    def __init__(self, output_dir: str = "figures", split_type: Optional[str] = None):
        """
        初始化回测引擎
        
        参数：
            output_dir: 图表输出目录
            split_type: 数据划分类型（如 "fixed"、"rolling"、"regime"），用于标识回测场景
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.signal_generator = SignalGenerator()
        self.strategy = TradingStrategy()
        self.metrics_calculator = MetricsCalculator()
        
        self.backtest_results: Dict[str, Dict] = {}
        self.split_type = split_type  # 数据划分类型标识
    
    def run_backtest(
        self,
        y_prob: np.ndarray,
        price_data: pd.Series,
        strategy_name: str = 'long_only',
        signal_method: str = 'simple_threshold',
        signal_threshold: float = 0.5,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        运行回测
        
        参数：
            y_prob: 预测概率
            price_data: 价格数据
            strategy_name: 策略名称
            signal_method: 信号生成方法
            signal_threshold: 信号阈值
            initial_capital: 初始资金
            risk_free_rate: 无风险利率
            
        返回：
            回测结果字典
        """
        # 计算收益率
        returns = price_data.pct_change().fillna(0).values
        
        # 确保长度匹配
        min_len = min(len(y_prob), len(returns))
        y_prob = y_prob[:min_len]
        returns = returns[:min_len]
        price_data = price_data.iloc[:min_len]
        
        # 生成交易信号
        signals = self.signal_generator.generate_signals(
            y_prob,
            method=signal_method,
            threshold=signal_threshold
        )
        
        # 修复：信号需要向前平移一天，因为当日信号只能在次日交易
        # 这避免了使用未来数据（当日收盘后才能生成信号）
        signals = pd.Series(signals).shift(1).fillna(0).values
        
        # 应用策略
        strategy_returns = self.strategy.apply_strategy(strategy_name, signals, returns)
        
        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        # 保留时间索引，使图表横轴显示日期
        equity_curve = pd.Series(
            initial_capital * cumulative_returns,
            index=price_data.index
        )
        
        # 计算回测指标
        metrics = self.metrics_calculator.backtest_metrics(strategy_returns, risk_free_rate)
        
        # 计算额外指标
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) if isinstance(equity_curve, pd.Series) else (equity_curve[-1] / initial_capital - 1)
        
        # 保存结果
        result = {
            'strategy_name': strategy_name,
            'signal_method': signal_method,
            'signals': signals,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'total_return': total_return,
            'initial_capital': initial_capital,
            'final_capital': equity_curve.iloc[-1] if isinstance(equity_curve, pd.Series) else equity_curve[-1]
        }
        
        return result
    
    def compare_strategies(
        self,
        y_prob: np.ndarray,
        price_data: pd.Series,
        strategies: Optional[List[str]] = None,
        signal_method: str = 'simple_threshold',
        signal_threshold: float = 0.5,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Dict]:
        """
        比较多个策略
        
        参数：
            y_prob: 预测概率
            price_data: 价格数据
            strategies: 策略列表
            signal_method: 信号生成方法
            signal_threshold: 信号阈值
            initial_capital: 初始资金
            risk_free_rate: 无风险利率
            
        返回：
            策略回测结果字典
        """
        if strategies is None:
            strategies = ['buy_and_hold', 'long_only', 'short_only', 'long_short']
        
        results = {}
        
        for strategy_name in strategies:
            try:
                result = self.run_backtest(
                    y_prob=y_prob,
                    price_data=price_data,
                    strategy_name=strategy_name,
                    signal_method=signal_method,
                    signal_threshold=signal_threshold,
                    initial_capital=initial_capital,
                    risk_free_rate=risk_free_rate
                )
                results[strategy_name] = result
            except Exception as e:
                print(f"策略 {strategy_name} 回测失败: {e}")
        
        self.backtest_results.update(results)
        
        return results
    
    def print_backtest_report(self, results: Dict[str, Dict]) -> None:
        """
        打印回测报告
        
        参数：
            results: 回测结果字典
        """
        print("\n" + "=" * 70)
        print("交易回测报告")
        print("=" * 70)
        
        for strategy_name, result in results.items():
            print(f"\n策略: {strategy_name}")
            print("-" * 70)
            print(f"  初始资金: ${result['initial_capital']:,.2f}")
            print(f"  最终资金: ${result['final_capital']:,.2f}")
            print(f"  总收益率: {result['total_return']:.2%}")
            print(f"\n  回测指标:")
            for metric, value in result['metrics'].items():
                if metric == 'Max_Drawdown':
                    print(f"    {metric}: {value:.2%}")
                else:
                    print(f"    {metric}: {value:.2%}")
        
        print("\n" + "=" * 70)
    

    def plot_equity_curves(
        self,
        results: Dict[str, Dict],
        save: bool = True,
        filename: str = "equity_curves.png",
        model_name_prefix: str = ""
        ) -> None:
        """
        绘制资金曲线
        
        参数：
            results: 回测结果字典
            save: 是否保存图表
            filename: 基础文件名
            model_name_prefix: 模型名称前缀 (用于区分不同模型的输出)
        """
        # 检查是否有结果
        if not results:
            print("没有回测结果，无法绘制资金曲线")
            return
        
        plt.figure(figsize=(12, 6))
        
        has_data = False
        for strategy_name, result in results.items():
            equity = result.get('equity_curve')
            if equity is not None:
                has_data = True
                
                # 构建带指标的图例标签
                metrics = result.get('metrics', {})
                ann_ret = metrics.get('Annualized_Return', 0)
                sharpe = metrics.get('Sharpe_Ratio', 0)
                max_dd = metrics.get('Max_Drawdown', 0)
                label = f"{strategy_name} (Ann.Ret: {ann_ret:.1%}, Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.1%})"
                
                if isinstance(equity, pd.Series):
                    equity.plot(label=label)
                else:
                    plt.plot(equity, label=label)
        
        if not has_data:
            print("没有有效的资金曲线数据")
            plt.close()
            return
        
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        # 建议在标题中也加上模型名称
        title_prefix = f"[{model_name_prefix}] " if model_name_prefix else ""
        plt.title(f'{title_prefix}Strategy Equity Curves Comparison')
        
        if has_data:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save:
            prefix = f"{model_name_prefix}_" if model_name_prefix else ""
            final_filename = f"{prefix}{filename}"
            
            # 使用新的文件名保存
            plt.savefig(self.output_dir / final_filename, dpi=300)
            print(f"资金曲线已保存至: {self.output_dir / final_filename}")
        
        plt.show()
        plt.close()
        
    
    def plot_drawdown(
        self,
        result: Dict,
        save: bool = True,
        filename: str = "drawdown.png",
        model_name_prefix: str = ""
    ) -> None:
        """
        绘制回撤图
        
        参数：
            result: 回测结果
            save: 是否保存图表
            filename: 文件名
        """
        equity = result['equity_curve']
        if isinstance(equity, pd.Series):
            cumulative = equity.values
        else:
            cumulative = equity
        
        # 计算回撤
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, color='red')
        plt.plot(drawdown * 100, color='red')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.title(f"Drawdown - {result['strategy_name']}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / filename, dpi=300)
        
        plt.show()
    

    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict],
        save: bool = True,
        filename: str = "strategy_comparison.png",
        model_name_prefix: str = "" 
    ) -> None:
        """
        绘制策略比较图
        
        参数：
            results: 回测结果字典
            save: 是否保存图表
            filename: 文件名
            model_name_prefix: 模型名称前缀 # <--- 新增注释
        """
        # 检查是否有结果
        if not results:
            print("没有回测结果，无法绘制策略比较图")
            return
        
        # 准备数据... (中间代码保持不变)
        metrics_data = []
        for strategy_name, result in results.items():
            row = {'Strategy': strategy_name}
            metrics = result.get('metrics', {})
            row.update(metrics)
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # 检查是否有数据和必需的列... (中间代码保持不变)
        if df.empty or len(df) == 0:
            print("回测结果为空，无法绘制策略比较图")
            return
            
        required_columns = ['Annualized_Return', 'Annualized_Volatility', 'Sharpe_Ratio', 'Max_Drawdown']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"缺少必需的指标列: {missing_columns}")
            print(f"可用的列: {df.columns.tolist()}")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        title_prefix = f"[{model_name_prefix}] " if model_name_prefix else ""
        fig.suptitle(f'{title_prefix}Strategy Comparison Metrics', fontsize=16)

        # 绘制4个子图... (绘图代码保持不变)
        # 年化收益率
        ax1 = axes[0, 0]
        df.plot(x='Strategy', y='Annualized_Return', kind='bar', ax=ax1, color='steelblue', legend=False)
        ax1.set_ylabel('Return')
        ax1.set_title('Annualized Return')
        ax1.tick_params(axis='x', rotation=45)
        
        # 年化波动率
        ax2 = axes[0, 1]
        df.plot(x='Strategy', y='Annualized_Volatility', kind='bar', ax=ax2, color='coral', legend=False)
        ax2.set_ylabel('Volatility')
        ax2.set_title('Annualized Volatility')
        ax2.tick_params(axis='x', rotation=45)
        
        # 夏普比率
        ax3 = axes[1, 0]
        df.plot(x='Strategy', y='Sharpe_Ratio', kind='bar', ax=ax3, color='seagreen', legend=False)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 最大回撤
        ax4 = axes[1, 1]
        df.plot(x='Strategy', y='Max_Drawdown', kind='bar', ax=ax4, color='crimson', legend=False)
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Maximum Drawdown')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        # 调整布局以适应可能的主标题
        fig.subplots_adjust(top=0.92) 
        
        if save:
            # === 构建带前缀的最终文件名 ===
            prefix = f"{model_name_prefix}_" if model_name_prefix else ""
            final_filename = f"{prefix}{filename}"
            
            # 使用新的文件名保存
            plt.savefig(self.output_dir / final_filename, dpi=300)
            print(f"策略比较图已保存至: {self.output_dir / final_filename}")
        
        plt.show()
        plt.close() 
    
    def get_backtest_summary(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        获取回测摘要表
        
        参数：
            results: 回测结果字典
            
        返回：
            摘要 DataFrame
        """
        summary_data = []
        
        for strategy_name, result in results.items():
            row = {
                'Strategy': strategy_name,
                'Total_Return': result['total_return'],
                **result['metrics']
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df
    
    def get_all_results(self) -> Dict[str, Dict]:
        """
        获取所有回测结果
        
        返回：
            回测结果字典
        """
        return self.backtest_results
