"""
交易策略模块

功能：
1. 仅做多策略
2. 仅做空策略
3. 多空双向策略
4. 买入并持有基准策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class TradingStrategy:
    """
    交易策略类
    
    支持的策略：
    1. 仅做多：正信号时持有多头，否则空仓
    2. 仅做空：负信号时持有空头，否则空仓
    3. 多空双向：按信号方向做多、做空或空仓
    4. 买入并持有：全程持有标的作为基准
    """
    
    @staticmethod
    def long_only(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        仅做多策略
        
        只在信号为 1 时持有，信号为 0 或 -1 时空仓
        
        参数：
            signals: 交易信号数组
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        # 只做多：信号为1时持有，否则空仓
        position = (signals > 0).astype(float)
        strategy_returns = position * returns
        
        return strategy_returns
    
    @staticmethod
    def short_only(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        仅做空策略
        
        只在信号为 -1 时做空，信号为 0 或 1 时空仓
        
        参数：
            signals: 交易信号数组
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        # 只做空：信号为-1时做空，否则空仓
        position = -(signals < 0).astype(float)
        strategy_returns = position * returns
        
        return strategy_returns
    
    @staticmethod
    def long_short(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        多空双向策略
        
        信号为 1 时做多，信号为 -1 时做空，信号为 0 时空仓
        
        参数：
            signals: 交易信号数组
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        # 多空双向：信号为1做多，-1做空，0空仓
        strategy_returns = signals * returns
        
        return strategy_returns
    
    @staticmethod
    def buy_and_hold(returns: np.ndarray) -> np.ndarray:
        """
        买入并持有基准策略
        
        始终持有标的
        
        参数：
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        return returns
    
    @staticmethod
    def equal_weighted(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        等权重策略
        
        无论信号如何，始终以 50% 仓位跟踪标的收益
        
        参数：
            signals: 交易信号数组（未使用）
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        return 0.5 * returns
    
    def apply_strategy(
        self,
        strategy_name: str,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        应用指定策略
        
        参数：
            strategy_name: 策略名称
            signals: 交易信号数组
            returns: 资产收益率数组
            
        返回：
            策略收益率数组
        """
        if strategy_name == 'buy_and_hold':
            # buy_and_hold 仅需收益率序列
            return self.buy_and_hold(returns)
        elif strategy_name == 'equal_weighted':
            # equal_weighted 仅需收益率序列（signals 形参未使用）
            return self.equal_weighted(signals, returns)
        elif strategy_name == 'long_only':
            return self.long_only(signals, returns)
        elif strategy_name == 'short_only':
            return self.short_only(signals, returns)
        elif strategy_name == 'long_short':
            return self.long_short(signals, returns)
        else:
            raise ValueError(f"不支持的策略: {strategy_name}")
    
    def compare_strategies(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        strategies: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        比较多个策略
        
        参数：
            signals: 交易信号数组
            returns: 资产收益率数组
            strategies: 要比较的策略列表
            
        返回：
            策略收益率字典
        """
        if strategies is None:
            strategies = ['buy_and_hold', 'long_only', 'short_only', 'long_short']
        
        results = {}
        
        for strategy_name in strategies:
            results[strategy_name] = self.apply_strategy(strategy_name, signals, returns)
        
        return results
