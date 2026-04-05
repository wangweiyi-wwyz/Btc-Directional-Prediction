"""
信号生成模块

功能：
1. 简单阈值法：上涨概率大于 0.5 即买入
2. 支持多种信号生成策略
"""

import pandas as pd
import numpy as np
from typing import Optional


class SignalGenerator:
    """
    交易信号生成器
    
    支持的信号生成方法：
    1. 简单阈值法
    2. 概率阈值法
    3. 动量信号法
    """
    
    @staticmethod
    def simple_threshold(
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        简单阈值法
        
        上涨概率 > threshold 则买入（信号=1），否则不持有（信号=0）
        
        参数：
            y_prob: 预测概率
            threshold: 阈值
            
        返回：
            交易信号数组
        """
        return (y_prob > threshold).astype(int)
    
    @staticmethod
    def probability_threshold(
        y_prob: np.ndarray,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4
    ) -> np.ndarray:
        """
        概率阈值法
        
        概率高于买入阈值：多头信号（1）
        概率低于卖出阈值：空头信号（-1）
        其余：空仓（0）
        
        参数：
            y_prob: 预测概率
            buy_threshold: 买入阈值
            sell_threshold: 卖出阈值
            
        返回：
            交易信号数组
        """
        signals = np.zeros(len(y_prob))
        signals[y_prob > buy_threshold] = 1
        signals[y_prob < sell_threshold] = -1
        
        return signals
    
    @staticmethod
    def momentum_signal(
        y_prob: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        动量信号法
        
        基于概率的移动平均生成信号
        
        参数：
            y_prob: 预测概率
            window: 移动平均窗口
            
        返回：
            交易信号数组
        """
        # 计算移动平均
        prob_series = pd.Series(y_prob)
        prob_ma = prob_series.rolling(window=window, min_periods=1).mean()
        
        # 生成信号
        signals = np.zeros(len(y_prob))
        signals[y_prob > prob_ma] = 1
        signals[y_prob < prob_ma] = -1
        
        return signals
    
    @staticmethod
    def confidence_weighted(
        y_prob: np.ndarray,
        min_confidence: float = 0.1
    ) -> np.ndarray:
        """
        置信度加权信号
        
        信号强度 = |概率 - 0.5|，只有置信度超过阈值才生成信号
        
        参数：
            y_prob: 预测概率
            min_confidence: 最小置信度阈值
            
        返回：
            交易信号数组（连续值）
        """
        confidence = np.abs(y_prob - 0.5)
        direction = np.sign(y_prob - 0.5)
        
        # 只有置信度超过阈值才生成信号
        signals = direction * confidence
        signals[confidence < min_confidence] = 0
        
        return signals
    
    def generate_signals(
        self,
        y_prob: np.ndarray,
        method: str = 'simple_threshold',
        **kwargs
    ) -> np.ndarray:
        """
        生成交易信号
        
        参数：
            y_prob: 预测概率
            method: 信号生成方法
            **kwargs: 方法参数
            
        返回：
            交易信号数组
        """
        method_map = {
            'simple_threshold': self.simple_threshold,
            'probability_threshold': self.probability_threshold,
            'momentum': self.momentum_signal,
            'confidence_weighted': self.confidence_weighted
        }
        
        if method not in method_map:
            raise ValueError(f"不支持的信号生成方法: {method}")
        
        return method_map[method](y_prob, **kwargs)
