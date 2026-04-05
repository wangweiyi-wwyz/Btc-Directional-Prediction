"""
数据划分模块（Data Splitter）

功能：
1. 固定 20/80 时间切分
2. 滚动时间窗口切分
3. 牛熊市周期切分

硬性要求：
- 禁止随机打乱，保持时间顺序
- 各折严格隔离训练集与测试集
- 防止泄露：特征标准化须在划分之后进行
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SplitResult:
    """单次数据划分结果。"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    split_info: Dict[str, Any]


class BaseSplitter(ABC):
    """数据划分器基类。"""
    
    def __init__(self, min_train_samples: int = 252):
        """
        初始化划分器。

        参数:
            min_train_samples: 最小训练样本数，默认 252（约一年交易日）
        """
        self.min_train_samples = min_train_samples
    
    @abstractmethod
    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Iterator[SplitResult]:
        """
        产生划分结果迭代器。

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
            dates: 日期索引

        产出:
            SplitResult: 每一折的训练/测试划分
        """
        pass
    
    def _validate_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """校验划分形状与样本量是否合法。"""
        if len(X_train) < self.min_train_samples:
            raise ValueError(
                f"训练集样本数 {len(X_train)} 小于最小要求 {self.min_train_samples}"
            )
        if len(X_test) == 0:
            raise ValueError("测试集为空")
        if len(X_train) != len(y_train):
            raise ValueError("训练集 X 与 y 长度不一致")
        if len(X_test) != len(y_test):
            raise ValueError("测试集 X 与 y 长度不一致")


class FixedTimeSplit(BaseSplitter):
    """
    按时间顺序的固定 20/80 划分。

    前 80% 样本为训练集，后 20% 为测试集。
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        min_train_samples: int = 252
    ):
        """
        初始化固定比例划分器。

        参数:
            train_ratio: 训练集占比，默认 0.8
            min_train_samples: 最小训练样本数
        """
        super().__init__(min_train_samples=min_train_samples)
        self.train_ratio = train_ratio
    
    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Iterator[SplitResult]:
        """
        产生一次固定比例划分。

        参数:
            X: 特征
            y: 目标
            dates: 日期索引

        产出:
            SplitResult: 单次划分结果
        """
        n_samples = len(X)
        train_end = int(n_samples * self.train_ratio)
        
        # 保证训练段足够长
        if train_end < self.min_train_samples:
            train_end = self.min_train_samples
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:]
        y_test = y[train_end:]
        
        # 校验划分
        self._validate_split(X_train, y_train, X_test, y_test)
        
        # 构造日期索引
        if dates is not None:
            train_index = dates[:train_end]
            test_index = dates[train_end:]
        else:
            train_index = pd.RangeIndex(start=0, stop=train_end)
            test_index = pd.RangeIndex(start=train_end, stop=n_samples)
        
        split_info = {
            'split_type': 'fixed',
            'train_ratio': self.train_ratio,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_start': str(train_index[0]) if dates is not None else 0,
            'train_end': str(train_index[-1]) if dates is not None else train_end - 1,
            'test_start': str(test_index[0]) if dates is not None else train_end,
            'test_end': str(test_index[-1]) if dates is not None else n_samples - 1
        }
        
        yield SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_index=train_index,
            test_index=test_index,
            split_info=split_info
        )


class RollingWindowSplit(BaseSplitter):
    """
    滚动时间窗口划分（步进式前向验证）。

    每次滚动严格隔离训练集与测试集。
    """
    
    def __init__(
        self,
        window_size: int = 252,
        step_size: int = 21,
        min_train_samples: int = 252
    ):
        """
        初始化滚动窗口划分器。

        参数:
            window_size: 训练窗口长度，默认 252（约一年交易日）
            step_size: 测试/步进窗口长度，默认 21（约一月交易日）
            min_train_samples: 最小训练样本数
        """
        super().__init__(min_train_samples=min_train_samples)
        self.window_size = max(window_size, min_train_samples)
        self.step_size = step_size
    
    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Iterator[SplitResult]:
        """
        产生滚动窗口划分序列。

        参数:
            X: 特征
            y: 目标
            dates: 日期索引

        产出:
            SplitResult: 每次滚动一折
        """
        n_samples = len(X)
        fold = 0
        
        # 初始训练窗口
        train_start = 0
        train_end = self.window_size
        
        while train_end + self.step_size <= n_samples:
            test_start = train_end
            test_end = min(test_start + self.step_size, n_samples)
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # 校验划分
            try:
                self._validate_split(X_train, y_train, X_test, y_test)
            except ValueError:
                # 训练样本不足则停止滚动
                break
            
            # 构造日期索引
            if dates is not None:
                train_index = dates[train_start:train_end]
                test_index = dates[test_start:test_end]
            else:
                train_index = pd.RangeIndex(start=train_start, stop=train_end)
                test_index = pd.RangeIndex(start=test_start, stop=test_end)
            
            split_info = {
                'split_type': 'rolling',
                'fold': fold,
                'window_size': self.window_size,
                'step_size': self.step_size,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_start': str(train_index[0]) if dates is not None else train_start,
                'train_end': str(train_index[-1]) if dates is not None else train_end - 1,
                'test_start': str(test_index[0]) if dates is not None else test_start,
                'test_end': str(test_index[-1]) if dates is not None else test_end - 1
            }
            
            yield SplitResult(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_index=train_index,
                test_index=test_index,
                split_info=split_info
            )
            
            # 窗口推进
            train_start += self.step_size
            train_end += self.step_size
            fold += 1


class MarketRegimeSplit(BaseSplitter):
    """
    牛熊市周期划分。

    使用外部给定的市场状态标签（如 market_regime）。
    支持同态验证（牛训牛测）或异态验证（牛训熊测）。

    判定规则（与 generate_regime_labels 一致）：
    - 从高点回撤 bear_threshold 判为熊市
    - 从低点反弹 bull_threshold 判为牛市
    """
    
    def __init__(
        self,
        regime_column: str = 'market_regime',
        min_train_samples: int = 252,
        validation_mode: str = 'same_regime',
        bull_threshold: float = 0.4,
        bear_threshold: float = 0.4
    ):
        """
        初始化市场周期划分器。

        参数:
            regime_column: 市场状态列名
            min_train_samples: 最小训练样本数
            validation_mode:
                - 'same_regime': 同一段周期内划分训练/测试
                - 'cross_regime': 上一段训练、下一段测试
            bull_threshold: 牛市阈值（相对低点涨幅比例）
            bear_threshold: 熊市阈值（相对高点跌幅比例）
        """
        super().__init__(min_train_samples=min_train_samples)
        self.regime_column = regime_column
        self.validation_mode = validation_mode
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
    
    @staticmethod
    def generate_regime_labels(
        prices: np.ndarray,
        bull_threshold: float = 0.4,
        bear_threshold: float = 0.4
    ) -> np.ndarray:
        """
        由价格序列生成牛熊标签。

        规则:
        - 从运行高点下跌 bear_threshold 进入熊市
        - 从运行低点上涨 bull_threshold 进入牛市

        参数:
            prices: 价格序列 (n_samples,)
            bull_threshold: 牛市阈值
            bear_threshold: 熊市阈值

        返回:
            regime: 标签 (n_samples,)，1=牛，0=熊
        """
        n = len(prices)
        regime = np.zeros(n, dtype=int)
        regime[0] = 1  # 初始视为牛市
        
        # 跟踪当前区间的最高/最低价
        current_high = prices[0]
        current_low = prices[0]
        current_regime = 1  # 1=牛，0=熊
        
        for i in range(1, n):
            price = prices[i]
            
            if current_regime == 1:
                # 牛市：更新高点
                current_high = max(current_high, price)
                # 从高点回撤达到阈值则转熊
                if price <= current_high * (1 - bear_threshold):
                    current_regime = 0
                    current_low = price
            else:
                # 熊市：更新低点
                current_low = min(current_low, price)
                # 从低点反弹达到阈值则转牛
                if price >= current_low * (1 + bull_threshold):
                    current_regime = 1
                    current_high = price
            
            regime[i] = current_regime
        
        return regime
    
    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        regime: Optional[np.ndarray] = None
    ) -> Iterator[SplitResult]:
        """
        按市场周期产生划分。

        参数:
            X: 特征
            y: 目标
            dates: 日期索引
            regime: 状态标签 (n_samples,)，1=牛，0=熊

        产出:
            SplitResult: 每个有效周期一折
        """
        if regime is None:
            raise ValueError("市场周期划分需要提供 regime 数组")
        
        n_samples = len(X)
        fold = 0
        
        # 状态切换点
        regime_changes = np.where(np.diff(regime) != 0)[0] + 1
        
        # 区间边界
        boundaries = [0] + regime_changes.tolist() + [n_samples]
        
        # 各段信息
        regimes_info = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            regimes_info.append({
                'start': start,
                'end': end,
                'regime': int(regime[start]),
                'length': end - start
            })
        
        if self.validation_mode == 'same_regime':
            # 同态：每段内部 80/20
            for info in regimes_info:
                if info['length'] < self.min_train_samples + 10:  # 需保留少量测试样本
                    # 周期过短，跳过
                    continue
                
                period_start = info['start']
                period_end = info['end']
                period_length = info['length']
                
                # 段内 80/20
                train_end = period_start + int(period_length * 0.8)
                
                if train_end - period_start < self.min_train_samples:
                    train_end = period_start + self.min_train_samples
                
                if period_end - train_end < 10:
                    # 测试集过小
                    continue
                
                X_train = X[period_start:train_end]
                y_train = y[period_start:train_end]
                X_test = X[train_end:period_end]
                y_test = y[train_end:period_end]
                
                try:
                    self._validate_split(X_train, y_train, X_test, y_test)
                except ValueError:
                    continue
                
                if dates is not None:
                    train_index = dates[period_start:train_end]
                    test_index = dates[train_end:period_end]
                else:
                    train_index = pd.RangeIndex(start=period_start, stop=train_end)
                    test_index = pd.RangeIndex(start=train_end, stop=period_end)
                
                split_info = {
                    'split_type': 'market_regime',
                    'validation_mode': 'same_regime',
                    'fold': fold,
                    'regime': info['regime'],
                    'regime_name': 'bull' if info['regime'] == 1 else 'bear',
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_start': str(train_index[0]) if dates is not None else period_start,
                    'train_end': str(train_index[-1]) if dates is not None else train_end - 1,
                    'test_start': str(test_index[0]) if dates is not None else train_end,
                    'test_end': str(test_index[-1]) if dates is not None else period_end - 1
                }
                
                yield SplitResult(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_index=train_index,
                    test_index=test_index,
                    split_info=split_info
                )
                
                fold += 1
        
        elif self.validation_mode == 'cross_regime':
            # 异态：前一段训练、后一段测试
            for i in range(len(regimes_info) - 1):
                train_info = regimes_info[i]
                test_info = regimes_info[i + 1]
                
                if train_info['length'] < self.min_train_samples:
                    continue
                
                if test_info['length'] < 10:
                    continue
                
                X_train = X[train_info['start']:train_info['end']]
                y_train = y[train_info['start']:train_info['end']]
                X_test = X[test_info['start']:test_info['end']]
                y_test = y[test_info['start']:test_info['end']]
                
                try:
                    self._validate_split(X_train, y_train, X_test, y_test)
                except ValueError:
                    continue
                
                if dates is not None:
                    train_index = dates[train_info['start']:train_info['end']]
                    test_index = dates[test_info['start']:test_info['end']]
                else:
                    train_index = pd.RangeIndex(
                        start=train_info['start'],
                        stop=train_info['end']
                    )
                    test_index = pd.RangeIndex(
                        start=test_info['start'],
                        stop=test_info['end']
                    )
                
                split_info = {
                    'split_type': 'market_regime',
                    'validation_mode': 'cross_regime',
                    'fold': fold,
                    'train_regime': train_info['regime'],
                    'train_regime_name': 'bull' if train_info['regime'] == 1 else 'bear',
                    'test_regime': test_info['regime'],
                    'test_regime_name': 'bull' if test_info['regime'] == 1 else 'bear',
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_start': str(train_index[0]) if dates is not None else train_info['start'],
                    'train_end': str(train_index[-1]) if dates is not None else train_info['end'] - 1,
                    'test_start': str(test_index[0]) if dates is not None else test_info['start'],
                    'test_end': str(test_index[-1]) if dates is not None else test_info['end'] - 1
                }
                
                yield SplitResult(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_index=train_index,
                    test_index=test_index,
                    split_info=split_info
                )
                
                fold += 1


class DataSplitter:
    """
    数据划分统一入口。

    策略:
    1. fixed: 固定 20/80 时间切分
    2. rolling: 滚动窗口
    3. regime: 牛熊周期
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化。

        参数:
            config: 含各策略参数的字典
        """
        self.config = config or {}
        self._splitters = {}
    
    def get_splitter(
        self,
        strategy: str,
        **kwargs
    ) -> BaseSplitter:
        """
        按策略名构造具体划分器实例。

        参数:
            strategy: 'fixed' / 'rolling' / 'regime'
            **kwargs: 覆盖配置项

        返回:
            BaseSplitter 实例
        """
        if strategy == 'fixed':
            params = self.config.get('fixed_split', {})
            params.update(kwargs)
            return FixedTimeSplit(
                train_ratio=params.get('train_ratio', 0.8),
                min_train_samples=params.get('min_train_samples', 252)
            )
        
        elif strategy == 'rolling':
            params = self.config.get('rolling_split', {})
            params.update(kwargs)
            return RollingWindowSplit(
                window_size=params.get('window_size', 252),
                step_size=params.get('step_size', 21),
                min_train_samples=params.get('min_train_samples', 252)
            )
        
        elif strategy == 'regime':
            params = self.config.get('regime_split', {})
            params.update(kwargs)
            return MarketRegimeSplit(
                regime_column=params.get('regime_column', 'market_regime'),
                min_train_samples=params.get('min_train_samples', 252),
                validation_mode=params.get('validation_mode', 'same_regime')
            )
        
        else:
            raise ValueError(f"不支持的划分策略: {strategy}")
    
    def get_splits(
        self,
        strategy: str,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        regime: Optional[np.ndarray] = None,
        **kwargs
    ) -> Iterator[SplitResult]:
        """
        按策略产生划分迭代器。

        参数:
            strategy: 'fixed' / 'rolling' / 'regime'
            X: 特征
            y: 目标
            dates: 日期索引
            regime: 周期标签（regime 策略必填）
            **kwargs: 策略参数覆盖

        产出:
            SplitResult
        """
        splitter = self.get_splitter(strategy, **kwargs)
        
        if strategy == 'regime':
            yield from splitter.get_splits(X, y, dates, regime)
        else:
            yield from splitter.get_splits(X, y, dates)
    
    def count_splits(
        self,
        strategy: str,
        X: np.ndarray,
        y: np.ndarray,
        regime: Optional[np.ndarray] = None,
        **kwargs
    ) -> int:
        """
        统计该策略下会产生多少次划分。

        参数:
            strategy: 策略名
            X: 特征
            y: 目标
            regime: 周期标签
            **kwargs: 策略参数覆盖

        返回:
            划分数
        """
        count = 0
        for _ in self.get_splits(strategy, X, y, regime=regime, **kwargs):
            count += 1
        return count
