"""
数据模块初始化

功能：
1. 从本地加载 BTC 量价数据、技术指标、宏观数据
2. 数据对齐与填充：宏观数据通过插值法与高频加密市场对齐
3. 缺失值处理：严格使用前向填充（Forward-fill）
4. 数据划分：支持固定比例、滚动窗口、市场周期三种划分策略
"""

from .data_preprocessor import DataPreprocessor, RollingWindowScaler
from .splitter import (
    DataSplitter,
    FixedTimeSplit,
    RollingWindowSplit,
    MarketRegimeSplit,
    SplitResult
)

__all__ = [
    'DataPreprocessor',
    'RollingWindowScaler',
    'DataSplitter',
    'FixedTimeSplit',
    'RollingWindowSplit',
    'MarketRegimeSplit',
    'SplitResult'
]
