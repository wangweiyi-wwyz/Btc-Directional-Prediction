"""
数据预处理模块

功能：
1. 从本地加载 BTC 量价数据、技术指标、宏观数据
2. 数据对齐与填充：宏观数据通过插值法与高频加密市场对齐
3. 缺失值处理：严格使用前向填充（Forward-fill）

红线要求：
- 禁止未来函数：任何插值、标准化、滑动窗口特征生成必须仅基于过去的时间窗口
- 因果对齐：特征观测期时间戳必须比对应的分类目标标签早一个交易日
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class DataPreprocessor:
    """
    数据预处理器
    
    负责：
    - 加载本地数据文件
    - 数据对齐与合并
    - 缺失值处理（仅使用 Forward-fill）
    - 创建目标变量（明日涨跌方向）
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据预处理器
        
        参数：
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.ohlcv_data: Optional[pd.DataFrame] = None
        self.technical_data: Optional[pd.DataFrame] = None
        self.macro_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        
    def load_ohlcv(self, filename: str = "btc_ohlc.csv") -> pd.DataFrame:
        """
        加载 BTC OHLCV 数据
        
        数据源: data/btc_ohlc.csv
        
        参数：
            filename: OHLCV 数据文件名
            
        返回：
            包含 OHLCV 数据的 DataFrame
        """
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        self.ohlcv_data = df
        return df
    
    def load_technical_indicators(self, filename: str = "btc_technical_data_raw.csv") -> pd.DataFrame:
        """
        加载技术指标数据
        
        数据源: data/btc_technical_data_raw.csv
        
        参数：
            filename: 技术指标数据文件名
            
        返回：
            包含技术指标的 DataFrame
        """
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        self.technical_data = df
        return df
    
    def load_macro_data(self, filename: str = "macro_data_raw.csv") -> pd.DataFrame:
        """
        加载宏观数据
        
        数据源: data/macro_data_raw.csv
        
        参数：
            filename: 宏观数据文件名
            
        返回：
            包含宏观数据的 DataFrame
        """
        file_path = self.data_dir / filename
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        self.macro_data = df
        return df
    
    def align_macro_data(
        self, 
        target_index: pd.DatetimeIndex,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        将宏观数据与高频市场数据对齐
        
        使用插值法将低频宏观数据对齐到日频
        
        参数：
            target_index: 目标时间索引（日频）
            method: 插值方法，默认前向填充
            
        返回：
            对齐后的宏观数据 DataFrame
        """
        if self.macro_data is None:
            raise ValueError("请先加载宏观数据")
        
        # 重新索引到目标频率
        aligned = self.macro_data.reindex(target_index, method=method)
        
        return aligned
    
    def merge_data(
        self,
        drop_missing: bool = False,
        missing_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        合并所有数据源
        
        参数：
            drop_missing: 是否删除缺失值过多的列
            missing_threshold: 缺失值比例阈值
            
        返回：
            合并后的 DataFrame
        """
        if self.ohlcv_data is None:
            raise ValueError("请先加载 OHLCV 数据")
        
        # 以 OHLCV 数据的时间索引为基准
        base_index = self.ohlcv_data.index
        merged = self.ohlcv_data.copy()
        
        # 合并技术指标数据
        if self.technical_data is not None:
            # 确保技术指标数据与 OHLCV 对齐
            technical_aligned = self.technical_data.reindex(base_index)
            merged = pd.concat([merged, technical_aligned], axis=1)
        
        # 合并宏观数据（使用前向填充对齐）
        if self.macro_data is not None:
            macro_aligned = self.align_macro_data(base_index, method="ffill")
            merged = pd.concat([merged, macro_aligned], axis=1)
        
        # 处理缺失值过多的列
        if drop_missing:
            missing_ratio = merged.isnull().sum() / len(merged)
            cols_to_keep = missing_ratio[missing_ratio <= missing_threshold].index
            merged = merged[cols_to_keep]
        
        # 前向填充缺失值（红线要求：严禁使用 Backward Fill）
        merged.ffill(inplace=True)
        
        # 如果仍有缺失值（开头部分），用零填充处理初始缺失
        # 严禁使用 bfill，因为会造成未来数据泄露
        merged.fillna(0, inplace=True)
        
        self.merged_data = merged
        return merged
    
    def create_target(
        self,
        df: Optional[pd.DataFrame] = None,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        创建目标变量：明日涨跌方向
        
        参数：
            df: 输入 DataFrame，默认使用 self.merged_data
            threshold: 涨跌阈值，默认为0（任何正涨幅为1，否则为0）
            
        返回：
            包含目标变量的 DataFrame
            - target: 1 表示明日上涨，0 表示明日下跌
        """
        if df is None:
            df = self.merged_data.copy()
        
        if df is None:
            raise ValueError("请先合并数据")
        
        # 计算明日收益率（正确方式：先计算收益率，再向前平移）
        # pct_change(shift=-1) 计算的是 (close[t] - close[t+1]) / close[t+1]，这是错误的
        # 正确的下一日收益率应该是 (close[t+1] - close[t]) / close[t]
        df['next_return'] = df['close'].pct_change().shift(-1)
        
        # 创建目标变量：明日涨跌方向
        df['target'] = (df['next_return'] > threshold).astype(int)
        
        # 删除最后一行（没有明日数据）
        df = df[:-1]
        
        return df
    
    def get_train_val_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序切分训练集、验证集、测试集
        
        参数：
            df: 输入 DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        返回：
            (train_df, val_df, test_df)
        """
        if df is None:
            df = self.merged_data
        
        if df is None:
            raise ValueError("请先合并数据")
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        return train_df, val_df, test_df
    
    def get_feature_columns(
        self,
        df: Optional[pd.DataFrame] = None,
        exclude_cols: Optional[list] = None
    ) -> list:
        """
        获取特征列名
        
        参数：
            df: 输入 DataFrame
            exclude_cols: 要排除的列名列表
            
        返回：
            特征列名列表
        """
        if df is None:
            df = self.merged_data
        
        if df is None:
            raise ValueError("请先合并数据")
        
        if exclude_cols is None:
            exclude_cols = ['target', 'next_return', 'date']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


class RollingWindowScaler:
    """
    滚动窗口标准化器
    
    红线要求：
    - 必须使用 252 天滚动窗口的 StandardScaler 进行标准化
    - 严禁计算全样本均值方差
    """
    
    def __init__(self, window: int = 252):
        """
        初始化滚动窗口标准化器
        
        参数：
            window: 滚动窗口大小，默认252（约一年交易日）
        """
        self.window = window
        
    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        使用滚动窗口标准化特征
        
        参数：
            df: 输入 DataFrame
            feature_cols: 需要标准化的特征列
            
        返回：
            标准化后的 DataFrame
        """
        result = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                # 计算滚动均值和标准差
                rolling_mean = df[col].rolling(window=self.window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=self.window, min_periods=1).std()
                
                # 标准化
                result[f'{col}_scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return result
    
    def transform(self, df: pd.DataFrame, feature_cols: list, stats: Dict) -> pd.DataFrame:
        """
        使用预计算的统计量进行标准化
        
        参数：
            df: 输入 DataFrame
            feature_cols: 需要标准化的特征列
            stats: 预计算的统计量字典
            
        返回：
            标准化后的 DataFrame
        """
        result = df.copy()
        
        for col in feature_cols:
            if col in df.columns and col in stats:
                mean = stats[col]['mean']
                std = stats[col]['std']
                result[f'{col}_scaled'] = (df[col] - mean) / (std + 1e-8)
        
        return result
