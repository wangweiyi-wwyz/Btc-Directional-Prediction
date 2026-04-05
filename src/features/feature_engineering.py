"""
特征工程主模块

整合技术指标和宏观特征工程
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from .technical_indicators import TechnicalIndicators
from .macro_features import MacroFeatures
from .feature_selector import FeatureSelector
from ..data.data_preprocessor import RollingWindowScaler


class FeatureEngineering:
    """
    特征工程主类
    
    整合：
    1. 技术指标计算
    2. 宏观特征工程
    3. 特征选择
    4. 滚动窗口标准化
    """
    
    def __init__(self, scaler_window: int = 252, random_state: int = 42):
        """
        初始化特征工程
        
        参数：
            scaler_window: 滚动窗口标准化窗口大小
            random_state: 随机种子
        """
        self.technical_engineer = TechnicalIndicators()
        self.macro_engineer = MacroFeatures()
        self.feature_selector = FeatureSelector(random_state=random_state)
        self.scaler = RollingWindowScaler(window=scaler_window)
        self.feature_columns: List[str] = []
        
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标特征
        
        参数：
            df: 包含 OHLCV 数据的 DataFrame
            
        返回：
            包含技术指标的 DataFrame
        """
        return self.technical_engineer.compute_all(df)
    
    def compute_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算宏观特征
        
        参数：
            df: 包含宏观数据的 DataFrame
            
        返回：
            包含宏观特征的 DataFrame
        """
        return self.macro_engineer.compute_all(df)
    
    def fit_transform_features(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
        apply_selection: bool = False,
        model=None,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        完整的特征工程流程
        
        1. 计算技术指标
        2. 计算宏观特征
        3. 滚动窗口标准化
        4. 特征选择（可选）
        
        参数：
            df: 输入 DataFrame
            exclude_cols: 要排除的列名列表
            apply_selection: 是否应用特征选择
            model: 用于特征选择的模型
            y: 目标变量（用于特征选择）
            
        返回：
            处理后的 DataFrame
        """
        if exclude_cols is None:
            exclude_cols = ['target', 'next_return', 'timestamp']
        
        result = df.copy()
        
        # 计算技术指标
        result = self.compute_technical_features(result)
        
        # 计算宏观特征
        result = self.compute_macro_features(result)
        
        # 获取特征列
        feature_cols = [col for col in result.columns if col not in exclude_cols]
        
        # 滚动窗口标准化
        result = self.scaler.fit_transform(result, feature_cols)
        
        # 使用标准化后的特征
        scaled_cols = [f'{col}_scaled' for col in feature_cols if f'{col}_scaled' in result.columns]
        result[scaled_cols] = result[scaled_cols].fillna(0)
        
        # 更新特征列
        self.feature_columns = scaled_cols
        
        # 特征选择
        if apply_selection and model is not None and y is not None:
            X = result[scaled_cols]
            _, selected = self.feature_selector.select_features(
                X, y, model, shap_top_k=50, rfe_n_features=30
            )
            self.feature_columns = [f'{col}_scaled' for col in selected]
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """
        获取特征列名列表
        
        返回：
            特征列名列表
        """
        return self.feature_columns
    
    def transform_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        对新数据进行特征转换
        
        参数：
            df: 输入 DataFrame
            feature_cols: 特征列名列表
            
        返回：
            转换后的 DataFrame
        """
        result = df.copy()
        
        # 计算技术指标
        result = self.compute_technical_features(result)
        
        # 计算宏观特征
        result = self.compute_macro_features(result)
        
        # 滚动窗口标准化
        result = self.scaler.fit_transform(result, feature_cols)
        
        return result
