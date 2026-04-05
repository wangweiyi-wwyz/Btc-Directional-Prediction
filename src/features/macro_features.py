"""
宏观数据特征工程模块

功能：
1. 价格指数类计算对数收益率
2. 利率类计算一阶差分
3. 规模金额类对数化后计算同比/环比
4. 衍生特征：CPI_Surprise（预期差）、Net_Liquidity（流动性）
5. 对利率族/就业族执行 PCA 保留前2个主成分
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.decomposition import PCA


class MacroFeatures:
    """
    宏观数据特征工程
    
    处理以下类型宏观数据：
    1. 价格指数类：CPI, PPI, 黄金价格, 原油价格等
    2. 利率类：联邦基金利率, 10年期国债收益率等
    3. 就业类：非农就业人口, 失业率等
    4. 规模金额类：M2, GDP等
    """
    
    # 宏观数据分类配置（匹配实际数据文件列名）
    PRICE_INDEX_COLS = ['Gold_Price', 'WTI_Oil_Price', 'Copper_Price', 'S&P_500', 'NASDAQ_Composite', 'USD_Index']
    RATE_COLS = ['Fed_Funds_Rate', 'UST_10Y_Yield', 'Term_Spread_10Y_2Y', 'VIX']
    EMPLOYMENT_COLS = ['Nonfarm_Payrolls', 'Unemployment_Rate', 'Initial_Claims', 'JOLTS_Job_Openings']
    SCALE_COLS = ['Fed_Balance_Sheet_Total', 'RRP_Balance', 'TGA_Balance', 'Govt_Debt_Total']
    
    def __init__(self):
        self.pca_models: Dict[str, PCA] = {}
        
    @staticmethod
    def log_return(series: pd.Series) -> pd.Series:
        """
        计算对数收益率
        
        参数：
            series: 价格序列
            
        返回：
            对数收益率序列
        """
        return np.log(series / series.shift(1))
    
    @staticmethod
    def first_difference(series: pd.Series) -> pd.Series:
        """
        计算一阶差分
        
        参数：
            series: 输入序列
            
        返回：
            一阶差分序列
        """
        return series.diff()
    
    @staticmethod
    def log_scale(series: pd.Series) -> pd.Series:
        """
        对数化
        
        参数：
            series: 输入序列
            
        返回：
            对数化序列
        """
        return np.log(series + 1e-8)
    
    @staticmethod
    def yoy_change(series: pd.Series, period: int = 12) -> pd.Series:
        """
        同比变化率
        
        参数：
            series: 输入序列
            period: 周期，默认12（月度数据）
            
        返回：
            同比变化率序列
        """
        return (series - series.shift(period)) / (series.shift(period) + 1e-8)
    
    @staticmethod
    def mom_change(series: pd.Series, period: int = 1) -> pd.Series:
        """
        环比变化率
        
        参数：
            series: 输入序列
            period: 周期，默认1
            
        返回：
            环比变化率序列
        """
        return (series - series.shift(period)) / (series.shift(period) + 1e-8)
    
    def create_cpi_surprise(self, df: pd.DataFrame, cpi_col: str = 'CPI',
                           expected_col: Optional[str] = None) -> pd.Series:
        """
        创建 CPI 预期差特征
        
        CPI_Surprise = 实际CPI - 预期CPI
        
        参数：
            df: 输入 DataFrame
            cpi_col: CPI 列名
            expected_col: 预期CPI列名，如果为None则使用移动平均作为预期
            
        返回：
            CPI_Surprise 序列
        """
        # 自动匹配实际列名
        possible_cpi_cols = ['CPI', 'CPI_YoY', 'Core_PCE_YoY']
        cpi_col = next((c for c in possible_cpi_cols if c in df.columns), None)
        
        if cpi_col is None:
            return pd.Series(0, index=df.index)
        
        if expected_col is not None and expected_col in df.columns:
            surprise = df[cpi_col] - df[expected_col]
        else:
            # 使用3个月移动平均作为预期
            expected = df[cpi_col].rolling(window=3, min_periods=1).mean()
            surprise = df[cpi_col] - expected
        
        return surprise
    
    def create_net_liquidity(self, df: pd.DataFrame,
                            fed_balance_col: str = 'FED_BALANCE_SHEET',
                            reverse_repo_col: str = 'REVERSE_REPO',
                            tga_col: str = 'TREASURY_GENERAL_ACCOUNT') -> pd.Series:
        """
        创建净流动性特征
        
        Net_Liquidity = Fed_Balance_Sheet - Reverse_Repo - TGA
        
        参数：
            df: 输入 DataFrame
            fed_balance_col: 美联储资产负债表列名
            reverse_repo_col: 逆回购列名
            tga_col: 财政部一般账户列名
            
        返回：
            Net_Liquidity 序列
        """
        # 自动匹配实际列名
        possible_fed_cols = ['FED_BALANCE_SHEET', 'Fed_Balance_Sheet_Total']
        possible_rrp_cols = ['REVERSE_REPO', 'RRP_Balance']
        possible_tga_cols = ['TREASURY_GENERAL_ACCOUNT', 'TGA_Balance']
        
        fed_col = next((c for c in possible_fed_cols if c in df.columns), None)
        rrp_col = next((c for c in possible_rrp_cols if c in df.columns), None)
        tga_col = next((c for c in possible_tga_cols if c in df.columns), None)
        
        if fed_col is None:
            return pd.Series(0, index=df.index)
        
        net_liquidity = df[fed_col].copy()
        
        if rrp_col is not None:
            net_liquidity = net_liquidity - df[rrp_col]
        
        if tga_col is not None:
            net_liquidity = net_liquidity - df[tga_col]
        
        return net_liquidity
    
    def apply_pca(self, df: pd.DataFrame, columns: List[str],
                  n_components: int = 2, prefix: str = 'pca',
                  fit: bool = True, pca_model: Optional[PCA] = None) -> pd.DataFrame:
        """
        对指定列应用 PCA 降维
        
        红线要求：
        1. 严禁使用全样本标准化，必须使用滚动窗口或仅训练集统计量
        2. PCA 必须仅在训练集上 fit，验证/测试集仅 transform
        
        参数：
            df: 输入 DataFrame
            columns: 要进行 PCA 的列名列表
            n_components: 主成分数量
            prefix: 输出列名前缀
            fit: 是否拟合 PCA 模型（训练集为 True，验证/测试集为 False）
            pca_model: 预训练的 PCA 模型（用于验证/测试集）
            
        返回：
            包含主成分的 DataFrame
        """
        # 筛选存在的列
        available_cols = [col for col in columns if col in df.columns]
        
        if len(available_cols) == 0:
            return df
        
        # 修复：提取数据并处理缺失值
        data = df[available_cols].ffill().fillna(0)
        
        # 修复：使用滚动窗口标准化（252天窗口）而非全样本标准化
        data_normalized = data.copy()
        for col in available_cols:
            rolling_mean = data[col].rolling(window=252, min_periods=1).mean()
            rolling_std = data[col].rolling(window=252, min_periods=1).std()
            data_normalized[col] = (data[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # 处理标准化后的 NaN（开头部分）
        data_normalized = data_normalized.fillna(0)
        
        # 应用 PCA
        if fit:
            # 训练集：拟合新的 PCA
            pca = PCA(n_components=min(n_components, len(available_cols)))
            components = pca.fit_transform(data_normalized.values)
            
            # 保存 PCA 模型
            self.pca_models[prefix] = pca
        else:
            # 验证/测试集：使用预训练的 PCA
            if pca_model is None and prefix not in self.pca_models:
                raise ValueError(f"PCA 模型 '{prefix}' 尚未训练，请先在训练集上调用 fit=True")
            
            pca = pca_model if pca_model is not None else self.pca_models[prefix]
            components = pca.transform(data_normalized.values)
        
        # 添加主成分到 DataFrame
        result = df.copy()
        for i in range(components.shape[1]):
            result[f'{prefix}_{i+1}'] = components[:, i]
        
        return result
    
    def transform_price_index_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换价格指数类特征：计算对数收益率
        
        参数：
            df: 输入 DataFrame
            
        返回：
            包含转换后特征的 DataFrame
        """
        result = df.copy()
        
        for col in self.PRICE_INDEX_COLS:
            if col in df.columns:
                result[f'{col}_log_return'] = self.log_return(df[col])
        
        return result
    
    def transform_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换利率类特征：计算一阶差分
        
        参数：
            df: 输入 DataFrame
            
        返回：
            包含转换后特征的 DataFrame
        """
        result = df.copy()
        
        for col in self.RATE_COLS:
            if col in df.columns:
                result[f'{col}_diff'] = self.first_difference(df[col])
        
        return result
    
    def transform_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换规模金额类特征：对数化后计算同比/环比
        
        参数：
            df: 输入 DataFrame
            
        返回：
            包含转换后特征的 DataFrame
        """
        result = df.copy()
        
        for col in self.SCALE_COLS:
            if col in df.columns:
                # 对数化
                result[f'{col}_log'] = self.log_scale(df[col])
                # 同比
                result[f'{col}_yoy'] = self.yoy_change(df[col])
                # 环比
                result[f'{col}_mom'] = self.mom_change(df[col])
        
        return result
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建衍生特征
        
        参数：
            df: 输入 DataFrame
            
        返回：
            包含衍生特征的 DataFrame
        """
        result = df.copy()
        
        # CPI 预期差列（CPI_Surprise）
        if 'CPI' in df.columns:
            result['CPI_Surprise'] = self.create_cpi_surprise(df)
        
        # 净流动性列（Net_Liquidity）
        result['Net_Liquidity'] = self.create_net_liquidity(df)
        
        return result
    
    def apply_pca_transformations(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        对利率族和就业族应用 PCA
        
        参数：
            df: 输入 DataFrame
            fit: 是否拟合 PCA 模型（训练集为 True，验证/测试集为 False）
            
        返回：
            包含主成分的 DataFrame
        """
        result = df.copy()
        
        # 利率族 PCA
        rate_cols = [col for col in self.RATE_COLS if col in df.columns]
        if len(rate_cols) > 2:
            result = self.apply_pca(result, rate_cols, n_components=2, prefix='rate_pca', fit=fit)
        
        # 就业族 PCA
        employment_cols = [col for col in self.EMPLOYMENT_COLS if col in df.columns]
        if len(employment_cols) > 2:
            result = self.apply_pca(result, employment_cols, n_components=2, prefix='employment_pca', fit=fit)
        
        return result
    
    def compute_all(self, df: pd.DataFrame, fit_pca: bool = True) -> pd.DataFrame:
        """
        计算所有宏观特征
        
        参数：
            df: 包含原始宏观数据的 DataFrame
            fit_pca: 是否拟合 PCA 模型（训练集为 True，验证/测试集为 False）
            
        返回：
            包含所有宏观特征的 DataFrame
        """
        result = df.copy()
        
        # 转换各类特征
        result = self.transform_price_index_features(result)
        result = self.transform_rate_features(result)
        result = self.transform_scale_features(result)
        
        # 创建衍生特征
        result = self.create_derived_features(result)
        
        # 应用 PCA（修复：添加 fit_pca 参数）
        result = self.apply_pca_transformations(result, fit=fit_pca)
        
        return result
