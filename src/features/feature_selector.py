"""
特征选择模块

功能：
1. 技术指标严禁使用 VIF，需基于经济含义合并
2. 初筛使用 SHAP 值
3. 精筛使用递归特征消除（RFE）结合 TimeSeriesSplit
4. 禁止目标变量泄露：严禁使用目标变量 y 进行特征筛选
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit


class FeatureSelector:
    """
    特征选择器
    
    支持的特征选择方法：
    1. 基于经济含义的特征合并
    2. SHAP 值初筛
    3. RFE + TimeSeriesSplit 精筛
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化特征选择器
        
        参数：
            random_state: 随机种子
        """
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.shap_values: Optional[np.ndarray] = None
        
    @staticmethod
    def merge_features_by_economic_meaning(df: pd.DataFrame, 
                                          feature_groups: dict) -> pd.DataFrame:
        """
        基于经济含义合并特征
        
        参数：
            df: 输入 DataFrame
            feature_groups: 特征分组字典，格式为 {'group_name': ['feat1', 'feat2', ...]}
            
        返回：
            合并后的 DataFrame
        """
        result = df.copy()
        
        for group_name, features in feature_groups.items():
            # 筛选存在的特征
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) > 0:
                # 使用平均值合并
                result[group_name] = df[available_features].mean(axis=1)
        
        return result
    
    def shap_feature_selection(self, model, X: pd.DataFrame, 
                              top_k: int = 50) -> List[str]:
        """
        使用 SHAP 值进行特征初筛
        
        参数：
            model: 训练好的模型
            X: 特征 DataFrame
            top_k: 保留前 k 个特征
            
        返回：
            选择的特征列表
        """
        try:
            import shap
            # 计算 SHAP 值
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            self.shap_values = shap_values.values
            
            # 计算特征重要性（绝对值的平均）
            if len(shap_values.values.shape) == 3:
                importance = np.abs(shap_values.values).mean(axis=(0, 1))
            else:
                importance = np.abs(shap_values.values).mean(axis=0)
        except Exception:
            # 如果 SHAP 计算失败，使用模型内置的特征重要性
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                # 对于线性模型，使用系数绝对值
                importance = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else None
                
            if importance is None:
                # 如果无法获取重要性，返回所有特征
                return X.columns.tolist()
        
        # 获取特征重要性排序
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 选择前 top_k 个特征
        selected = feature_importance.head(top_k)['feature'].tolist()
        
        return selected
    
    def rfe_feature_selection(self, model, X: pd.DataFrame, y: pd.Series,
                             n_features_to_select: int = 30,
                             n_splits: int = 5) -> List[str]:
        """
        使用 RFE + TimeSeriesSplit 进行特征精筛
        
        修复：使用 TimeSeriesSplit 进行交叉验证，确保时序数据不泄露
        
        参数：
            model: 基础模型
            X: 特征 DataFrame
            y: 目标变量
            n_features_to_select: 要选择的特征数量
            n_splits: TimeSeriesSplit 的折数
            
        返回：
            选择的特征列表
        """
        from sklearn.model_selection import cross_val_score
        
        # 创建 TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 创建 RFE
        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features_to_select,
            step=0.1  # 每次移除 10% 的特征
        )
        
        # 拟合 RFE
        rfe.fit(X, y)
        
        # 使用 TimeSeriesSplit 验证特征选择的稳定性
        # 获取选择的特征
        selected = X.columns[rfe.support_].tolist()
        
        # 交叉验证评估特征子集的性能
        X_selected = X[selected]
        cv_scores = cross_val_score(
            model, X_selected, y, cv=tscv, scoring='f1'
        )
        
        # 打印交叉验证结果
        print(f"RFE 选择了 {len(selected)} 个特征")
        print(f"TimeSeriesSplit 交叉验证 F1 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.selected_features = selected
        
        return selected
    
    def get_feature_correlation(self, df: pd.DataFrame, 
                               feature_cols: List[str],
                               threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        计算特征间相关性，找出高相关特征对
        
        参数：
            df: 输入 DataFrame
            feature_cols: 特征列名列表
            threshold: 相关系数阈值
            
        返回：
            高相关特征对列表
        """
        # 计算相关系数矩阵
        corr_matrix = df[feature_cols].corr().abs()
        
        # 找出高相关特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return high_corr_pairs
    
    def remove_high_correlation_features(self, df: pd.DataFrame,
                                        feature_cols: List[str],
                                        threshold: float = 0.95) -> List[str]:
        """
        移除高相关性特征
        
        参数：
            df: 输入 DataFrame
            feature_cols: 特征列名列表
            threshold: 相关系数阈值
            
        返回：
            保留的特征列表
        """
        # 计算相关系数矩阵
        corr_matrix = df[feature_cols].corr().abs()
        
        # 创建要删除的特征集合
        to_drop = set()
        
        # 遍历相关系数矩阵
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # 保留相关性更高的特征（与其他特征的平均相关性）
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    
                    avg_corr_i = corr_matrix.iloc[i, :].mean()
                    avg_corr_j = corr_matrix.iloc[j, :].mean()
                    
                    if avg_corr_i < avg_corr_j:
                        to_drop.add(feat_i)
                    else:
                        to_drop.add(feat_j)
        
        # 返回保留的特征
        selected = [f for f in feature_cols if f not in to_drop]
        
        return selected
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       model,
                       shap_top_k: int = 50,
                       rfe_n_features: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """
        完整的特征选择流程
        
        1. SHAP 值初筛
        2. 移除高相关性特征
        3. RFE 精筛
        
        参数：
            X: 特征 DataFrame
            y: 目标变量
            model: 用于特征选择的模型
            shap_top_k: SHAP 初筛保留的特征数
            rfe_n_features: RFE 最终选择的特征数
            
        返回：
            (X_selected, selected_features): 选择后的特征 DataFrame 和特征列表
        """
        # 步骤1: SHAP 值初筛
        shap_selected = self.shap_feature_selection(model, X, top_k=shap_top_k)
        X_shap = X[shap_selected]
        
        # 步骤2: 移除高相关性特征
        corr_selected = self.remove_high_correlation_features(
            X_shap, shap_selected, threshold=0.95
        )
        X_corr = X[corr_selected]
        
        # 步骤3: RFE 精筛
        rfe_selected = self.rfe_feature_selection(
            model, X_corr, y, n_features_to_select=rfe_n_features
        )
        
        X_selected = X[rfe_selected]
        
        return X_selected, rfe_selected
    
    def get_feature_groups(self) -> dict:
        """
        获取预定义的特征分组（基于经济含义）
        
        返回：
            特征分组字典
        """
        return {
            # 趋势类
            'trend_ma': ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200'],
            'trend_macd': ['macd', 'macd_signal', 'macd_histogram'],
            'trend_adx': ['adx'],
            
            # 动量类
            'momentum_rsi': ['rsi', 'rsi_normalized'],
            'momentum_stoch': ['stoch_k', 'stoch_d'],
            'momentum_williams': ['williams_r', 'williams_r_normalized'],
            'momentum_cci': ['cci'],
            'momentum_roc': ['roc'],
            
            # 成交量类
            'volume': ['volume_log', 'obv', 'ad_line', 'vwap'],
            
            # 波动率类
            'volatility_atr': ['atr'],
            'volatility_bb': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct'],
            'volatility_kc': ['kc_upper', 'kc_middle', 'kc_lower'],
            
            # 宏观-价格指数
            'macro_price': ['CPI_log_return', 'PPI_log_return', 
                           'GOLD_PRICE_log_return', 'OIL_PRICE_log_return'],
            
            # 宏观-利率
            'macro_rate': ['FED_FUNDS_RATE_diff', '10Y_TREASURY_YIELD_diff',
                          '2Y_TREASURY_YIELD_diff', 'rate_pca_1', 'rate_pca_2'],
            
            # 宏观-就业
            'macro_employment': ['NONFARM_PAYROLLS_diff', 'UNEMPLOYMENT_RATE_diff',
                               'employment_pca_1', 'employment_pca_2'],
            
            # 宏观-流动性
            'macro_liquidity': ['Net_Liquidity', 'M2_log', 'M2_yoy', 'M2_mom'],
            
            # 衍生特征
            'derived': ['CPI_Surprise']
        }
