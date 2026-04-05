"""
基础模型模块

包含：
1. Ridge 分类器（线性基准）
2. 随机森林分类器
3. XGBoost 分类器
4. LightGBM 分类器
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class BaseModels:
    """
    基础机器学习模型
    
    支持的模型：
    1. Ridge 分类器——线性基准
    2. 随机森林
    3. XGBoost 梯度提升树
    4. LightGBM 轻量梯度提升
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化基础模型
        
        参数：
            random_state: 随机种子
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        
    def get_ridge_classifier(self, alpha: float = 1.0) -> RidgeClassifier:
        """
        获取 Ridge 分类器（线性基准）
        
        参数：
            alpha: 正则化强度
            
        返回：
            RidgeClassifier 实例
        """
        return RidgeClassifier(
            alpha=alpha,
            random_state=self.random_state,
            class_weight='balanced'
        )
    
    def get_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ) -> RandomForestClassifier:
        """
        获取随机森林分类器
        
        参数：
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            
        返回：
            RandomForestClassifier 实例
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def get_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ) -> XGBClassifier:
        """
        获取 XGBoost 分类器
        
        参数：
            n_estimators: 提升轮数
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            
        返回：
            XGBClassifier 实例
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1,
            use_label_encoder=False
        )
    
    def get_lightgbm(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ) -> LGBMClassifier:
        """
        获取 LightGBM 分类器
        
        参数：
            n_estimators: 提升轮数
            max_depth: 树的最大深度
            learning_rate: 学习率
            num_leaves: 叶子节点数
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            
        返回：
            LGBMClassifier 实例
        """
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='binary',
            metric='binary_logloss',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    def get_all_models(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        获取所有模型
        
        参数：
            config: 模型配置字典
            
        返回：
            模型字典
        """
        if config is None:
            config = self.get_default_config()
        
        models = {
            'ridge': self.get_ridge_classifier(**config.get('ridge', {})),
            'random_forest': self.get_random_forest(**config.get('random_forest', {})),
            'xgboost': self.get_xgboost(**config.get('xgboost', {})),
            'lightgbm': self.get_lightgbm(**config.get('lightgbm', {}))
        }
        
        self.models = models
        return models
    
    def get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """
        获取默认模型配置
        
        返回：
            默认配置字典
        """
        return {
            'ridge': {
                'alpha': 1.0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
    
    def get_model(self, model_name: str, **kwargs) -> Any:
        """
        获取指定模型
        
        参数：
            model_name: 模型名称 ('ridge', 'random_forest', 'xgboost', 'lightgbm')
            **kwargs: 模型参数
            
        返回：
            模型实例
        """
        model_map = {
            'ridge': self.get_ridge_classifier,
            'random_forest': self.get_random_forest,
            'xgboost': self.get_xgboost,
            'lightgbm': self.get_lightgbm
        }
        
        if model_name not in model_map:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return model_map[model_name](**kwargs)
    
    def get_feature_importance(self, model_name: str, model: Any, 
                              feature_names: list) -> pd.DataFrame:
        """
        获取特征重要性
        
        参数：
            model_name: 模型名称
            model: 训练好的模型
            feature_names: 特征名称列表
            
        返回：
            特征重要性 DataFrame
        """
        importance = None
        
        if model_name == 'ridge':
            # Ridge 使用系数绝对值
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            raise ValueError(f"模型 {model_name} 不支持特征重要性")
        
        result = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return result
