"""
BTC 分类型目标预测系统 - 主程序入口

项目概述：
- 任务类型：对比特币（BTC）价格明日涨跌方向进行二分类预测
- 数据范围：日频数据，涵盖2016年1月至2024年6月30日
- 技术栈：Python, scikit-learn, XGBoost, LightGBM, PyTorch

模块：
1. 数据获取与预处理
2. 特征工程
3. 模型开发与训练
4. 验证与评估
5. 交易回测
"""

from .data import DataPreprocessor, RollingWindowScaler
from .features import FeatureEngineering, TechnicalIndicators, MacroFeatures, FeatureSelector
from .models import BaseModels, DeepLearningModels, ModelTrainer
from .evaluation import MetricsCalculator, ModelExplainer, ModelEvaluator
from .backtesting import Backtester, TradingStrategy, SignalGenerator
from .utils import setup_logger, ConfigLoader

__all__ = [
    'DataPreprocessor', 'RollingWindowScaler',
    'FeatureEngineering', 'TechnicalIndicators', 'MacroFeatures', 'FeatureSelector',
    'BaseModels', 'DeepLearningModels', 'ModelTrainer',
    'MetricsCalculator', 'ModelExplainer', 'ModelEvaluator',
    'Backtester', 'TradingStrategy', 'SignalGenerator',
    'setup_logger', 'ConfigLoader'
]
