"""
模型模块初始化

包含：
1. 基础模型：Ridge, Random Forest, XGBoost, LightGBM
2. 深度学习模型：LSTM, GRU
3. 模型包装器：统一接口，支持自动数据格式转换
4. 模型工厂：根据配置创建模型实例
"""

from .base_models import BaseModels
from .deep_learning_models import DeepLearningModels, LSTMModel, GRUModel
from .model_trainer import ModelTrainer
from .model_wrapper import (
    BaseModelWrapper,
    MLModelWrapper,
    DLModelWrapper,
    ModelFactory
)

__all__ = [
    'BaseModels',
    'DeepLearningModels',
    'LSTMModel',
    'GRUModel',
    'ModelTrainer',
    'BaseModelWrapper',
    'MLModelWrapper',
    'DLModelWrapper',
    'ModelFactory'
]
