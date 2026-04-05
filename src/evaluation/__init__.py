"""
评估模块初始化

包含：
1. 指标计算：分类、回归、回测指标
2. 模型评估：单模型评估、多模型比较
3. 模型解释：SHAP、特征重要性
4. 消融实验：自动化实验调度器
"""

from .metrics import MetricsCalculator
from .model_explainer import ModelExplainer
from .evaluator import ModelEvaluator
from .ablation_runner import AblationRunner, ExperimentResult

__all__ = [
    'MetricsCalculator',
    'ModelExplainer',
    'ModelEvaluator',
    'AblationRunner',
    'ExperimentResult'
]
