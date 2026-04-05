"""
模型评估器模块

功能：
1. 数据切分：按时间顺序进行整体样本切分（60% 训练，20% 验证，20% 测试）
2. 模型评估与比较
3. 消融实验
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from pathlib import Path

from .metrics import MetricsCalculator
from .model_explainer import ModelExplainer


class ModelEvaluator:
    """
    模型评估器
    
    负责：
    1. 数据切分
    2. 模型评估
    3. 消融实验
    4. 结果可视化
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        初始化模型评估器
        
        参数：
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.model_explainer = ModelExplainer(output_dir)
        
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
    
    def time_series_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序切分数据集
        
        参数：
            df: 输入 DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        返回：
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"数据切分结果:")
        print(f"  训练集: {len(train_df)} 样本 ({train_df.index[0]} 至 {train_df.index[-1]})")
        print(f"  验证集: {len(val_df)} 样本 ({val_df.index[0]} 至 {val_df.index[-1]})")
        print(f"  测试集: {len(test_df)} 样本 ({test_df.index[0]} 至 {test_df.index[-1]})")
        
        return train_df, val_df, test_df
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        calculate_backtest: bool = False,
        price_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        评估单个模型
        
        参数：
            model: 训练好的模型
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称列表
            calculate_backtest: 是否计算回测指标
            price_data: 价格数据（用于回测）
            
        返回：
            评估结果字典
        """
        results = {
            'model_name': model_name,
            'predictions': {},
            'metrics': {}
        }
        
        # 预测
        if hasattr(model, 'predict_proba'):
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_val_prob = model.predict_proba(X_val)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            y_train_pred = (y_train_prob > 0.5).astype(int)
            y_val_pred = (y_val_prob > 0.5).astype(int)
            y_test_pred = (y_test_prob > 0.5).astype(int)
        else:
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            y_train_prob = y_train_pred
            y_val_prob = y_val_pred
            y_test_prob = y_test_pred
        
        results['predictions'] = {
            'train': (y_train_pred, y_train_prob),
            'val': (y_val_pred, y_val_prob),
            'test': (y_test_pred, y_test_prob)
        }
        
        # 计算分类指标
        results['metrics']['train'] = self.metrics_calculator.classification_metrics(
            y_train, y_train_pred, y_train_prob
        )
        results['metrics']['val'] = self.metrics_calculator.classification_metrics(
            y_val, y_val_pred, y_val_prob
        )
        results['metrics']['test'] = self.metrics_calculator.classification_metrics(
            y_test, y_test_pred, y_test_prob
        )
        
        # 计算回测指标
        if calculate_backtest and price_data is not None:
            # 使用测试集预测计算策略收益
            test_returns = self._calculate_strategy_returns(
                y_test_prob, price_data.iloc[-len(y_test):]
            )
            results['metrics']['backtest'] = self.metrics_calculator.backtest_metrics(test_returns)
        
        # 保存结果
        self.evaluation_results[model_name] = results
        
        return results
    
    def _calculate_strategy_returns(
        self,
        y_prob: np.ndarray,
        price_data: pd.Series
    ) -> np.ndarray:
        """
        计算策略收益
        
        参数：
            y_prob: 预测概率
            price_data: 价格数据
            
        返回：
            策略收益率序列
        """
        # 生成交易信号
        signals = (y_prob > 0.5).astype(int)
        
        # 计算收益率
        returns = price_data.pct_change().fillna(0)
        
        # 策略收益
        strategy_returns = signals.shift(1) * returns  # 使用前一日的信号
        
        return strategy_returns.dropna().values
    
    def compare_models(
        self,
        results: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        参数：
            results: 评估结果字典，默认使用 self.evaluation_results
            
        返回：
            比较结果 DataFrame
        """
        if results is None:
            results = self.evaluation_results
        
        comparison_data = []
        
        for model_name, model_results in results.items():
            for dataset in ['train', 'val', 'test']:
                if dataset in model_results['metrics']:
                    row = {
                        'Model': model_name,
                        'Dataset': dataset
                    }
                    row.update(model_results['metrics'][dataset])
                    comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def print_evaluation_report(
        self,
        model_name: str,
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        打印评估报告
        
        参数：
            model_name: 模型名称
            results: 评估结果字典
        """
        if results is None:
            results = self.evaluation_results.get(model_name)
        
        if results is None:
            print(f"未找到模型 {model_name} 的评估结果")
            return
        
        print(f"\n{'='*60}")
        print(f"模型评估报告: {model_name}")
        print(f"{'='*60}")
        
        for dataset in ['train', 'val', 'test']:
            if dataset in results['metrics']:
                print(f"\n{dataset.upper()} 集指标:")
                for metric, value in results['metrics'][dataset].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
        
        if 'backtest' in results['metrics']:
            print(f"\n回测指标:")
            for metric, value in results['metrics']['backtest'].items():
                if metric == 'Max_Drawdown':
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.2%}")
        
        print(f"{'='*60}\n")
    
    def plot_model_comparison(
        self,
        metric: str = 'Accuracy',
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        save: bool = True
    ) -> None:
        """
        绘制模型比较图
        
        参数：
            metric: 要比较的指标
            results: 评估结果字典
            save: 是否保存图表
        """
        if results is None:
            results = self.evaluation_results
        
        models = list(results.keys())
        datasets = ['train', 'val', 'test']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            values = []
            for model in models:
                if dataset in results[model]['metrics'] and metric in results[model]['metrics'][dataset]:
                    values.append(results[model]['metrics'][dataset][metric])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=dataset.capitalize())
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'model_comparison_{metric.lower()}.png', dpi=300)
        
        plt.show()
    
    def ablation_study(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_groups: Dict[str, List[int]],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        消融实验：测试不同特征组的影响
        
        参数：
            model: 基础模型
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            feature_groups: 特征组字典 {'group_name': [feature_indices]}
            feature_names: 特征名称列表
            
        返回：
            消融实验结果 DataFrame
        """
        ablation_results = []
        
        # 全特征基线
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        baseline_metrics = self.metrics_calculator.classification_metrics(y_test, y_pred, y_prob)
        
        ablation_results.append({
            'Feature_Group': 'All_Features',
            'Num_Features': X_train.shape[1],
            **baseline_metrics
        })
        
        # 逐个移除特征组
        for group_name, feature_indices in feature_groups.items():
            # 创建移除该特征组后的特征集
            mask = np.ones(X_train.shape[1], dtype=bool)
            mask[feature_indices] = False
            
            X_train_ablated = X_train[:, mask]
            X_test_ablated = X_test[:, mask]
            
            # 训练和评估
            model_ablated = type(model)(**model.get_params())
            model_ablated.fit(X_train_ablated, y_train)
            
            y_pred_ablated = model_ablated.predict(X_test_ablated)
            y_prob_ablated = model_ablated.predict_proba(X_test_ablated)[:, 1] if hasattr(model_ablated, 'predict_proba') else y_pred_ablated
            
            metrics = self.metrics_calculator.classification_metrics(y_test, y_pred_ablated, y_prob_ablated)
            
            ablation_results.append({
                'Feature_Group': f'Without_{group_name}',
                'Num_Features': X_train_ablated.shape[1],
                **metrics
            })
        
        ablation_df = pd.DataFrame(ablation_results)
        
        return ablation_df
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有评估结果
        
        返回：
            评估结果字典
        """
        return self.evaluation_results
