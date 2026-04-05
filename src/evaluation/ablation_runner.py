"""
消融实验调度器。

功能:
1. 自动运行 6 种模型 × 3 种划分策略 = 18 组实验
2. 记录每组指标
3. 输出 6×3 结果矩阵
4. 保存模型权重
5. 详细实验日志

要求:
- 防数据泄露：仅在划分后对特征标准化
- 内存：每次实验后回收
- 容错：单个模型失败不中断整批
"""

import os
import time
import gc
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.splitter import DataSplitter, SplitResult
from ..models.model_wrapper import ModelFactory, BaseModelWrapper
from .metrics import MetricsCalculator


class ExperimentResult:
    """单次实验结果。"""
    
    def __init__(
        self,
        model_name: str,
        split_strategy: str,
        split_info: Dict[str, Any],
        metrics: Dict[str, float],
        duration: float,
        timestamp: str
    ):
        self.model_name = model_name
        self.split_strategy = split_strategy
        self.split_info = split_info
        self.metrics = metrics
        self.duration = duration
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典。"""
        return {
            'model': self.model_name,
            'split_strategy': self.split_strategy,
            'split_info': str(self.split_info),
            **self.metrics,
            'duration_seconds': self.duration,
            'timestamp': self.timestamp
        }


class AblationRunner:
    """
    消融实验调度。

    职责:
    1. 遍历模型 × 划分组合
    2. 训练与评估
    3. 汇总结果
    4. 保存模型与日志
    """
    
    MODELS = ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru']
    
    SPLIT_STRATEGIES = ['fixed', 'rolling', 'regime']
    
    METRICS = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall']
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        results_dir: str = "results",
        device: str = 'cpu'
    ):
        """
        参数:
            config: 完整配置
            checkpoint_dir: 检查点目录
            log_dir: 日志目录
            results_dir: 结果目录
            device: 'cpu' 或 'cuda'
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.splitter = DataSplitter(config.get('data_split', {}))
        self.metrics_calculator = MetricsCalculator()
        
        self.results: List[ExperimentResult] = []
        self.errors: List[Dict[str, Any]] = []
        
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """配置实验专用日志。"""
        log_file = self.log_dir / "experiment.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        self.logger = logging.getLogger("AblationRunner")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
    
    def run_experiments(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        regime: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None,
        split_strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        运行完整消融网格。

        参数:
            X: 特征 (n_samples, n_features)
            y: 目标 (n_samples,)
            dates: 日期索引
            regime: 周期标签（regime 策略需要）
            model_names: 要跑的模型（默认全部）
            split_strategies: 要跑的划分（默认全部）

        返回:
            结果 DataFrame（约 6 行 × 3 列）
        """
        if model_names is None:
            model_names = self.MODELS
        if split_strategies is None:
            split_strategies = self.SPLIT_STRATEGIES
        
        self.logger.info("=" * 80)
        self.logger.info("消融实验开始")
        self.logger.info(f"模型数: {len(model_names)}, 划分策略数: {len(split_strategies)}")
        self.logger.info(f"总实验数: {len(model_names) * len(split_strategies)}")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        for model_name in model_names:
            for split_strategy in split_strategies:
                self._run_single_experiment(
                    model_name=model_name,
                    split_strategy=split_strategy,
                    X=X,
                    y=y,
                    dates=dates,
                    regime=regime
                )
        
        total_duration = time.time() - start_time
        
        self.logger.info("=" * 80)
        self.logger.info(f"消融实验完成，总耗时 {total_duration:.2f} 秒")
        self.logger.info(f"成功: {len(self.results)}, 失败: {len(self.errors)}")
        self.logger.info("=" * 80)
        
        results_df = self._generate_results_matrix()
        
        self._save_results(results_df)
        
        return results_df
    
    def _run_single_experiment(
        self,
        model_name: str,
        split_strategy: str,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        regime: Optional[np.ndarray] = None
    ) -> None:
        """单次 模型×划分 实验。"""
        experiment_id = f"{model_name}_{split_strategy}"
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.logger.info("-" * 60)
        self.logger.info(f"开始实验: {experiment_id}")
        self.logger.info(f"时间: {timestamp}")
        
        try:
            model_config = self.config.get('models', {}).get(model_name, {})
            
            model = ModelFactory.create_model(
                model_name=model_name,
                config=model_config,
                checkpoint_dir=str(self.checkpoint_dir),
                device=self.device
            )
            
            split_results = list(self.splitter.get_splits(
                strategy=split_strategy,
                X=X,
                y=y,
                dates=dates,
                regime=regime
            ))
            
            if not split_results:
                raise ValueError(f"划分策略 {split_strategy} 未产生任何有效折")
            
            fold_metrics = []
            
            for fold_idx, split_result in enumerate(split_results):
                self.logger.info(
                    f"  折 {fold_idx + 1}/{len(split_results)}: "
                    f"训练={len(split_result.X_train)}, 测试={len(split_result.X_test)}"
                )
                
                # 划分后再标准化，避免泄露
                X_train_scaled, X_test_scaled = self._scale_features(
                    split_result.X_train,
                    split_result.X_test
                )
                
                model.fit(
                    X_train_scaled,
                    split_result.y_train
                )
                
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)
                
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                
                y_test_aligned = split_result.y_test
                if model_name in ['lstm', 'gru']:
                    seq_len = model_config.get('training', {}).get('seq_len', 15)
                    y_test_aligned = split_result.y_test[seq_len - 1:]
                
                fold_metric = self.metrics_calculator.classification_metrics(
                    y_test_aligned,
                    y_pred,
                    y_prob
                )
                fold_metrics.append(fold_metric)
            
            aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
            
            model_path = model.save_model()
            
            duration = time.time() - start_time
            
            result = ExperimentResult(
                model_name=model_name,
                split_strategy=split_strategy,
                split_info=split_results[0].split_info,
                metrics=aggregated_metrics,
                duration=duration,
                timestamp=timestamp
            )
            self.results.append(result)
            
            self.logger.info(f"  实验成功: {experiment_id}")
            self.logger.info(f"  指标: {aggregated_metrics}")
            self.logger.info(f"  耗时: {duration:.2f} 秒")
            self.logger.info(f"  模型已保存: {model_path}")
            
        except Exception as e:
            duration = time.time() - start_time
            error_info = {
                'model': model_name,
                'split_strategy': split_strategy,
                'error': str(e),
                'duration': duration,
                'timestamp': timestamp
            }
            self.errors.append(error_info)
            
            self.logger.error(f"  实验失败: {experiment_id}")
            self.logger.error(f"  错误: {str(e)}")
        
        finally:
            gc.collect()
    
    def _scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征标准化（仅在训练集上 fit）。

        参数:
            X_train: 训练特征
            X_test: 测试特征

        返回:
            标准化后的训练集与测试集
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """多折指标取均值（及标准差）。"""
        if not fold_metrics:
            return {}
        
        if len(fold_metrics) == 1:
            return fold_metrics[0]
        
        aggregated = {}
        for metric_name in fold_metrics[0].keys():
            values = [m[metric_name] for m in fold_metrics]
            aggregated[metric_name] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
        
        return aggregated
    
    def _generate_results_matrix(self) -> pd.DataFrame:
        """构建 6×3 准确率矩阵。"""
        matrix_data = []
        
        for model_name in self.MODELS:
            row_data = {'model': model_name}
            
            for split_strategy in self.SPLIT_STRATEGIES:
                result = next(
                    (r for r in self.results
                     if r.model_name == model_name and r.split_strategy == split_strategy),
                    None
                )
                
                if result:
                    metric_key = 'Accuracy'
                    row_data[split_strategy] = result.metrics.get(metric_key, np.nan)
                else:
                    row_data[split_strategy] = np.nan
            
            matrix_data.append(row_data)
        
        df = pd.DataFrame(matrix_data)
        df.set_index('model', inplace=True)
        
        return df
    
    def _save_results(self, results_df: pd.DataFrame) -> None:
        """写出 CSV。"""
        matrix_file = self.results_dir / "ablation_results.csv"
        results_df.to_csv(matrix_file)
        self.logger.info(f"结果矩阵已保存: {matrix_file}")
        
        detailed_results = [r.to_dict() for r in self.results]
        detailed_df = pd.DataFrame(detailed_results)
        detailed_file = self.results_dir / "ablation_results_detailed.csv"
        detailed_df.to_csv(detailed_file, index=False)
        self.logger.info(f"详细结果已保存: {detailed_file}")
        
        if self.errors:
            errors_df = pd.DataFrame(self.errors)
            errors_file = self.results_dir / "ablation_errors.csv"
            errors_df.to_csv(errors_file, index=False)
            self.logger.info(f"错误记录已保存: {errors_file}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """供日志使用的摘要字典。"""
        return {
            'total_experiments': len(self.MODELS) * len(self.SPLIT_STRATEGIES),
            'successful': len(self.results),
            'failed': len(self.errors),
            'results_matrix': self._generate_results_matrix().to_dict(),
            'best_single_run': self._get_best_single_run(), 
            'most_stable_model': self._get_most_stable_model(),      
            'best_split_strategy': self._get_best_split_strategy()
        }
    
    def _get_most_stable_model(self) -> str:
        """各划分上平均准确率最高的模型（最稳健）。"""
        model_scores = {}
        for model_name in self.MODELS:
            scores = [r.metrics.get('Accuracy', 0) for r in self.results if r.model_name == model_name]
            if scores:
                model_scores[model_name] = np.mean(scores)
        return max(model_scores, key=model_scores.get) if model_scores else None

    def _get_best_single_run(self) -> str:
        """单次实验准确率最高的组合。"""
        if not self.results: return None
        best_run = max(self.results, key=lambda r: r.metrics.get('Accuracy', 0))
        return f"{best_run.model_name} (在 {best_run.split_strategy} 划分下: {best_run.metrics.get('Accuracy', 0):.4f})"

    def _get_best_split_strategy(self) -> str:
        """平均准确率最高的划分策略。"""
        strategy_scores = {}
        for strategy in self.SPLIT_STRATEGIES:
            scores = [r.metrics.get('Accuracy', 0) for r in self.results if r.split_strategy == strategy]
            if scores:
                strategy_scores[strategy] = np.mean(scores)
        return max(strategy_scores, key=strategy_scores.get) if strategy_scores else None

    def print_results_table(self) -> None:
        """在标准输出打印矩阵。"""
        results_df = self._generate_results_matrix()
        
        print("\n" + "=" * 80)
        print("消融实验结果矩阵（Accuracy）")
        print("=" * 80)
        print(results_df.to_string())
        print("=" * 80)
        
        print(f"\n最佳单次表现: {self._get_best_single_run()}")
        print(f"最稳健模型 (跨策略均值): {self._get_most_stable_model()}")
        print(f"最佳划分策略 (跨模型均值): {self._get_best_split_strategy()}")