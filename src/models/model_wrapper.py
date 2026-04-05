"""
模型包装器基类与统一接口。

功能:
1. 统一的 fit / predict / save 接口
2. 支持传统机器学习与深度学习
3. RNN 所需时自动做 2D→3D 变换
4. 模型权重的保存与加载
"""

import abc
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import gc


class BaseModelWrapper(abc.ABC):
    """
    模型包装器抽象基类。

    子类须实现 fit 与 predict。
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        参数:
            model_name: 模型名称
            config: 超参数字典
            checkpoint_dir: 检查点目录
        """
        self.model_name = model_name
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.is_fitted = False
        self._scaler = None
        self._feature_names = None
    
    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        训练模型。

        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别标签。

        参数:
            X: 特征矩阵

        返回:
            预测标签
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率。

        参数:
            X: 特征矩阵

        返回:
            概率矩阵
        """
        # 默认实现；子类可覆盖
        pred = self.predict(X)
        # 二分类兜底
        if pred.ndim == 1:
            return np.stack([1 - pred, pred], axis=1)
        return pred
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        将模型持久化到磁盘（pickle）。

        参数:
            path: 输出路径；默认 checkpoint_dir/model_name.pkl

        返回:
            写入路径字符串
        """
        if path is None:
            path = self.checkpoint_dir / f"{self.model_name}.pkl"
        else:
            path = Path(path)
        
        # 模型与元数据一并保存
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_names': self._feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return str(path)
    
    def load_model(self, path: str) -> None:
        """
        从 pickle 加载 sklearn 风格模型。

        参数:
            path: pickle 文件路径
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self._feature_names = model_data.get('feature_names')
    
    def get_params(self) -> Dict[str, Any]:
        """
        若底层模型支持 get_params，则返回之。

        返回:
            参数字典
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.config.get('params', {})
    
    def set_feature_names(self, feature_names: list) -> None:
        """
        保存特征列名，便于解释与调试。

        参数:
            feature_names: 列名列表
        """
        self._feature_names = feature_names
    
    def get_feature_names(self) -> Optional[list]:
        """
        返回已保存的特征名。

        返回:
            列名列表或 None
        """
        return self._feature_names
    
    def cleanup(self) -> None:
        """释放引用并执行 gc。"""
        gc.collect()


class MLModelWrapper(BaseModelWrapper):
    """
    传统机器学习包装器（Ridge、随机森林、XGBoost、LightGBM）。
    """
    
    def __init__(
        self,
        model_name: str,
        model_class,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        参数:
            model_name: 模型标识
            model_class: 估计器类
            config: 超参数
            checkpoint_dir: 检查点目录
        """
        super().__init__(model_name, config, checkpoint_dir)
        self.model_class = model_class
        
        params = self.config.get('params', {})
        self.model = model_class(**params)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        训练；若提供验证集且支持 eval_set，则可用于早停（XGBoost/LightGBM）。

        参数:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
        """
        if X_val is not None and y_val is not None:
            if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                eval_set = [(X_val, y_val)]
                
                fit_params = self.config.get('training', {})
                early_stopping_rounds = fit_params.get('early_stopping_rounds')
                
                if early_stopping_rounds:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        参数:
            X: 特征 (n_samples, n_features)

        返回:
            标签 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        参数:
            X: 特征

        返回:
            概率 (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # 如 Ridge：用预测值构造伪概率
            pred = self.predict(X)
            return np.stack([1 - pred, pred], axis=1)


class DLModelWrapper(BaseModelWrapper):
    """
    深度学习包装器（LSTM、GRU），内含 2D→3D 序列构造。
    """
    
    def __init__(
        self,
        model_name: str,
        model_class,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints",
        device: str = 'cpu'
    ):
        """
        参数:
            model_name: 模型标识
            model_class: Torch 模块类
            config: 超参数
            checkpoint_dir: 检查点目录
            device: 'cpu' 或 'cuda'
        """
        super().__init__(model_name, config, checkpoint_dir)
        self.model_class = model_class
        self.device = torch.device(device)
        
        training_config = self.config.get('training', {})
        self.seq_len = training_config.get('seq_len', 15)
        self.batch_size = training_config.get('batch_size', 64)
        self.epochs = training_config.get('epochs', 100)
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.patience = training_config.get('early_stopping_patience', 10)
        
        # 在 fit 中根据 input_size 实例化
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.input_size = None
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        滑动窗口堆叠为 (batch, seq_len, n_features)。

        参数:
            X: 二维数组 (n_samples, n_features)

        返回:
            三维数组 (n_samples - seq_len + 1, seq_len, n_features)
        """
        n_samples, n_features = X.shape
        
        sequences = []
        for i in range(n_samples - self.seq_len + 1):
            sequences.append(X[i:i + self.seq_len])
        
        return np.array(sequences)
    
    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        构造序列张量并对齐标签。

        参数:
            X: 特征
            y: 目标（可选）

        返回:
            (X_seq, y_seq) 或 (X_seq, None)
        """
        X_seq = self._create_sequences(X)
        
        if y is not None:
            y_seq = y[self.seq_len - 1:]
            return X_seq, y_seq
        
        return X_seq, None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        训练 PyTorch 模型，可选验证集与早停。

        参数:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征
            y_val: 验证标签
        """
        self.input_size = X_train.shape[1]
        
        X_train_seq, y_train_seq = self._prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_data(X_val, y_val)
        else:
            X_val_seq = y_val_seq = None
        
        params = self.config.get('params', {})
        self.model = self.model_class(
            input_size=self.input_size,
            **params
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        X_train_t = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        
        if X_val_seq is not None:
            X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        else:
            X_val_t = y_val_t = None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = self._train_epoch(X_train_t, y_train_t)
            
            if X_val_t is not None:
                self.model.eval()
                val_loss = self._validate(X_val_t, y_val_t)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    break
        
        self._load_checkpoint()
        self.is_fitted = True
    
    def _train_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        参数:
            X: 特征批次张量
            y: 目标张量

        返回:
            本 epoch 平均损失
        """
        n_samples = len(X)
        n_batches = n_samples // self.batch_size + (1 if n_samples % self.batch_size else 0)
        
        total_loss = 0.0
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, n_samples)
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * (end - start)
        
        return total_loss / n_samples
    
    def _validate(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        验证集损失。

        参数:
            X: 特征张量
            y: 目标张量

        返回:
            平均损失
        """
        with torch.no_grad():
            n_samples = len(X)
            n_batches = n_samples // self.batch_size + (1 if n_samples % self.batch_size else 0)
            
            total_loss = 0.0
            
            for i in range(n_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, n_samples)
                
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * (end - start)
        
        return total_loss / n_samples
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        参数:
            X: 特征 (n_samples, n_features)

        返回:
            标签 (n_samples - seq_len + 1,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        X_seq, _ = self._prepare_data(X)
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_t))
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        参数:
            X: 特征

        返回:
            概率 (n_samples - seq_len + 1, 2)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        X_seq, _ = self._prepare_data(X)
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_t))
            probs = outputs.cpu().numpy().flatten()
        
        return np.stack([1 - probs, probs], axis=1)
    
    def _save_checkpoint(self) -> None:
        """将当前最优权重写入磁盘。"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'input_size': self.input_size
        }
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def _load_checkpoint(self) -> None:
        """从磁盘加载最优检查点。"""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        参数:
            path: 输出 .pt 路径

        返回:
            写入路径
        """
        if path is None:
            path = self.checkpoint_dir / f"{self.model_name}.pt"
        else:
            path = Path(path)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_size': self.input_size,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted
        }
        
        torch.save(checkpoint, path)
        return str(path)
    
    def load_model(self, path: str) -> None:
        """
        参数:
            path: Torch 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.input_size = checkpoint['input_size']
        params = checkpoint['config'].get('params', {})
        self.model = self.model_class(
            input_size=self.input_size,
            **params
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.config = checkpoint['config']
        self.is_fitted = checkpoint['is_fitted']
    
    def cleanup(self) -> None:
        """清空 CUDA 缓存并 gc。"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ModelFactory:
    """
    根据名称创建 sklearn 或 torch 包装器。
    """
    
    ML_MODELS = {
        'ridge': 'RidgeClassifier',
        'random_forest': 'RandomForestClassifier',
        'xgboost': 'XGBClassifier',
        'lightgbm': 'LGBMClassifier'
    }
    
    @staticmethod
    def create_model(
        model_name: str,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        device: str = 'cpu'
    ) -> BaseModelWrapper:
        """
        参数:
            model_name: 支持的模型名之一
            config: 超参数
            checkpoint_dir: 保存目录
            device: torch 设备字符串

        返回:
            包装器实例
        """
        if model_name in ['lstm', 'gru']:
            from .deep_learning_models import LSTMModel, GRUModel
            
            model_classes = {
                'lstm': LSTMModel,
                'gru': GRUModel
            }
            
            return DLModelWrapper(
                model_name=model_name,
                model_class=model_classes[model_name],
                config=config,
                checkpoint_dir=checkpoint_dir,
                device=device
            )
        
        else:
            from .base_models import BaseModels
            
            base_models = BaseModels()
            model = base_models.get_model(model_name)
            
            wrapper = MLModelWrapper(
                model_name=model_name,
                model_class=type(model),
                config={},
                checkpoint_dir=checkpoint_dir
            )
            wrapper.model = model
            
            return wrapper
    
    @staticmethod
    def get_supported_models() -> list:
        """返回支持的模型名称列表。"""
        return ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru']
