"""
模型训练器模块

负责：
1. 模型训练与验证
2. 超参数调优
3. 模型保存与加载
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import log_loss

from .base_models import BaseModels
from .deep_learning_models import DeepLearningModels, LSTMModel, GRUModel


class ModelTrainer:
    """
    模型训练器
    
    支持：
    1. 传统机器学习模型训练
    2. 深度学习模型训练
    3. 超参数调优
    4. 模型保存与加载
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = 'cpu'):
        """
        初始化模型训练器
        
        参数：
            checkpoint_dir: 模型保存目录
            device: 设备 ('cpu' 或 'cuda')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.base_models = BaseModels()
        self.dl_models = DeepLearningModels(device=device)
        
        self.trained_models: Dict[str, Any] = {}
        
    def train_ml_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        训练机器学习模型
        
        参数：
            model_name: 模型名称 ('ridge', 'random_forest', 'xgboost', 'lightgbm')
            X_train: 训练特征
            y_train: 训练目标
            model_params: 模型参数
            
        返回：
            训练好的模型
        """
        if model_params is None:
            model_params = {}
        
        # 获取模型
        model = self.base_models.get_model(model_name, **model_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 保存模型
        self.trained_models[model_name] = model
        
        return model
    
    def train_deep_learning_model(
        self,
        model_type: str,  # 取值 'lstm' 或 'gru'
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seq_len: int = 15,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        训练深度学习模型
        
        参数：
            model_type: 模型类型 ('lstm' or 'gru')
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            seq_len: 序列长度
            hidden_size: 隐藏层单元数
            num_layers: 层数
            dropout: Dropout 比例
            bidirectional: 是否双向
            batch_size: 批次大小
            learning_rate: 学习率
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            verbose: 是否显示训练进度
            
        返回：
            (model, history): 训练好的模型和训练历史
        """
        input_size = X_train.shape[1]
        
        # 准备序列数据
        X_train_seq, y_train_seq = self.dl_models.prepare_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = self.dl_models.prepare_sequences(X_val, y_val, seq_len)
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        if model_type == 'lstm':
            model = self.dl_models.create_lstm_model(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type == 'gru':
            model = self.dl_models.create_gru_model(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 获取损失函数和优化器
        criterion = self.dl_models.get_loss_function()
        optimizer = self.dl_models.get_optimizer(model, learning_rate)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), self.checkpoint_dir / f'{model_type}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(self.checkpoint_dir / f'{model_type}_best.pth'))
        
        # 保存模型
        model_name = f'{model_type}_seq{seq_len}_h{hidden_size}_l{num_layers}'
        self.trained_models[model_name] = model
        
        return model, history
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'neg_log_loss'
    ) -> Any:
        """
        超参数调优（使用 TimeSeriesSplit）
        
        参数：
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练目标
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评分指标
            
        返回：
            最佳模型
        """
        # 获取基础模型
        base_model = self.base_models.get_model(model_name)
        
        # 创建 TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳得分: {grid_search.best_score_:.4f}")
        
        # 保存最佳模型
        self.trained_models[f'{model_name}_best'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        保存模型
        
        参数：
            model: 模型对象
            model_name: 模型名称
        """
        if isinstance(model, nn.Module):
            # PyTorch 模型
            torch.save(model.state_dict(), self.checkpoint_dir / f'{model_name}.pth')
        else:
            # sklearn 模型
            joblib.dump(model, self.checkpoint_dir / f'{model_name}.pkl')
    
    def load_model(self, model_name: str, model_type: str = 'sklearn') -> Any:
        """
        加载模型
        
        参数：
            model_name: 模型名称
            model_type: 模型类型 ('sklearn' or 'pytorch')
            
        返回：
            加载的模型
        """
        if model_type == 'pytorch':
            # PyTorch 模型需要先创建模型实例
            state_dict = torch.load(self.checkpoint_dir / f'{model_name}.pth')
            return state_dict
        else:
            # sklearn 模型
            return joblib.load(self.checkpoint_dir / f'{model_name}.pkl')
    
    def predict(self, model: Any, X: np.ndarray, model_type: str = 'sklearn') -> np.ndarray:
        """
        预测
        
        参数：
            model: 模型对象
            X: 特征数据
            model_type: 模型类型
            
        返回：
            预测结果
        """
        if model_type == 'pytorch':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                predictions = outputs.cpu().numpy().flatten()
        else:
            predictions = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        
        return predictions
    
    def get_all_trained_models(self) -> Dict[str, Any]:
        """
        获取所有训练好的模型
        
        返回：
            模型字典
        """
        return self.trained_models
