"""
深度学习模型模块

包含：
1. LSTM 模型（1-2层，32-256单元）
2. GRU 模型（1-2层，32-256单元）
3. 全连接 Dense(1) + Sigmoid 输出层
4. 二元交叉熵损失
5. 张量重构：步长为 1 的滑动窗口
6. 内存优化：使用 NumPy 的 as_strided（stride_tricks）
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from numpy.lib.stride_tricks import as_strided


class LSTMModel(nn.Module):
    """
    LSTM 二分类模型
    
    架构：
    - LSTM 层（1-2层，32-256单元）
    - Dense(1) + Sigmoid
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        初始化 LSTM 模型
        
        参数：
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: LSTM 层数
            dropout: Dropout 比例
            bidirectional: 是否使用双向 LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量 (batch_size, seq_len, input_size)
            
        返回：
            输出张量 (batch_size, 1)
        """
        # LSTM 层
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向：拼接前向和后向的最后隐藏状态
            last_output = torch.cat([
                h_n[-2],  # 前向最后层
                h_n[-1]   # 后向最后层
            ], dim=1)
        else:
            last_output = h_n[-1]  # 单向：取最后层的隐藏状态
        
        # 全连接层
        out = self.fc(last_output)
        
        # Sigmoid 激活
        out = self.sigmoid(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU 二分类模型
    
    架构：
    - GRU 层（1-2层，32-256单元）
    - Dense(1) + Sigmoid
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        初始化 GRU 模型
        
        参数：
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: GRU 层数
            dropout: Dropout 比例
            bidirectional: 是否使用双向 GRU
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量 (batch_size, seq_len, input_size)
            
        返回：
            输出张量 (batch_size, 1)
        """
        # GRU 层
        gru_out, h_n = self.gru(x)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向：拼接前向和后向的最后隐藏状态
            last_output = torch.cat([
                h_n[-2],  # 前向最后层
                h_n[-1]   # 后向最后层
            ], dim=1)
        else:
            last_output = h_n[-1]  # 单向：取最后层的隐藏状态
        
        # 全连接层
        out = self.fc(last_output)
        
        # Sigmoid 激活
        out = self.sigmoid(out)
        
        return out


class DeepLearningModels:
    """
    深度学习模型管理类
    
    负责：
    1. 模型创建
    2. 张量重构（滑动窗口）
    3. 内存优化（as_strided）
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化深度学习模型管理器
        
        参数：
            device: 设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        
    def create_lstm_model(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ) -> LSTMModel:
        """
        创建 LSTM 模型
        
        参数：
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: LSTM 层数
            dropout: Dropout 比例
            bidirectional: 是否使用双向 LSTM
            
        返回：
            LSTMModel 实例
        """
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        model = model.to(self.device)
        self.model = model
        return model
    
    def create_gru_model(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ) -> GRUModel:
        """
        创建 GRU 模型
        
        参数：
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: GRU 层数
            dropout: Dropout 比例
            bidirectional: 是否使用双向 GRU
            
        返回：
            GRUModel 实例
        """
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        model = model.to(self.device)
        self.model = model
        return model
    
    @staticmethod
    def create_sequences_with_strides(
        data: np.ndarray,
        seq_len: int
    ) -> np.ndarray:
        """
        使用 as_strided 创建滑动窗口序列（内存优化）
        
        红线要求：
        1. 使用视图切片而非复制，避免内存溢出
        2. 必须立即 .copy() 确保内存安全，防止视图共享导致的数据污染
        
        参数：
            data: 输入数据 (n_samples, n_features)
            seq_len: 序列长度
            
        返回：
            序列数据 (n_samples - seq_len + 1, seq_len, n_features)
        """
        n_samples, n_features = data.shape
        
        # 使用 as_strided 创建视图
        shape = (n_samples - seq_len + 1, seq_len, n_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])
        
        sequences = as_strided(
            data,
            shape=shape,
            strides=strides
        )
        
        # 修复：立即复制以确保内存安全
        # as_strided 返回的是视图，多个窗口共享底层内存
        # 如果不复制，修改一个窗口会影响所有窗口
        return sequences.copy()
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据
        
        红线要求：特征窗口 [t-seq_len+1, t] 对应的目标必须是 y[t+1]
        即：窗口最后一天是 t，预测的是 t+1 的涨跌
        
        参数：
            X: 特征数据 (n_samples, n_features)
            y: 目标变量 (n_samples,)，其中 y[t] 是 t+1 时刻的涨跌
            seq_len: 序列长度
            
        返回：
            (X_seq, y_seq): 序列特征和对应的目标
        """
        # 创建序列
        X_seq = self.create_sequences_with_strides(X, seq_len)
        
        # 目标变量对齐逻辑：
        # X_seq[i] 包含 X[i] 到 X[i+seq_len-1] 的数据
        # 窗口最后一天是 i+seq_len-1，对应的目标应该是 y[i+seq_len-1]
        # 因为 y[t] 已经是 t+1 时刻的涨跌（在 create_target 中已平移）
        y_seq = y[seq_len - 1:]
        
        # 断言验证时间对齐
        assert len(X_seq) == len(y_seq), \
            f"序列长度不匹配: X_seq={len(X_seq)}, y_seq={len(y_seq)}"
        
        return X_seq, y_seq
    
    def get_model_config(self) -> dict:
        """
        获取默认模型配置
        
        返回：
            配置字典
        """
        return {
            'lstm': {
                'hidden_size': [32, 64, 128, 256],
                'num_layers': [1, 2],
                'dropout': [0.0, 0.2, 0.5],
                'bidirectional': [False, True]
            },
            'gru': {
                'hidden_size': [32, 64, 128, 256],
                'num_layers': [1, 2],
                'dropout': [0.0, 0.2, 0.5],
                'bidirectional': [False, True]
            },
            'training': {
                'seq_len': [7, 15, 30],  # 滑动窗口搜索空间
                'batch_size': [32, 64, 128],
                'learning_rate': [0.001, 0.0001],
                'epochs': [50, 100, 200]
            }
        }
    
    def get_loss_function(self) -> nn.Module:
        """
        获取损失函数（二元交叉熵）
        
        优化：使用 BCEWithLogitsLoss（logits 上的二元交叉熵）以获得更好的数值稳定性。
        若使用该损失，模型 forward 中不应再接 Sigmoid，而在推理时单独施加 Sigmoid。
        
        返回：
            BCEWithLogitsLoss 实例
        """
        return nn.BCEWithLogitsLoss()
    
    def get_optimizer(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ) -> torch.optim.Optimizer:
        """
        获取优化器
        
        参数：
            model: 模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            
        返回：
            Adam 优化器
        """
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
