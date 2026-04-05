"""
配置加载模块

支持分层配置文件结构：
- base_config.yaml: 通用配置
- model_params/: 模型专属超参数
- data_split.yaml: 数据划分策略参数
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy


class ConfigLoader:
    """
    配置加载器
    
    支持 YAML 和 JSON 格式的配置文件
    支持分层配置合并与覆盖
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        参数：
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self._loaded_files: List[str] = []
    
    def load_yaml(self, filename: str, merge: bool = True) -> Dict[str, Any]:
        """
        加载 YAML 配置文件
        
        参数：
            filename: 配置文件名或路径
            merge: 是否合并到现有配置
            
        返回：
            配置字典
        """
        # 支持相对路径和绝对路径
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if merge:
            self.config = self._deep_merge(self.config, config)
        
        self._loaded_files.append(str(file_path))
        return config
    
    def load_json(self, filename: str, merge: bool = True) -> Dict[str, Any]:
        """
        加载 JSON 配置文件
        
        参数：
            filename: 配置文件名或路径
            merge: 是否合并到现有配置
            
        返回：
            配置字典
        """
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if merge:
            self.config = self._deep_merge(self.config, config)
        
        self._loaded_files.append(str(file_path))
        return config
    
    def load_base_config(self, filename: str = "base_config.yaml") -> Dict[str, Any]:
        """
        加载基础配置文件
        
        参数：
            filename: 基础配置文件名
            
        返回：
            配置字典
        """
        return self.load_yaml(filename)
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        加载模型专属配置
        
        参数：
            model_name: 模型名称 (ridge, random_forest, xgboost, lightgbm, lstm, gru)
            
        返回：
            模型配置字典
        """
        config_path = f"model_params/{model_name}.yaml"
        return self.load_yaml(config_path)
    
    def load_split_config(self, filename: str = "data_split.yaml") -> Dict[str, Any]:
        """
        加载数据划分配置
        
        参数：
            filename: 划分配置文件名
            
        返回：
            划分配置字典
        """
        return self.load_yaml(filename)
    
    def load_all_configs(
        self,
        base_config: str = "base_config.yaml",
        split_config: str = "data_split.yaml"
    ) -> Dict[str, Any]:
        """
        加载所有配置文件
        
        参数：
            base_config: 基础配置文件名
            split_config: 划分配置文件名
            
        返回：
            合并后的配置字典
        """
        # 清空现有配置
        self.config = {}
        self._loaded_files = []
        
        # 加载基础配置
        self.load_base_config(base_config)
        
        # 加载数据划分配置
        self.load_split_config(split_config)
        
        return self.config
    
    def load_model_params(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量加载模型参数
        
        参数：
            model_names: 模型名称列表
            
        返回：
            模型参数字典 {model_name: model_config}
        """
        model_params = {}
        for model_name in model_names:
            try:
                config = self.load_model_config(model_name)
                model_params[model_name] = config
            except FileNotFoundError:
                # 使用默认配置
                model_params[model_name] = self._get_default_model_config(model_name)
        
        return model_params
    
    def _get_default_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型默认配置
        
        参数：
            model_name: 模型名称
            
        返回：
            默认配置字典
        """
        defaults = {
            'ridge': {
                'model_type': 'ridge',
                'params': {'alpha': 1.0, 'random_state': 42}
            },
            'random_forest': {
                'model_type': 'random_forest',
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            },
            'xgboost': {
                'model_type': 'xgboost',
                'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
            },
            'lightgbm': {
                'model_type': 'lightgbm',
                'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
            },
            'lstm': {
                'model_type': 'lstm',
                'params': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.2},
                'training': {'seq_len': 15, 'batch_size': 64, 'epochs': 100}
            },
            'gru': {
                'model_type': 'gru',
                'params': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.2},
                'training': {'seq_len': 15, 'batch_size': 64, 'epochs': 100}
            }
        }
        return defaults.get(model_name, {})
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        深度合并两个字典
        
        参数：
            base: 基础字典
            update: 更新字典
            
        返回：
            合并后的字典
        """
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def override_from_args(self, **kwargs) -> None:
        """
        从命令行参数覆盖配置
        
        参数：
            **kwargs: 配置键值对，支持点分隔的嵌套键
                例如: models.xgboost.learning_rate=0.05
        """
        for key, value in kwargs.items():
            self.set(key, value)
    
    def get_loaded_files(self) -> List[str]:
        """
        获取已加载的配置文件列表
        
        返回：
            配置文件路径列表
        """
        return self._loaded_files.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        参数：
            key: 配置键（支持点分隔的嵌套键）
            default: 默认值
            
        返回：
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        参数：
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_yaml(self, filename: str, config: Optional[Dict] = None) -> None:
        """
        保存配置到 YAML 文件
        
        参数：
            filename: 配置文件名
            config: 要保存的配置字典
        """
        file_path = self.config_dir / filename
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config or self.config, f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, filename: str, config: Optional[Dict] = None) -> None:
        """
        保存配置到 JSON 文件
        
        参数：
            filename: 配置文件名
            config: 要保存的配置字典
        """
        file_path = self.config_dir / filename
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config or self.config, f, indent=2, ensure_ascii=False)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        返回：
            默认配置字典
        """
        return {
            # 数据配置
            'data': {
                'ohlcv_file': 'data/btc_ohlc.csv',
                'technical_file': 'data/btc_technical_data_raw.csv',
                'macro_file': 'data/macro_data_raw.csv',
                'start_date': '2016-01-01',
                'end_date': '2024-06-30'
            },
            
            # 特征工程配置
            'features': {
                'scaler_window': 252,
                'hampel_window': 5,
                'hampel_n_sigmas': 3.0,
                'pca_components': 2,
                'shap_top_k': 50,
                'rfe_n_features': 30
            },
            
            # 模型配置
            'models': {
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
                },
                'lstm': {
                    'hidden_size': 128,
                    'num_layers': 1,
                    'dropout': 0.2,
                    'seq_len': 15,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'epochs': 100
                },
                'gru': {
                    'hidden_size': 128,
                    'num_layers': 1,
                    'dropout': 0.2,
                    'seq_len': 15,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            },
            
            # 训练配置
            'training': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'random_state': 42
            },
            
            # 回测配置
            'backtesting': {
                'initial_capital': 10000,
                'risk_free_rate': 0.0,
                'signal_threshold': 0.5
            }
        }
    
    def load_default_config(self) -> Dict[str, Any]:
        """
        加载默认配置
        
        返回：
            默认配置字典
        """
        self.config = self.get_default_config()
        return self.config
