# BTC 分类型目标预测系统 | BTC Directional Prediction System

## 项目概述 | Project Overview

对比特币（BTC）价格明日涨跌方向进行二分类预测的量化交易系统，支持自动化消融实验流水线。
A quantitative trading system for predicting the next-day upward/downward direction of Bitcoin (BTC) prices, supporting an automated ablation study pipeline.

### 核心目标 | Core Objectives
- **任务类型 (Task Type)**：二分类预测（明日涨跌方向）| Binary classification (next-day price direction)
- **数据范围 (Data Range)**：日频数据，2016年1月至2024年6月30日 | Daily data, January 2016 to June 30, 2024
- **技术栈 (Tech Stack)**：Python 3.8+, scikit-learn, XGBoost, LightGBM, PyTorch

### 消融实验 | Ablation Study
- **6种预测模型 (6 Models)**：Ridge, Random Forest, XGBoost, LightGBM, LSTM, GRU
- **3种划分策略 (3 Split Strategies)**：80/20固定比例 (Fixed)、滚动时间窗口 (Rolling)、牛熊市周期 (Regime)
- **18组独立实验 (18 Experiments)**：自动遍历所有模型×策略组合 | Automatically iterates through all model × strategy combinations

---

## 快速开始 | Quick Start

```bash
# 1. 安装依赖 | Install dependencies
pip install -r requirements.txt

# 2. 运行完整流程 | Run the complete pipeline
python main.py --mode all

# 3. 运行消融实验（6模型×3策略=18组） | Run ablation study (6 models × 3 strategies = 18 runs)
python main.py --mode ablation

# 4. 指定模型和策略 | Run specific models and split strategies
python main.py --mode ablation --models xgboost lightgbm --splits fixed rolling

# 5. GPU 加速深度学习模型 | Use GPU acceleration for Deep Learning models
python main.py --mode ablation --device cuda
```

---

## 实验结果 | Experimental Results

基于 2016年1月至2024年6月30日 的日频数据，各模型在不同数据划分策略下的测试集准确率：
Test set accuracy of each model under different data split strategies, based on daily data from Jan 2016 to Jun 2024:

| 模型 | fixed | rolling | regime |
|------|-------|---------|--------|
| Ridge | 55.72% | 50.16% | 48.62% |
| Random Forest | 54.43% | 52.20% | 51.99% |
| XGBoost | 52.17% | 50.76% | 52.65% |
| **LightGBM** | **57.17%** | 50.93% | 47.78% |
| LSTM | 48.93% | 51.85% | 48.80% |
| GRU | 49.42% | 51.96% | 48.80% |

**最佳表现 (Best Performance)**：LightGBM 在 fixed 划分下达到 **57.17%** 准确率 | LightGBM achieved **57.17%** accuracy under the fixed split.

---

## 项目结构 | Project Structure

```text
src/
├── data/           # 数据加载与预处理 | Data loading and preprocessing
├── features/       # 特征工程（技术指标、宏观特征、特征选择）| Feature engineering (Technical, Macro, Selection)
├── models/         # 模型开发（传统ML + 深度学习）| Model development (Traditional ML + Deep Learning)
├── evaluation/     # 评估与消融实验 | Evaluation and ablation studies
├── backtesting/    # 回测系统 | Backtesting engine
└── utils/          # 工具（日志、配置加载）| Utilities (Logging, Config loader)

config/
├── base_config.yaml      # 通用配置 | Global base configuration
├── default_config.yaml   # 完整默认配置 | Complete default configuration
├── data_split.yaml       # 数据划分策略参数 | Data split strategy parameters
├── my_config.yaml        # 用户自定义配置 | User custom configuration
└── model_params/         # 模型专属超参数（6个yaml文件）| Model-specific hyperparameters (6 yaml files)

data/                     # 数据目录 | Data directory
├── btc_ohlc.csv          # BTC 量价数据 | BTC Price/Volume data
├── btc_technical_data_raw.csv  # 技术指标数据 | Technical indicators data
└── macro_data_raw.csv    # 宏观数据 | Macroeconomic data

results/                  # 实验结果 | Experimental results
├── ablation_results.csv          # 6×3 结果矩阵表 | 6x3 results matrix
└── ablation_results_detailed.csv # 详细实验结果 | Detailed experimental results

figures/                  # 可视化图表 | Visualizations
├── *_equity_curves.png   # 权益曲线 | Equity curves
├── *_feature_importance.png  # 特征重要性 | Feature importance
└── *_shap_summary.png    # SHAP解释图 | SHAP summary plots

checkpoints/              # 模型检查点 | Model checkpoints
logs/                     # 运行日志 | Execution logs
```

---

## 支持的模型 | Supported Models

| 类型 (Type) | 模型 (Model) | 说明 (Description) |
|------|------|------|
| 传统 ML (Traditional ML) | Ridge | 线性基准模型 | Linear baseline model |
| 传统 ML (Traditional ML) | Random Forest | 随机森林 | Random Forest classifier |
| 传统 ML (Traditional ML) | XGBoost | 梯度提升树 | Extreme Gradient Boosting |
| 传统 ML (Traditional ML) | LightGBM | 轻量级梯度提升 | Light Gradient Boosting Machine |
| 深度学习 (Deep Learning) | LSTM | 长短期记忆网络 | Long Short-Term Memory network |
| 深度学习 (Deep Learning) | GRU | 门控循环单元 | Gated Recurrent Unit |

---

## 数据划分策略 | Data Split Strategies

| 策略 (Strategy) | 说明 (Description) | 参数 (Parameters) |
|------|------|------|
| `fixed` | 80/20 固定比例划分 | 80/20 Fixed ratio split | train_ratio |
| `rolling` | 滚动时间窗口 | Walk-forward validation | window_size, step_size |
| `regime` | 牛熊市周期划分 | Bull/Bear market regime split | validation_mode |

---

## 使用方法 | Usage

### 主程序命令 | Main CLI Commands

```bash
# 完整流程 | Complete pipeline
python main.py --mode all

# 仅训练 | Training only
python main.py --mode train

# 仅评估 | Evaluation only
python main.py --mode evaluate

# 仅回测 | Backtesting only
python main.py --mode backtest

# 消融实验（6模型×3策略=18组）| Ablation study (18 runs)
python main.py --mode ablation

# 指定模型和策略 | Specific models and splits
python main.py --mode ablation --models xgboost lightgbm --splits fixed rolling

# GPU 加速 | GPU acceleration
python main.py --mode ablation --device cuda

# 使用自定义配置 | Use custom configuration
python main.py --config config/my_config.yaml --mode all
```

### 配置系统 | Configuration System

#### 分层配置结构 | Hierarchical Configuration Structure
```text
config/
├── base_config.yaml        # 通用配置（日志路径、数据路径）| Base configs (paths)
├── default_config.yaml     # 完整默认配置 | Default full configs
├── data_split.yaml         # 数据划分策略参数 | Data split parameters
├── my_config.yaml          # 用户自定义配置 | User custom configs
└── model_params/           # 模型专属超参数 | Model hyperparameters
    ├── ridge.yaml
    ├── random_forest.yaml
    ├── xgboost.yaml
    ├── lightgbm.yaml
    ├── lstm.yaml
    └── gru.yaml
```

#### 配置合并规则 | Configuration Merge Rules
1. 首先加载 `base_config.yaml` | Load `base_config.yaml` first.
2. 然后加载 `data_split.yaml`（覆盖同名配置）| Load `data_split.yaml` (overwrites duplicates).
3. 最后加载模型专属配置（如 `model_params/xgboost.yaml`）| Load model-specific configs.
4. 命令行参数可覆盖所有配置 | CLI arguments override all YAML configurations.

---

## 模块说明 | Module Descriptions

### 数据获取与预处理 (src/data/) | Data Fetching & Preprocessing
- **DataPreprocessor**: 数据加载、对齐、合并 | Data loading, alignment, and merging.
- **RollingWindowScaler**: 252天滚动窗口标准化 | 252-day rolling window standardization.
- **DataSplitter**: 三种数据划分策略 | Three data split strategies.
  - `FixedTimeSplit`: 80/20 固定比例划分 | Fixed 80/20 ratio.
  - `RollingWindowSplit`: 滚动时间窗口划分 | Walk-forward Validation.
  - `MarketRegimeSplit`: 牛熊市周期划分 | Bull/bear market regimes.

### 特征工程 (src/features/) | Feature Engineering
- **TechnicalIndicators**: 技术指标计算（趋势、动量、成交量、波动率）| Trend, momentum, volume, volatility.
- **MacroFeatures**: 宏观特征工程（价格指数、利率、就业、流动性）| CPI, interest rates, employment, liquidity.
- **FeatureSelector**: 特征选择（SHAP初筛 + RFE精筛）| Feature selection via SHAP and RFE.
- Hampel Filter 处理异常值 | Hampel Filter for outlier removal.
- PCA 降维（利率族/就业族）| PCA dimensionality reduction for macro groups.

### 模型开发 (src/models/) | Model Development
- **BaseModels**: Ridge, Random Forest, XGBoost, LightGBM
- **DeepLearningModels**: LSTM, GRU（1-2层，32-256单元 | 1-2 layers, 32-256 units）
- **ModelWrapper**: 统一模型接口 | Unified model interface.
  - `MLModelWrapper`: 传统 ML 模型包装器 | Wrapper for standard ML models.
  - `DLModelWrapper`: 深度学习模型包装器（自动 2D→3D 转换、早停、权重保存）| Wrapper for DL models (handles 2D to 3D reshape, early stopping, checkpointing).
  - `ModelFactory`: 模型工厂 | Factory pattern for model instantiation.

### 验证与评估 (src/evaluation/) | Validation & Evaluation
- **MetricsCalculator**: 分类指标（Accuracy, F1, AUC）+ 回测指标 | Classification metrics & backtesting stats.
- **ModelExplainer**: XGBoost Gain 前20特征 + SHAP Summary Plot | Top 20 Gain features & SHAP plots.
- **AblationRunner**: 自动化消融实验调度器 | Automated ablation study runner.

### 交易回测 (src/backtesting/) | Trading Backtest
- **SignalGenerator**: 简单阈值法信号生成 | Simple threshold-based signal generation.
- **TradingStrategy**: long-only, short-only, long-short 策略 | Supported execution strategies.
- **Backtester**: 回测引擎（年化收益、夏普比率、最大回撤）| Backtest engine calculating Ann. Return, Sharpe Ratio, Max Drawdown.

---

## 输出结果 | Output Results

| 目录 (Directory) | 内容 (Contents) |
|------|------|
| `results/` | 消融实验结果 CSV | Ablation results in CSV |
| `figures/` | 权益曲线、特征重要性、SHAP解释图 | Equity curves, Feature Importance, SHAP plots |
| `logs/` | 运行日志 | Execution logs |
| `checkpoints/` | 模型检查点 | Model state dictionaries / weights |

---

## 技术亮点 | Technical Highlights

1. **防数据穿越 (Prevent Data Leakage)**：特征标准化严格在数据切分后进行 | Feature scaling is strictly applied *after* train/test splits.
2. **统一模型接口 (Unified Model Interface)**：ModelWrapper 抽象了 ML 和 DL 模型的差异 | Wrappers abstract the API differences between scikit-learn and PyTorch.
3. **容错机制 (Fault Tolerance)**：单个模型失败不中断整批消融实验 | The failure of a single model won't crash the entire ablation grid.
4. **分层配置 (Hierarchical Configs)**：支持多配置文件合并和命令行覆盖 | Supports merging multiple YAML files and overriding via CLI.

---

## 系统要求 | System Requirements

- Python 3.8+
- PyTorch（深度学习模型 | For Deep Learning models）
- CUDA（可选，GPU 加速 | Optional, for GPU acceleration）

---

## 许可证 | License

MIT License
