"""
BTC 分类型目标预测 — 主程序入口

示例：
    # 默认流程
    python main.py --config config/default_config.yaml --mode train

    # 消融实验（6 模型 × 3 种划分 = 18 次）
    python main.py --config config/base_config.yaml --mode ablation
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataPreprocessor
from src.features import FeatureEngineering
from src.models import ModelTrainer, BaseModels
from src.evaluation import ModelEvaluator, ModelExplainer, AblationRunner
from src.backtesting import Backtester
from src.utils import setup_logger, ConfigLoader


def main():
    """主函数入口。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description='BTC 分类型目标预测')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'backtest', 'all', 'ablation'],
                       help='运行模式 (train/evaluate/backtest/all/ablation)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='要运行的模型 (ridge random_forest xgboost lightgbm lstm gru)')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                       help='数据划分策略 (fixed rolling regime)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备 (cpu/cuda)')
    args = parser.parse_args()
    
    # 加载配置
    config_loader = ConfigLoader()
    try:
        if args.mode == 'ablation':
            config_loader.load_all_configs(
                base_config='base_config.yaml',
                split_config='data_split.yaml'
            )
            model_names = args.models or ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru']
            model_params = config_loader.load_model_params(model_names)
            config_loader.config['models'] = model_params
        else:
            config_loader.load_yaml(args.config)
    except FileNotFoundError as e:
        print(f"配置文件加载失败: {e}")
        print("使用默认配置")
        config_loader.load_default_config()
    
    # 日志
    logger = setup_logger(
        name="btc_prediction",
        log_dir=config_loader.get('output.log_dir', 'logs')
    )
    
    logger.info("=" * 60)
    logger.info("BTC 预测系统启动")
    logger.info("=" * 60)
    
    # 1. 数据加载与预处理
    logger.info("步骤 1: 数据加载与预处理")
    preprocessor = DataPreprocessor(data_dir="data")
    
    # 从配置解析文件名（可含 data/ 前缀）
    from pathlib import Path
    ohlcv_file = Path(config_loader.get('data.ohlcv_file', 'data/btc_ohlc.csv')).name
    technical_file = Path(config_loader.get('data.technical_file', 'data/btc_technical_data_raw.csv')).name
    macro_file = Path(config_loader.get('data.macro_file', 'data/macro_data_raw.csv')).name
    
    ohlcv_df = preprocessor.load_ohlcv(ohlcv_file)
    logger.info(f"已加载 OHLCV: {len(ohlcv_df)} 行")
    
    technical_df = preprocessor.load_technical_indicators(technical_file)
    logger.info(f"已加载技术指标: {len(technical_df)} 行")
    
    macro_df = preprocessor.load_macro_data(macro_file)
    logger.info(f"已加载宏观数据: {len(macro_df)} 行")
    
    merged_df = preprocessor.merge_data()
    logger.info(f"合并后: {len(merged_df)} 行, {len(merged_df.columns)} 列")
    
    merged_df = preprocessor.create_target(merged_df)
    logger.info(
        f"目标分布: 涨={merged_df['target'].sum()}, "
        f"跌={len(merged_df) - merged_df['target'].sum()}"
    )
    
    # 2. 特征工程
    logger.info("步骤 2: 特征工程")
    feature_engineering = FeatureEngineering(
        scaler_window=config_loader.get('features.scaler_window', 252)
    )
    
    df_with_features = feature_engineering.fit_transform_features(
        merged_df,
        exclude_cols=['target', 'next_return']
    )
    logger.info(f"特征工程完成: {len(df_with_features.columns)} 列")
    
    # 3. 训练/验证/测试划分
    logger.info("步骤 3: 训练/验证/测试划分")
    train_df, val_df, test_df = preprocessor.get_train_val_test_split(
        df_with_features,
        train_ratio=config_loader.get('training.train_ratio', 0.6),
        val_ratio=config_loader.get('training.val_ratio', 0.2),
        test_ratio=config_loader.get('training.test_ratio', 0.2)
    )
    
    feature_cols = feature_engineering.get_feature_columns()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['target'].values
    
    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # === 用于存储所有训练好的模型的字典 ===
    trained_models = {}

    # 4. 模型训练
    if args.mode in ['train', 'all']:
        logger.info("步骤 4: 模型训练")
        
        trainer = ModelTrainer(
            checkpoint_dir=config_loader.get('output.checkpoint_dir', 'checkpoints')
        )
        
        # 动态获取要训练的模型列表
        # 如果命令行指定了 --models，优先使用；否则使用配置中的所有模型
        active_models = args.models or list(config_loader.get('models', {}).keys())
        
        for model_name in active_models:
            logger.info(f"正在训练 {model_name}...")
            if model_name in ['lstm', 'gru']:
                # 深度学习模型
                model = trainer.train_dl_model(
                    model_name,
                    X_train, y_train,
                    model_params=config_loader.get(f'models.{model_name}', {})
                )
            else:
                # 传统机器学习路线
                model = trainer.train_ml_model(
                    model_name,
                    X_train, y_train,
                    model_params=config_loader.get(f'models.{model_name}', {})
                )
            
            trained_models[model_name] = model
            
        logger.info("所有模型训练完成")

    
    # 5. 模型评估
    if args.mode in ['evaluate', 'all']:
        logger.info("步骤 5: 模型评估")
        
        evaluator = ModelEvaluator(
            output_dir=config_loader.get('output.figure_dir', 'figures')
        )
        explainer = ModelExplainer(
            output_dir=config_loader.get('output.figure_dir', 'figures')
        )
        
        # 遍历所有已训练的模型进行评估
        for model_name, model in trained_models.items():
            logger.info(f"评估 {model_name}...")
            results = evaluator.evaluate_model(
                model, model_name,
                X_train, y_train, X_val, y_val, X_test, y_test,
                feature_names=feature_cols
            )
            evaluator.print_evaluation_report(model_name, results)
            
            # SHAP 解释通常只适用于树模型
            if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                logger.info(f"生成 {model_name} 的 SHAP 解释图...")
                explainer.explain_model(
                    model, test_df[feature_cols], feature_cols, model_name=model_name
                )
    
    # 6. 回测
    if args.mode in ['backtest', 'all']:
        logger.info("步骤 6: 交易回测")
        
        backtester = Backtester(
            output_dir=config_loader.get('output.figure_dir', 'figures')
        )
        
        # 遍历所有已训练的模型进行回测
        for model_name, model in trained_models.items():
            logger.info(f"回测 {model_name} 产生的信号...")
            
            # 兼容不同 API 的预测概率获取
            try:
                y_test_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                # 针对 LSTM/GRU 等深度学习模型（假设你的输出已经是在 [0,1] 之间的概率）
                y_test_prob = model.predict(X_test).flatten()
            
            backtest_results = backtester.compare_strategies(
                y_test_prob,
                test_df['close'],
                strategies=config_loader.get('backtesting.strategies', ['buy_and_hold', 'long_only', 'short_only', 'long_short']),
                signal_threshold=config_loader.get('backtesting.signal_threshold', 0.5),
                initial_capital=config_loader.get('backtesting.initial_capital', 10000)
            )
            
            backtester.print_backtest_report(backtest_results) 
            # 传入 model_name_prefix
            backtester.plot_equity_curves(
                backtest_results, 
                model_name_prefix=model_name 
            )
            backtester.plot_strategy_comparison(
                backtest_results, 
                model_name_prefix=model_name
            )
    
    # logger.info("=" * 60)
    # logger.info("BTC 预测流程结束")
    # logger.info("=" * 60)


    # 7. 消融实验
    if args.mode == 'ablation':
        logger.info("步骤 7: 消融实验")
        
        feature_cols = feature_engineering.get_feature_columns()
        X = df_with_features[feature_cols].fillna(0).values
        y = df_with_features['target'].values
        dates = df_with_features.index
        
        # 牛熊市标签：高点回撤 40% 判熊，低点反弹 40% 判牛
        regime = None
        if args.splits is None or 'regime' in args.splits:
            close_prices = df_with_features['close'].values
            
            regime = np.zeros(len(close_prices), dtype=int)
            regime[0] = 1
            
            current_high = close_prices[0]
            current_low = close_prices[0]
            current_regime = 1  # 1=牛市, 0=熊市
            
            for i in range(1, len(close_prices)):
                price = close_prices[i]
                
                if current_regime == 1:
                    current_high = max(current_high, price)
                    if price <= current_high * 0.6:
                        current_regime = 0
                        current_low = price
                else:
                    current_low = min(current_low, price)
                    if price >= current_low * 1.4:
                        current_regime = 1
                        current_high = price
                
                regime[i] = current_regime
        
        ablation_runner = AblationRunner(
            config=config_loader.config,
            checkpoint_dir=config_loader.get('output.checkpoint_dir', 'checkpoints'),
            log_dir=config_loader.get('output.log_dir', 'logs'),
            results_dir=config_loader.get('output.results_dir', 'results'),
            device=args.device
        )
        
        results_df = ablation_runner.run_experiments(
            X=X,
            y=y,
            dates=dates,
            regime=regime,
            model_names=args.models,
            split_strategies=args.splits
        )
        
        ablation_runner.print_results_table()
        
        import json
        summary = ablation_runner.get_results_summary()
        logger.info("实验摘要详情:")
        formatted_summary = json.dumps(summary, indent=4, ensure_ascii=False)
        for line in formatted_summary.split('\n'):
            logger.info(line)
    
    logger.info("=" * 60)
    logger.info("BTC 预测流程结束")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
