"""
模型解释模块

功能：
1. 输出 XGBoost Gain 前20特征
2. 绘制 SHAP Summary Plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


class ModelExplainer:
    """
    模型解释器
    
    负责：
    1. 特征重要性分析
    2. SHAP 值可视化
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        初始化模型解释器
        
        参数：
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_xgboost_feature_importance(
        self,
        model,
        feature_names: List[str],
        importance_type: str = 'gain',
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        获取 XGBoost 特征重要性
        
        参数：
            model: 训练好的 XGBoost 模型
            feature_names: 特征名称列表
            importance_type: 重要性类型 ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
            top_k: 显示前 k 个特征
            
        返回：
            特征重要性 DataFrame
        """
        # 获取特征重要性
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
        
        # 创建 DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        # 处理缺失的特征（重要性为0）
        all_features = set(feature_names)
        existing_features = set(importance_df['feature'])
        missing_features = all_features - existing_features
        
        if missing_features:
            missing_df = pd.DataFrame({
                'feature': list(missing_features),
                'importance': [0.0] * len(missing_features)
            })
            importance_df = pd.concat([importance_df, missing_df], ignore_index=True)
        
        # 排序并选择前 k 个
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_k)
        
        return importance_df
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str = "model",
        save: bool = True
    ) -> None:
        """
        绘制特征重要性图
        
        参数：
            importance_df: 特征重要性 DataFrame
            model_name: 模型名称
            save: 是否保存图表
        """
        plt.figure(figsize=(10, 8))
        
        # 按重要性排序
        plot_data = importance_df.sort_values('importance', ascending=True)
        
        plt.barh(plot_data['feature'], plot_data['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{model_name} - Top {len(plot_data)} Feature Importance')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_feature_importance.png', dpi=300)
        
        plt.show()
        plt.close()
    

    def plot_shap_summary(
        self,
        model,
        X: pd.DataFrame,
        model_name: str = "model",
        max_display: int = 20,
        save: bool = True
    ) -> None:
        """绘制 SHAP Summary Plot"""
        try:
            import shap
            
            # --- 增强对不同模型(如 LightGBM)的兼容性 ---
            if model_name.lower() == 'lightgbm' and hasattr(model, 'booster_'):
                # 针对 sklearn wrapper 的 LightGBM，提取底层 booster
                explainer = shap.TreeExplainer(model.booster_)
            elif hasattr(model, 'get_booster'):
                # 针对 XGBoost
                explainer = shap.TreeExplainer(model.get_booster())
            else:
                # 默认 TreeExplainer 通常比泛用的 Explainer 更稳健，适用于 RF 等
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer(X)
            # -----------------------------------------------------

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            
            if save:
                filepath = self.output_dir / f'{model_name}_shap_summary.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"[{model_name}] SHAP Summary 图已保存至: {filepath}")
            
            plt.show()
            plt.close() 
            
        except ImportError:
            print("SHAP 库未安装，无法绘制 SHAP 图表")
        except Exception as e:
            print(f"[{model_name}] 绘制 SHAP 图表时出错: {e}")
    
    def plot_shap_waterfall(
        self,
        model,
        X: pd.DataFrame,
        instance_idx: int = 0,
        model_name: str = "model",
        save: bool = True
    ) -> None:
        """
        绘制单个样本的 SHAP Waterfall Plot
        
        参数：
            model: 训练好的模型
            X: 特征 DataFrame
            instance_idx: 样本索引
            model_name: 模型名称
            save: 是否保存图表
        """
        try:
            import shap
            
            # 创建 SHAP 解释器
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            # 绘制 Waterfall Plot
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap_values[instance_idx], show=False)
            plt.title(f'{model_name} - SHAP Waterfall Plot (Instance {instance_idx})')
            plt.tight_layout()
            
            if save:
                plt.savefig(self.output_dir / f'{model_name}_shap_waterfall_{instance_idx}.png', 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("SHAP 库未安装，无法绘制 SHAP 图表")
        except Exception as e:
            print(f"绘制 SHAP 图表时出错: {e}")
        plt.close()
    
    def explain_model(
        self,
        model,
        X: pd.DataFrame,
        feature_names: List[str],
        model_name: str = "model",
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        完整的模型解释流程
        
        参数：
            model: 训练好的模型
            X: 特征 DataFrame
            feature_names: 特征名称列表
            model_name: 模型名称
            top_k: 显示前 k 个特征
            
        返回：
            特征重要性 DataFrame
        """
        # 获取特征重要性
        if hasattr(model, 'get_booster'):
            # XGBoost 模型
            importance_df = self.get_xgboost_feature_importance(
                model, feature_names, importance_type='gain', top_k=top_k
            )
        elif hasattr(model, 'feature_importances_'):
            # 其他树模型
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_k)
        else:
            print(f"模型 {model_name} 不支持特征重要性分析")
            return pd.DataFrame()
        
        # 打印特征重要性
        print(f"\n{model_name} - Top {len(importance_df)} Feature Importance:")
        print(importance_df.to_string(index=False))
        
        # 绘制特征重要性图
        self.plot_feature_importance(importance_df, model_name, save=True)
        
        # 绘制 SHAP 图
        self.plot_shap_summary(model, X, model_name, max_display=top_k, save=True)
        plt.close()
        return importance_df

