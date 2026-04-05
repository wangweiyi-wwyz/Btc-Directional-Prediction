"""
日志配置模块
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "btc_prediction",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    参数：
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        console: 是否输出到控制台
        
    返回：
        配置好的日志记录器
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(
        log_path / f"{name}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
