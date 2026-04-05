"""
技术指标计算模块

功能：
1. 处理量价类异常值需使用 Hampel Filter
2. 震荡类指标需归一化至[-1,1]
3. 成交量需做 log(1+Volume) 变换
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.ndimage import median_filter


class TechnicalIndicators:
    """
    技术指标计算器
    
    支持四类技术指标：
    1. 趋势类：MA, EMA, MACD, ADX, DMI, Parabolic SAR
    2. 动量类：RSI, Stochastic, Williams %R, CCI, ROC
    3. 成交量类：OBV, A/D Line, Chaikin MF, VWAP
    4. 波动率类：ATR, Bollinger Bands, Keltner Channels
    """
    
    def __init__(self):
        self.indicators: Dict[str, pd.Series] = {}
        
    @staticmethod
    def hampel_filter(series: pd.Series, window_size: int = 5, n_sigmas: float = 3.0) -> pd.Series:
        """
        Hampel 滤波器：检测并处理异常值
        
        红线要求：窗口必须仅使用当前点及之前的数据，严禁包含未来数据
        
        参数：
            series: 输入序列
            window_size: 窗口大小
            n_sigmas: 标准差倍数阈值
            
        返回：
            处理后的序列
        """
        series_copy = series.copy()
        
        for i in range(len(series)):
            # 修复：窗口仅使用当前点及之前的数据（不包含未来数据）
            start = max(0, i - window_size + 1)
            end = i + 1  # 包含当前点
            window = series.iloc[start:end]
            
            median = window.median()
            std = window.std()
            
            if abs(series.iloc[i] - median) > n_sigmas * std:
                series_copy.iloc[i] = median
        
        return series_copy
    
    @staticmethod
    def normalize_rsi_to_range(rsi: pd.Series, min_val: float = -1, max_val: float = 1) -> pd.Series:
        """
        将 RSI 归一化到 [-1, 1] 范围
        
        红线要求：严禁使用全样本 min/max，应使用 RSI 的固定范围 [0, 100]
        
        参数：
            rsi: RSI 序列（范围 0-100）
            min_val: 目标最小值
            max_val: 目标最大值
            
        返回：
            归一化后的序列
        """
        # RSI 固定范围是 [0, 100]，直接映射到 [-1, 1]
        # 公式：(RSI - 50) / 50
        normalized = (rsi - 50) / 50
        
        # 裁剪到 [-1, 1] 范围（处理可能的异常值）
        normalized = normalized.clip(min_val, max_val)
        
        return normalized
    
    @staticmethod
    def normalize_williams_r_to_range(williams_r: pd.Series, min_val: float = -1, max_val: float = 1) -> pd.Series:
        """
        将 Williams %R 归一化到 [-1, 1] 范围
        
        红线要求：Williams %R 固定范围是 [-100, 0]，直接映射到 [-1, 1]
        
        参数：
            williams_r: Williams %R 序列（范围 -100 到 0）
            min_val: 目标最小值
            max_val: 目标最大值
            
        返回：
            归一化后的序列
        """
        # Williams %R 固定范围是 [-100, 0]，直接除以 100
        normalized = williams_r / 100
        
        # 裁剪到 [-1, 1] 范围
        normalized = normalized.clip(min_val, max_val)
        
        return normalized
    
    @staticmethod
    def log_volume(volume: pd.Series) -> pd.Series:
        """
        成交量对数变换：log(1 + Volume)
        
        参数：
            volume: 成交量序列
            
        返回：
            对数变换后的序列
        """
        return np.log1p(volume)
    
    # ==================== 趋势类指标 ====================
    
    def sma(self, close: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        return close.rolling(window=period, min_periods=1).mean()
    
    def ema(self, close: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        return close.ewm(span=period, adjust=False).mean()
    
    def macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD 指标
        
        返回：
            字典，键为 'macd'、'macd_signal'、'macd_histogram'
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均趋向指标"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    # ==================== 动量类指标 ====================
    
    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        随机指标
        
        返回：
            字典，键为 'stoch_k'、'stoch_d'
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d = k.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k,
            'stoch_d': d
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
        
        return wr
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """顺势指标"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        return cci
    
    def roc(self, close: pd.Series, period: int = 12) -> pd.Series:
        """变动率指标"""
        roc = 100 * (close - close.shift(period)) / (close.shift(period) + 1e-8)
        return roc
    
    # ==================== 成交量类指标 ====================
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """累积/派发线"""
        clv = ((close - low) - (high - close)) / (high - low + 1e-8)
        clv = clv * volume
        ad = clv.cumsum()
        
        return ad
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量加权平均价"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    # ==================== 波动率类指标 ====================
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        布林带
        
        返回：
            字典，键为 'bb_upper'、'bb_middle'、'bb_lower'、'bb_width'
        """
        middle = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        width = (upper - lower) / (middle + 1e-8)
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': width
        }
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int = 20, multiplier: float = 2) -> Dict[str, pd.Series]:
        """
        肯特纳通道
        
        返回：
            字典，键为 'kc_upper'、'kc_middle'、'kc_lower'
        """
        middle = close.rolling(window=period, min_periods=1).mean()
        atr = self.atr(high, low, close, period)
        
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        return {
            'kc_upper': upper,
            'kc_middle': middle,
            'kc_lower': lower
        }
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        参数：
            df: 包含 OHLCV 数据的 DataFrame
            
        返回：
            包含所有技术指标的 DataFrame
        """
        result = df.copy()
        
        # 应用 Hampel Filter 处理异常值
        result['close_filtered'] = self.hampel_filter(result['close'])
        result['volume_log'] = self.log_volume(result['volume'])
        
        # 趋势类指标
        for period in [5, 10, 20, 50, 200]:
            result[f'sma_{period}'] = self.sma(result['close'], period)
            result[f'ema_{period}'] = self.ema(result['close'], period)
        
        macd = self.macd(result['close'])
        result['macd'] = macd['macd']
        result['macd_signal'] = macd['macd_signal']
        result['macd_histogram'] = macd['macd_histogram']
        
        result['adx'] = self.adx(result['high'], result['low'], result['close'])
        
        # 动量类指标
        result['rsi'] = self.rsi(result['close'])
        # 修复：使用固定范围归一化，避免全样本 min/max 泄露
        result['rsi_normalized'] = self.normalize_rsi_to_range(result['rsi'])
        
        stoch = self.stochastic(result['high'], result['low'], result['close'])
        result['stoch_k'] = stoch['stoch_k']
        result['stoch_d'] = stoch['stoch_d']
        
        result['williams_r'] = self.williams_r(result['high'], result['low'], result['close'])
        # 修复：使用固定范围归一化，避免全样本 min/max 泄露
        result['williams_r_normalized'] = self.normalize_williams_r_to_range(result['williams_r'])
        
        result['cci'] = self.cci(result['high'], result['low'], result['close'])
        result['roc'] = self.roc(result['close'])
        
        # 成交量类指标
        result['obv'] = self.obv(result['close'], result['volume'])
        result['ad_line'] = self.ad_line(result['high'], result['low'], result['close'], result['volume'])
        result['vwap'] = self.vwap(result['high'], result['low'], result['close'], result['volume'])
        
        # 波动率类指标
        result['atr'] = self.atr(result['high'], result['low'], result['close'])
        
        bb = self.bollinger_bands(result['close'])
        result['bb_upper'] = bb['bb_upper']
        result['bb_middle'] = bb['bb_middle']
        result['bb_lower'] = bb['bb_lower']
        result['bb_width'] = bb['bb_width']
        result['bb_pct'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)
        
        kc = self.keltner_channels(result['high'], result['low'], result['close'])
        result['kc_upper'] = kc['kc_upper']
        result['kc_middle'] = kc['kc_middle']
        result['kc_lower'] = kc['kc_lower']
        
        return result
