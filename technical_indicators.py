"""
技术指标计算模块
实现同花顺风格的MACD和大单净量等技术指标
"""
import pandas as pd
from config import DEFAULT_CONFIG

# 从配置中获取指标参数
MACD_PARAMS = DEFAULT_CONFIG.macd_params
ORDER_PARAMS = DEFAULT_CONFIG.order_params

def calculate_ema(series, window):
    """
    计算指数移动平均线(EMA)
    使用pandas内置EWM实现，与同花顺算法保持一致
    
    参数:
        series: 价格序列
        window: 计算窗口
    
    返回:
        pd.Series: EMA计算结果
    """
    # 使用adjust=False确保与同花顺递归算法一致: EMA今日 = 昨日EMA × (n-1)/(n+1) + 今日价格 × 2/(n+1)
    return series.ewm(span=window, adjust=False).mean()

def calculate_macd(df, close_col='close'):
    """
    计算MACD指标
    包含EMA12、EMA26、DIF、DEA和MACD柱状图
    
    参数:
        df: 包含价格数据的DataFrame
        close_col: 收盘价列名
    
    返回:
        pd.DataFrame: 包含MACD指标的DataFrame
    """
    df = df.copy()
    
    # 计算EMA
    df['ema12'] = calculate_ema(df[close_col], MACD_PARAMS['short_window'])
    df['ema26'] = calculate_ema(df[close_col], MACD_PARAMS['long_window'])
    
    # 计算DIF
    df['dif'] = df['ema12'] - df['ema26']
    
    # 计算DEA
    df['dea'] = calculate_ema(df['dif'], MACD_PARAMS['signal_window'])
    
    # 计算MACD柱状图
    df['macd'] = 2 * (df['dif'] - df['dea'])
    
    return df

def calculate_big_order_net(df, volume_col='volume', close_col='close', open_col='open', share_capital=None):
    """
    计算同花顺风格大单净量指标
    
    参数:
        df: 包含成交量和价格数据的DataFrame
        volume_col: 成交量列名
        close_col: 收盘价列名
        open_col: 开盘价列名
        share_capital: 流通股数(股)
    
    返回:
        pd.DataFrame: 包含大单净量指标的DataFrame
    """
    if share_capital is None:
        raise ValueError("需要提供流通股数(share_capital)参数")
        
    df = df.copy()
    
    # 确定大单标准
    if share_capital <= 1e8:
        threshold = ORDER_PARAMS['large_order_thresholds']['small_cap']
    elif share_capital <= 1e9:
        threshold = ORDER_PARAMS['large_order_thresholds']['mid_cap']
    else:
        threshold = ORDER_PARAMS['large_order_thresholds']['large_cap']
    
    # 转换为股数
    large_order_threshold = threshold * ORDER_PARAMS['lot_size']
    
    # 模拟主动买卖判断
    df['price_change'] = df[close_col] - df[open_col]
    df['direction'] = df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # 识别大单
    df['is_large_order'] = df[volume_col] > large_order_threshold
    
    # 计算主动买卖大单量
    df['active_buy'] = df.apply(
        lambda row: row[volume_col] if row['direction'] > 0 and row['is_large_order'] else 0, axis=1)
    df['active_sell'] = df.apply(
        lambda row: row[volume_col] if row['direction'] < 0 and row['is_large_order'] else 0, axis=1)
    
    # 计算大单净量(百分比)
    df['big_order_net'] = (df['active_buy'] - df['active_sell']) / share_capital * 100
    
    # 清理临时列
    df = df.drop(['price_change', 'direction', 'is_large_order', 'active_buy', 'active_sell'], axis=1)
    
    return df

def calculate_kdj(df, low_col='low', high_col='high', close_col='close', n=9, m1=3, m2=3):
    """
    计算KDJ指标
    KDJ指标是一种动量指标，包含K线、D线和J线
    
    参数:
        df: 包含价格数据的DataFrame
        low_col: 最低价列名
        high_col: 最高价列名
        close_col: 收盘价列名
        n: 计算RSV的窗口大小
        m1: K线的平滑参数
        m2: D线的平滑参数
    
    返回:
        pd.DataFrame: 添加了KDJ指标的DataFrame
    """
    df = df.copy()
    
    # 计算最低和最高价
    df['lowest_low'] = df[low_col].rolling(window=n).min()
    df['highest_high'] = df[high_col].rolling(window=n).max()
    
    # 计算RSV (未成熟随机值)
    df['rsv'] = (df[close_col] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100
    
    # 计算K值和D值
    df['k'] = df['rsv'].ewm(alpha=1/m1, adjust=False).mean()
    df['d'] = df['k'].ewm(alpha=1/m2, adjust=False).mean()
    
    # 计算J值
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 清理临时列
    df = df.drop(['lowest_low', 'highest_high', 'rsv'], axis=1)
    
    return df

def calculate_technical_indicators(df):
    """
    计算常用技术指标
    :param df: 包含OHLCV数据的DataFrame
    :return: 添加了技术指标的DataFrame
    """
    # 计算MACD指标
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 计算RSI指标
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算布林带
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # 计算成交量指标
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume_ma5'] / df['volume_ma20']
    
    # 计算KDJ指标
    df = calculate_kdj(df)
    
    # 计算每日涨跌幅（百分比）
    df['daily_return'] = df['close'].pct_change() * 100
    df['daily_return_ma5'] = df['daily_return'].rolling(window=5).mean()  # 5日平均涨跌幅
    
    # 假设已有大单净量数据
    if 'big_order_net' not in df.columns:
        df['big_order_net'] = 0  # 实际应用中需要替换为真实数据源
    
    # 填充NaN值
    df = df.bfill()
    df = df.ffill()
    
    return df