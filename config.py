"""
策略配置参数模块
集中管理所有可调整的策略参数，便于维护和优化
"""
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class StrategyConfig:
    """策略核心配置参数"""
    # 数据采集参数
    data_range_days: int = 1000  # 历史数据采集天数（扩展至1000天以提高模型稳定性）
    max_retries: int = 3        # API调用最大重试次数
    retry_wait_time: int = 5    # API重试等待时间(秒)
    cache_expire_days: int = 1  # 缓存过期天数
    
    # 技术指标参数
    macd_params: Dict[str, int] = None
    order_params: Dict = None
    
    # 交易信号参数
    buy_threshold: float = 0.2   # 买入信号阈值
    sell_threshold: float = -1.0 # 卖出信号阈值
    
    # 默认流通盘假设(股)，实际应用中应从数据源获取
    default_share_capital: float = 5e8  # 5亿股
    # 数据容错参数
    min_valid_data_ratio: float = 0.8  # 最小有效数据比例
    max_consecutive_failures: int = 5  # 最大连续失败次数

    def __post_init__(self):
        # 参数范围验证
        if self.data_range_days < 365:
            raise ValueError("数据采集天数不能少于365天")
        if self.max_retries < 1:
            raise ValueError("最大重试次数必须大于0")
        if self.retry_wait_time < 1:
            raise ValueError("重试等待时间不能少于1秒")
        if self.min_valid_data_ratio <= 0 or self.min_valid_data_ratio > 1:
            raise ValueError("最小有效数据比例必须在(0, 1]范围内")
        
        # MACD默认参数
        if self.macd_params is None:
            self.macd_params = {
                'short_window': 12,
                'long_window': 26,
                'signal_window': 9
            }
        
        # MACD参数验证
        required_macd_keys = ['short_window', 'long_window', 'signal_window']
        if not all(key in self.macd_params for key in required_macd_keys):
            raise ValueError(f"MACD参数缺少必要键: {required_macd_keys}")
        if self.macd_params['short_window'] >= self.macd_params['long_window']:
            raise ValueError("MACD短期窗口必须小于长期窗口")
        if self.macd_params['signal_window'] >= self.macd_params['short_window']:
            raise ValueError("MACD信号窗口必须小于短期窗口")
        
        # 大单参数默认值
        if self.order_params is None:
            self.order_params = {
                'lot_size': 100,  # 1手=100股
                'large_order_thresholds': {
                    'small_cap': 500,   # 流通盘≤1亿股: 500手
                    'mid_cap': 1000,    # 1亿<流通盘≤10亿股: 1000手
                    'large_cap': 3000   # 流通盘>10亿股: 3000手
                }
            }

# 股票代码配置
STOCK_CODES = [
    'sh.600036',  # 招商银行
    'sh.601318',  # 中国平安
    'sh.600000'   # 浦发银行
]

# 日期范围配置
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'

# 创建默认配置实例
DEFAULT_CONFIG = StrategyConfig()

# 导出配置字典供非dataclass环境使用
CONFIG_DICT = {
    'data_range_days': DEFAULT_CONFIG.data_range_days,
    'max_retries': DEFAULT_CONFIG.max_retries,
    'retry_wait_time': DEFAULT_CONFIG.retry_wait_time,
    'cache_expire_days': DEFAULT_CONFIG.cache_expire_days,
    'macd_params': DEFAULT_CONFIG.macd_params,
    'order_params': DEFAULT_CONFIG.order_params,
    'buy_threshold': DEFAULT_CONFIG.buy_threshold,
    'sell_threshold': DEFAULT_CONFIG.sell_threshold,
    'default_share_capital': DEFAULT_CONFIG.default_share_capital
}