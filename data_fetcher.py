import baostock as bs
import pandas as pd
import re
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from config import DEFAULT_CONFIG
from technical_indicators import calculate_macd, calculate_big_order_net, calculate_technical_indicators

# 配置日志
logger = logging.getLogger(__name__)

# 获取股票历史数据
def get_stock_data(code: str, name: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, retry_count: int = 0) -> Optional[pd.DataFrame]:
    """获取股票历史K线数据

    使用Baostock API获取指定日期范围的日线数据，并进行数据清洗和格式转换

    参数:
        code (str): 股票代码，格式如'sh.600000'

    返回:
        Optional[pd.DataFrame]: 包含日期索引和技术指标的DataFrame，获取失败返回None
    """
    # 构建缓存路径
    CACHE_DIR = Path(__file__).resolve().parent / 'data'
    CACHE_DIR.mkdir(exist_ok=True)
    logger.debug(f"缓存目录已确认: {CACHE_DIR}")

    # 生成缓存文件名
    # 使用股票代码和名称组合作为缓存文件名，移除特殊字符
    safe_name = re.sub(r'[\\/:*?\"<>|]', '_', name)
    cache_filename = f"{code}_{safe_name}.pkl"
    cache_path = CACHE_DIR / cache_filename

    # 检查缓存是否存在
    if cache_path.exists():
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - modified_time).days < DEFAULT_CONFIG.cache_expire_days:
            logger.debug(f"缓存文件有效，加载数据: {cache_path}")
            try:
                return pd.read_pickle(cache_path)
            except Exception as e:
                logger.warning(f"缓存文件损坏，重新获取数据: {str(e)}")
        else:
            logger.debug(f"缓存文件已过期，重新获取数据: {cache_path}")
    else:
        logger.debug(f"缓存文件不存在，将获取新数据: {cache_path}")

    # 计算所需数据量
    min_required_rows = max(DEFAULT_CONFIG.macd_params['short_window'], DEFAULT_CONFIG.macd_params['long_window']) * 2
    buffer_factor = 2.0  # 增加100%缓冲以应对非交易日和数据缺失

    # 计算日期范围
    if end_date is None:
        end_date = datetime.today() - timedelta(days=1)
    if start_date is None:
        required_days = int(min_required_rows * buffer_factor)
        start_date = end_date - timedelta(days=DEFAULT_CONFIG.data_range_days)
    required_days = (end_date - start_date).days
    logger.info(f'基于所需{min_required_rows}行数据，设置日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} (共{required_days}天)')




    # 获取数据并处理重试
    rs = bs.query_history_k_data_plus(
        code, 'date,open,high,low,close,volume',
        start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
        frequency='d', adjustflag='3'
    )

    max_retries = DEFAULT_CONFIG.max_retries
    retry_count = 0
    while retry_count < max_retries:
        if rs.error_code != '0':
            logger.warning(f'{code}查询失败(尝试{retry_count+1}/{max_retries}): {rs.error_code}, {rs.error_msg}')

            # 处理会话错误
            if rs.error_code in ['10001', '10002']:
                logger.info('尝试重新登录Baostock...')
                bs.login()
                rs = bs.query_history_k_data_plus(
                    code, 'date,open,high,low,close,volume',
                    start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
                    frequency='d', adjustflag='3'
                )
            time.sleep(DEFAULT_CONFIG.retry_wait_time)
            retry_count += 1
        else:
            break

    if rs.error_code != '0':
        logger.error(f'{code}数据获取失败: {rs.error_code}, {rs.error_msg}')
        return None

    # 处理数据
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        logger.warning(f'{code}未返回任何数据')
        return None

    # 创建DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 转换数值类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 检查数据完整性
    valid_data_ratio = df[numeric_columns].notnull().mean().min()
    if valid_data_ratio < DEFAULT_CONFIG.min_valid_data_ratio:
        logger.warning(f'{code}数据完整性不足({valid_data_ratio:.2f}), 低于阈值{DEFAULT_CONFIG.min_valid_data_ratio}')
        return None

    # 填充缺失值
    df[numeric_columns] = df[numeric_columns].ffill().bfill()

    # 获取流通股数
    try:
        # 查询股票基本信息获取流通股数
        stock_basic_rs = bs.query_stock_basic(code=code)
        if stock_basic_rs.error_code == '0':
            stock_basic_df = stock_basic_rs.get_data()
            if not stock_basic_df.empty:
                share_capital = float(stock_basic_df['outstandingShares'].iloc[0])
                logger.info(f'成功获取{code}流通股数据: {share_capital}股')
            else:
                share_capital = DEFAULT_CONFIG.default_share_capital
                logger.warning(f'{code}未获取到流通股数据，使用默认值: {share_capital}股')
        else:
            share_capital = DEFAULT_CONFIG.default_share_capital
            logger.warning(f'{code}流通股数据查询失败({stock_basic_rs.error_code}): {stock_basic_rs.error_msg}，使用默认值')
    except Exception as e:
        share_capital = DEFAULT_CONFIG.default_share_capital
        logger.error(f'{code}获取流通股数据异常: {str(e)}，使用默认值')

    # 计算技术指标
    df = calculate_technical_indicators(df, share_capital=share_capital)

    # 保存缓存
    try:
        df.to_pickle(cache_path)
        logger.info(f'数据已缓存至: {cache_path}')
    except Exception as e:
        logger.warning(f'缓存保存失败: {str(e)}')

    return df

def load_data_from_pkl(file_path: str) -> Optional[pd.DataFrame]:
    """从pickle文件加载数据

    参数:
        file_path (str): pickle文件路径

    返回:
        Optional[pd.DataFrame]: 加载的数据，失败则返回None
    """
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        logger.error(f"从{file_path}加载数据失败: {str(e)}")
        return None


# 初始化Baostock连接
def init_baostock() -> bool:
    """初始化Baostock API连接"""
    try:
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f'登录失败: {lg.error_code} - {lg.error_msg}')
            return False
        logger.info('Baostock登录成功')
        return True
    except Exception as e:
        logger.error(f'登录异常: {str(e)}', exc_info=True)
        return False