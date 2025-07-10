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
    logger.info(f"缓存目录已确认: {CACHE_DIR}")

    # 生成缓存文件名
    # 使用股票代码和名称组合作为缓存文件名，移除特殊字符
    safe_name = re.sub(r'[\\/:*?\"<>|]', '_', name)
    cache_filename = f"{code}_{safe_name}.pkl"
    cache_path = CACHE_DIR / cache_filename

    # 检查缓存是否存在
    if cache_path.exists():
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - modified_time).days < DEFAULT_CONFIG.cache_expire_days:
            logger.info(f"缓存文件有效，加载数据: {cache_path}")
            try:
                return pd.read_pickle(cache_path)
            except Exception as e:
                logger.warning(f"缓存文件损坏，重新获取数据: {str(e)}")
        else:
            logger.info(f"缓存文件已过期，重新获取数据: {cache_path}")
    else:
        logger.info(f"缓存文件不存在，将获取新数据: {cache_path}")

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
                if not init_baostock():
                    logger.error('重新登录失败，无法继续查询')
                    break
                time.sleep(1)

            # 处理限流错误
            elif rs.error_code in ['10010', '10011']:
                wait_time = DEFAULT_CONFIG.retry_wait_time * (2 ** retry_count)  # 指数退避
                logger.info(f'请求过于频繁，等待{wait_time}秒后重试...')
                time.sleep(wait_time)

            retry_count += 1
            rs = bs.query_history_k_data_plus(
                code, 'date,open,high,low,close,volume',
                start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
                frequency='d', adjustflag='3'
            )
        else:
            break

    if rs.error_code != '0':
        logger.error(f'{code}多次尝试后仍失败: {rs.error_code}, {rs.error_msg}')
        return None

    # 处理数据
    df = pd.DataFrame(rs.get_data())
    logger.info(f'API原始返回数据行数: {len(df)}')
    if df.empty:
        logger.warning(f'API返回数据为空，代码: {code}')
        return None

    # 数据清洗和转换
    df = df.replace('', np.nan).infer_objects(copy=False)
    pre_clean_rows = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    logger.info(f'数据清洗后剩余行数: {len(df)} (删除了{pre_clean_rows - len(df)}行缺失数据)')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # 获取行业信息
    rs_industry = bs.query_stock_basic(code=code)
    industry_df = rs_industry.get_data()
    if not industry_df.empty and 'industry' in industry_df.columns:
        industry = industry_df['industry'].iloc[0]
    else:
        industry = '未知行业'
    df['industry'] = industry

    # 计算技术指标
    df['return'] = df['close'].pct_change()
    df['return_category'] = 1
    df.loc[df['return'] > 0.01, 'return_category'] = 2
    df.loc[df['return'] < -0.01, 'return_category'] = 0

    df = calculate_macd(df)
    pre_macd_rows = len(df)
    df = df.dropna(subset=['macd'])
    logger.info(f'MACD计算后剩余行数: {len(df)} (删除了{pre_macd_rows - len(df)}行无法计算MACD的数据)')

    share_capital = DEFAULT_CONFIG.default_share_capital
    df = calculate_big_order_net(df, share_capital=share_capital)
    pre_big_order_rows = len(df)
    df = df.dropna(subset=['big_order_net'])
    logger.info(f'大单净量计算后剩余行数: {len(df)} (删除了{pre_big_order_rows - len(df)}行无法计算大单净量的数据)')

    # 添加完整技术指标计算
    df = calculate_technical_indicators(df)
    
    # 使用IQR方法处理异常值
    numerical_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'bb_upper', 'bb_lower', 'volume_ratio', 'daily_return']
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 验证关键指标列是否存在
    required_columns = ['daily_return', 'daily_return_ma5']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f'技术指标计算失败，缺少必要列: {missing_columns}')
        # 如果存在缺失列，强制重新生成缓存
        if os.path.exists(cache_path):
            logger.warning(f'删除无效缓存文件: {cache_path}')
            os.remove(cache_path)
        return None

    # 数据验证
    min_required_rows = max(DEFAULT_CONFIG.macd_params['short_window'], DEFAULT_CONFIG.macd_params['long_window']) * 2
    # 最终数据验证
    if len(df) < min_required_rows:
        # 极端情况处理：如果数据严重不足，尝试扩大日期范围
        if len(df) == 0:
            logger.error(f'所有数据处理后为空，代码: {code}')
            return None

        # 检查是否已获取最早可用数据
        earliest_date = df.index.min()
        if earliest_date == start_date:
            logger.warning(f'已获取最早可用数据({earliest_date})，无法进一步扩大日期范围，代码: {code}')
            # 尽管数据不足，仍返回可用数据
            return df

        # 计算仍需的天数
        missing_days = int((min_required_rows - len(df)) * buffer_factor)
        logger.warning(f'数据量不足，仅{len(df)}行，需要至少{min_required_rows}行，尝试扩大日期范围{missing_days}天')

        # 检查是否达到最大重试次数
        if retry_count >= 3:
            logger.error(f'已尝试{retry_count}次扩大日期范围，仍无法获取足够数据，代码: {code}')
            return df  # 返回现有不足数据而非None

        # 扩大日期范围并重新获取
        new_start_date = end_date - timedelta(days=required_days + missing_days)
        logger.info(f'扩大后日期范围: {new_start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} (重试次数: {retry_count + 1})')
        return get_stock_data(code, start_date=new_start_date, end_date=end_date, retry_count=retry_count + 1)  # 递归重试

    df = df.dropna(subset=['return_category', 'macd', 'big_order_net'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 保存缓存
    if not df.empty:
        try:
            df.to_pickle(cache_path)
            logger.info(f"数据已保存到缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {str(e)}")
    else:
        logger.warning("数据为空，不保存到缓存")

    return df

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