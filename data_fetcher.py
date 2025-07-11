import akshare as ak
import pandas as pd
import re
import time
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from config import DEFAULT_CONFIG
from technical_indicators import  calculate_technical_indicators

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
    # 设置默认日期范围并转换为Timestamp
    if end_date is None:
        end_date = pd.Timestamp(datetime.today())
    else:
        end_date = pd.Timestamp(end_date)
    if start_date is None:
        start_date = end_date - timedelta(days=DEFAULT_CONFIG.data_range_days)
    else:
        start_date = pd.Timestamp(start_date)

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
    existing_df = None
    if cache_path.exists():
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - modified_time).days < DEFAULT_CONFIG.cache_expire_days:
            logger.debug(f"缓存文件有效，加载数据: {cache_path}")
            try:
                existing_df = pd.read_pickle(cache_path)
                # 获取缓存数据中的最新日期
                if not existing_df.empty:
                    latest_cache_date = existing_df.index.max()
                    logger.debug(f"缓存数据最新日期: {latest_cache_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.warning(f"缓存文件损坏，重新获取数据: {str(e)}")
        else:
            logger.debug(f"缓存文件已过期，重新获取数据: {cache_path}")
    else:
        logger.debug(f"缓存文件不存在，将获取新数据: {cache_path}")

    # 确定实际需要获取的日期范围
    if existing_df is not None and not existing_df.empty and isinstance(latest_cache_date, pd.Timestamp) and pd.notna(latest_cache_date) and end_date > latest_cache_date:
        # 增量更新：只获取最新缓存日期之后的数据
        start_date = latest_cache_date + timedelta(days=1)
        # 检查起始日期是否大于终止日期
        if start_date > end_date:
            logger.info(f"起始日期({start_date.strftime('%Y-%m-%d')})大于终止日期({end_date.strftime('%Y-%m-%d')})，无需增量更新")
            return existing_df
        logger.info(f"检测到现有缓存，执行增量更新: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    elif existing_df is not None and not existing_df.empty:
        # 缓存数据已是最新，直接返回
        return existing_df

    # 计算所需数据量
    min_required_rows = max(DEFAULT_CONFIG.macd_params['short_window'], DEFAULT_CONFIG.macd_params['long_window']) * 2
    buffer_factor = 2.0  # 增加100%缓冲以应对非交易日和数据缺失

    required_days = (end_date - start_date).days
    logger.info(f'基于所需最少{min_required_rows}行数据，设置日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} (共{required_days}天)')




    # 获取数据并处理重试
    max_retries = DEFAULT_CONFIG.max_retries
    retry_count = 0
    df = pd.DataFrame()
    ak_code = code.split('.')[-1]  # 转换代码格式，移除点号及前面的字符串



    while retry_count < max_retries:
        try:
            # 使用AKShare获取日线数据，前复权
            # 统一使用stock_zh_a_hist接口获取历史数据
            try:
                time.sleep(2)
                df = ak.stock_zh_a_hist(
                    symbol=ak_code,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
                # 标准化列名并转换数据类型
                column_mapping = {
                    '日期': 'date',
                    '股票代码': 'code',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',  # 单位: 手
                    '成交额': 'amount',  # 单位: 元
                    '振幅': 'amplitude',  # 单位: %
                    '涨跌幅': 'pct_change',  # 单位: %
                    '涨跌额': 'price_change',  # 单位: 元
                    '换手率': 'turnover_rate'  # 单位: %
                }
                df.rename(columns=column_mapping, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
            except requests.exceptions.ProxyError as e:
                logger.error(f"股票{ak_code}代理连接错误: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    sleep_time = min(2 ** retry_count, 30)  # 限制最大等待时间为30秒
                    logger.info(f"代理错误，{sleep_time}秒后重试({retry_count}/{max_retries})")
                    time.sleep(sleep_time)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"股票{ak_code}网络请求异常: {str(e)}")
                continue
            # 验证返回数据格式
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"接口返回非DataFrame类型: {type(df)}")
            if not df.empty:
                # 过滤日期范围
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if not df.empty:
                    break
            logger.warning(f'{code}未返回数据，重试中...')

        except Exception as e:
            logger.warning(f'{code}查询失败(尝试{retry_count+1}/{max_retries}): {str(e)}')
            time.sleep(DEFAULT_CONFIG.retry_wait_time)
            retry_count += 1

    if df.empty:
        logger.error(f'{code}数据获取失败，所有重试均未成功')
        return None

    # 标准化列名并转换数据类型
    column_mapping = {
        '日期': 'date',
        '股票代码': 'code',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',  # 单位: 手
        '成交额': 'amount',  # 单位: 元
        '振幅': 'amplitude',  # 单位: %
        '涨跌幅': 'pct_change',  # 单位: %
        '涨跌额': 'price_change',  # 单位: 元
        '换手率': 'turnover_rate'  # 单位: %
    }
    df = df.rename(columns=column_mapping)
    df['date'] = pd.to_datetime(df['date'])

    # 确保必要的列存在
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f'{code}数据缺少必要列: {required_columns}')
        return None

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
        # 使用东方财富个股信息接口获取流通股数据（更稳定可靠）
        stock_info_df = ak.stock_individual_info_em(symbol=ak_code, timeout=10)

        # 查找流通股数据（单位：股）
        if not stock_info_df.empty:
            # 筛选"流通股"行并提取数值
            float_share_data = stock_info_df[stock_info_df['item'] == '流通股']['value'].values
            if len(float_share_data) > 0:
                share_capital = float(float_share_data[0])
                logger.info(f'成功获取{code}流通股数据: {share_capital}股')
            else:
                share_capital = DEFAULT_CONFIG.default_share_capital
                logger.warning(f'{code}未找到流通股数据，使用默认值: {share_capital}股')
        else:
            share_capital = DEFAULT_CONFIG.default_share_capital
            logger.warning(f'{code}返回数据为空，使用默认流通股值: {share_capital}股')
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


