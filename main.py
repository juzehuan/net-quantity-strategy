import akshare as ak
import re
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import concurrent.futures
import logging
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

from datetime import datetime
import pandas as pd
# 配置日志系统
class ExitOnErrorHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            sys.exit(1)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log', encoding='utf-8', mode='w'),
        stream_handler,
        ExitOnErrorHandler()
    ]
)
logger = logging.getLogger(__name__)
# 获取所有A股代码和名称
def get_all_stock_codes() -> List[Tuple[str, str]]:
    """获取A股股票代码和名称列表"""
    try:
        # 使用AKShare获取A股实时行情数据
        df = ak.stock_zh_a_spot_em()
        logger.info(f'获取A股数据成功，原始数据量: {len(df)}')

        # 筛选上海和深圳证券交易所A股
        # 代码格式: 6开头为沪市，0或3开头为深市
        df = df[df['代码'].str.match(r'^(6|0|3)')]
        logger.info(f'筛选后A股数量: {len(df)}')

        # 转换代码格式为sh.600000或sz.000000
        def format_code(code):
            if code.startswith('6'):
                return f'sh.{code}'
            else:
                return f'sz.{code}'

        # 提取代码和名称并转换为元组列表
        stock_info = [(format_code(row['代码']), row['名称']) for _, row in df.iterrows()]
        logger.info(f'成功获取{len(stock_info)}只A股股票')
        return stock_info
    except Exception as e:
        logger.error(f'获取A股股票列表异常: {str(e)}', exc_info=True)
        return []


# 从数据获取模块导入函数
from data_fetcher import get_stock_data


def apply_strategy(df):
    """应用交易策略并计算收益率和持有天数

    参数:
        df: 包含技术指标的股票数据DataFrame

    返回:
        pd.DataFrame: 添加了交易信号、收益率和持有天数的DataFrame
    """
    df = df.copy()

    # 初始化策略列
    df['operation'] = '观望'
    df['position'] = 0  # 0: 空仓, 1: 持仓
    df['entry_price'] = np.nan
    df['holding_return'] = 0.0
    df['total_return'] = 0.0
    df['holding_days'] = 0  # 新增持有天数列

    position = 0  # 当前持仓状态
    entry_price = 0.0
    entry_date = None  # 新增：记录买入日期
    total_profit = 0.0
    trade_count = 0

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 买入信号: MACD < 0 且 大单净量 > 0.2
        if position == 0 and current['macd'] < 0 and current['big_order_net'] > 0.2:
            position = 1
            entry_price = current['close']
            entry_date = current.name  # 假设索引是日期
            df.at[df.index[i], 'operation'] = '买入'
            df.at[df.index[i], 'position'] = 1
            df.at[df.index[i], 'entry_price'] = entry_price
            df.at[df.index[i], 'holding_days'] = 1  # 第一天持有

        # 卖出信号: MACD > 0 且 大单净量 < -1
        elif position == 1 and current['macd'] > 0 and current['big_order_net'] < -1:
            position = 0
            exit_price = current['close']
            profit = (exit_price - entry_price) / entry_price * 100
            total_profit += profit
            trade_count += 1

            df.at[df.index[i], 'operation'] = '卖出'
            df.at[df.index[i], 'position'] = 0
            df.at[df.index[i], 'holding_return'] = profit
            # 计算持有天数
            holding_days = (current.name - entry_date).days
            df.at[df.index[i], 'holding_days'] = holding_days

            # 计算累计收益率
            if trade_count > 0:
                df.at[df.index[i], 'total_return'] = total_profit

        # 持有状态
        elif position == 1:
            df.at[df.index[i], 'operation'] = '持有'
            df.at[df.index[i], 'position'] = 1
            df.at[df.index[i], 'entry_price'] = entry_price
            current_return = (current['close'] - entry_price) / entry_price * 100
            df.at[df.index[i], 'holding_return'] = current_return
            df.at[df.index[i], 'total_return'] = total_profit + current_return
            # 计算持有天数
            holding_days = (current.name - entry_date).days
            df.at[df.index[i], 'holding_days'] = holding_days

    # 填充最后持仓的总收益率和持有天数
    if position == 1:
        last_idx = df.index[-1]
        current_return = (df.loc[last_idx, 'close'] - entry_price) / entry_price * 100
        df.at[last_idx, 'total_return'] = total_profit + current_return
        # 计算持有天数
        holding_days = (df.loc[last_idx].name - entry_date).days
        df.at[last_idx, 'holding_days'] = holding_days

    return df


# 主函数
main_executed = False
# 策略参数配置 - 集中管理可调整参数
STRATEGY_PARAMS = {
    'analysis_days': 600,      # 分析周期
    'session_refresh_interval': 100  # 会话刷新间隔
}



# 处理单只股票数据并应用策略
def process_stock(code, info):
    thread_start = time.time()
    try:
        # 应用交易策略
        df = apply_strategy(info['data'])
        if df.empty:
            logger.warning(f'{code}策略应用后数据为空，跳过处理')
            return

        # 获取最新交易信号
        latest_data = df.iloc[-1]

        # 提取所需结果字段
        result = {
            '股票代码': code,
            '股票名称': info['name'],
            '当日股价': round(latest_data['close'], 2),
            '当日涨跌幅': round(latest_data['daily_return'], 2),
            '操作策略': latest_data['operation'],
            '持有天数': latest_data['holding_days'],
            '持有期间总收益率': round(latest_data['holding_return'], 2),
            '回测期间总收益率': round(latest_data['total_return'], 2),
            '处理耗时(秒)': round(time.time() - thread_start, 4)
        }
        return result
        logger.info(f'{code}策略应用完成，耗时{round(time.time() - thread_start, 4)}秒')
    except Exception as e:
        logger.error(f'{code}策略应用失败: {str(e)}', exc_info=True)
        sys.exit(1)

def main():
    global main_executed
    if main_executed:
        return
    main_executed = True
    logger.info('主函数开始执行')
    logger.info("交易策略: MACD+大单净量策略")
    start_time = time.time()

    try:
        # 获取日期范围
        today = datetime.now()
        delta = 1
        while True:
            prev_date = today - timedelta(days=delta)
            if prev_date.weekday() < 5:  # 0-4代表周一至周五
                end_date = prev_date.strftime('%Y-%m-%d')
                break
            delta += 1

        # 获取股票列表
        # 获取全部成分股并过滤科创板(688/689开头)和创业板(300开头)股票
        # 假设get_all_stock_codes()返回元组列表，取第一个元素作为股票代码字符串
        stock_info = [code for code in get_all_stock_codes() if not code[0].startswith(('sh688', 'sh689', 'sz300'))]
        logger.info(f'获取到{len(stock_info)}只股票')

        if not stock_info:
            logger.error('错误：未获取到任何股票代码，请检查akshare连接或筛选条件')
            return

        # 初始化已处理代码集合和数据存储字典
        processed_codes = set()
        all_stock_data = {}

        # 第一阶段：批量获取所有股票数据
        logger.info('开始批量获取股票数据...')
        for item in tqdm(stock_info, desc='获取股票数据', unit='只'):
            # 确保item是元组且至少包含代码
            if not isinstance(item, tuple) or len(item) < 1:
                logger.warning(f'无效的股票信息格式: {item}，跳过处理')
                continue
            code = item[0]
            # 验证股票代码格式 (必须为sh.或sz.开头，共9位)
            if not re.match(r'^(sh|sz)\.\d{6}$', code):
                logger.warning(f'股票代码格式无效: {code}，跳过处理')
                continue
            try:
                # 获取股票名称（如果可用）
                name = item[1] if len(item) > 1 else '未知名称'
                logger.debug(f'开始获取股票数据: {code} - {name}')
                # 获取股票数据
                try:
                      stock_data = get_stock_data(code, name)
                      if stock_data is not None and not stock_data.empty:
                          all_stock_data[code] = {'data': stock_data, 'name': name}
                          processed_codes.add(code)
                          logger.debug(f'{code}数据获取成功，共{len(stock_data)}条记录')
                      else:
                          logger.warning(f'{code}未获取到有效数据')
                except Exception as e:
                    logger.error(f"获取{code}数据时发生错误: {str(e)}", exc_info=True)
                    sys.exit(1)
            except Exception as e:
                logger.error(f'处理{code}时发生错误: {str(e)}', exc_info=True)
                sys.exit(1)
            # 控制请求频率
            # time.sleep(0.5)

        # 应用策略并汇总结果
        logger.info(f'共获取到{len(all_stock_data)}只股票数据，开始应用交易策略...')
        results = []


        # 使用线程池处理股票数据
        max_workers = os.cpu_count() or 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_stock, code, info) for code, info in all_stock_data.items()]
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='应用策略', unit='只'):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f'处理任务时发生异常: {str(e)}')

        # 保存汇总结果
        if results:
            results_df = pd.DataFrame(results)
            # 添加最新日期和回测时间区间列
            results_df['最新日期'] = end_date
            results_df['回测时间区间'] = f'{start_date}至{end_date}'
            results_df = results_df[['股票代码', '股票名称', '最新日期', '回测时间区间', '当日股价', '当日涨跌幅', '操作策略', '持有天数', '持有期间总收益率', '回测期间总收益率']]

            results_path = 'result/strategy_results.csv'
            # 确保目录存在
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            logger.info(f'策略结果已保存至{results_path}，共{len(results)}条记录')
        else:
            logger.warning('未生成任何策略结果')

    except Exception as e:
        logger.error(f'主程序执行异常: {str(e)}', exc_info=True)
        sys.exit(1)
    total_time = time.time() - start_time
    logger.info(f'程序总执行时间: {total_time:.2f}秒')
    logger.info('主函数执行完毕')
    print(f'所有股票处理完毕')
    print(f'去重后处理股票数量: {len(processed_codes)}')


if __name__ == '__main__':
    result_file = 'result/strategy_results.csv'
    if os.path.exists(result_file):
        os.remove(result_file)
    main()










