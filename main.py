import baostock as bs
import re
import pandas as pd
from datetime import datetime, timedelta
import time
import baostock as bs
import random
import logging
from typing import Dict, List, Optional, Tuple
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from typing import List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
from technical_indicators import calculate_macd, calculate_big_order_net
from config import DEFAULT_CONFIG
import deep_learning_model as dl_model

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 初始化Baostock
def init_baostock() -> bool:
    """初始化Baostock API连接

    尝试登录Baostock服务，如果登录成功返回True，否则返回False

    返回:
        bool: 登录成功返回True，失败返回False
    """
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

# 获取所有A股代码和名称
def get_all_stock_codes() -> List[Tuple[str, str]]:
    """获取上证指数(000001.SH)的成分股代码和名称列表

    返回:
        List[Tuple[str, str]]: 成分股代码和名称的元组列表
    """
    try:

        # 使用固定历史日期测试数据获取（2023年1月3日，周二）
        trade_date_str = '2025-07-08'
        # 恢复date关键字参数，使用API要求的YYYY-MM-DD格式
        rs = bs.query_all_stock(trade_date_str)  # 获取全市场股票列表，传递日期参数
        logger.info(f'Baostock API调用结果 - 错误代码: {rs.error_code}, 错误信息: {rs.error_msg}')
        if rs.error_code != '0':
            logger.error(f'获取指数成分股失败: {rs.error_code} - {rs.error_msg}')
            return []
        # 直接使用API提供的get_data()方法获取DataFrame
        df = rs.get_data()
        data_list = df.values.tolist()
        logger.info(f'原始股票数据量: {len(data_list)}')
        # 筛选上海和深圳证券交易所A股（代码以sh.6、sz.0或sz.3开头）
        # 调整筛选逻辑：支持sh.6、sz.0、sz.3开头或纯数字6、0、3开头的代码格式
        df = df[df['code'].str.match(r'^(sh\.)?6|(sz\.)?(0|3)')]
        logger.info(f'筛选后A股数量: {len(df)}')
        # 提取代码和名称并转换为元组列表
        stock_info = [(row['code'], row['code_name']) for _, row in df.iterrows()]
        logger.info(f'成功获取{len(stock_info)}只A股股票')
        return stock_info
    except Exception as e:
        logger.error(f'获取指数成分股异常: {str(e)}', exc_info=True)
        return []


# 从数据获取模块导入函数
from data_fetcher import get_stock_data




# 主函数
main_executed = False
# 策略参数配置 - 集中管理可调整参数
STRATEGY_PARAMS = {
    'macd': {
        'a_share_daily': {'fast': 6, 'slow': 19, 'signal': 6},
        'standard': {'fast': 12, 'slow': 26, 'signal': 9},
        'bull_market': {'fast': 5, 'slow': 15, 'signal': 3},
        'bear_market': {'fast': 8, 'slow': 22, 'signal': 6}
    },
    'data_range_days': 600,  # 数据获取范围
    'analysis_days': 600,      # 分析周期
    'session_refresh_interval': 100  # 会话刷新间隔
}





def main():
    global main_executed
    if main_executed:
        return
    main_executed = True
    logger.info('主函数开始执行')
    logger.info("AI策略: 使用逻辑回归模型")

    # 初始化Baostock
    if not init_baostock():
        logger.error('Baostock初始化失败')
        return

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

        # 使用配置的参数控制数据获取范围
        start_date = (prev_date - timedelta(days=STRATEGY_PARAMS['data_range_days'])).strftime('%Y-%m-%d')
        logger.info(f'使用日期范围: {start_date} 至 {end_date}')

        # 获取股票列表
        stock_info = get_all_stock_codes()  # 获取全部成分股
        logger.info(f'获取到{len(stock_info)}只股票')

        if not stock_info:
            logger.error('错误：未获取到任何股票代码，请检查Baostock连接或筛选条件')
            return

        # 初始化已处理代码集合和数据存储字典
        processed_codes = set()
        all_stock_data = {}

        # 第一阶段：批量获取所有股票数据
        logger.info('开始批量获取股票数据...')
        for item in stock_info:
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
                logger.info(f'开始获取股票数据: {code} - {name}')
                # 获取股票数据
                try:
                      stock_data = get_stock_data(code, name)
                      if stock_data is not None and not stock_data.empty:
                          all_stock_data[code] = {'data': stock_data, 'name': name}
                          processed_codes.add(code)
                          logger.info(f'{code}数据获取成功，共{len(stock_data)}条记录')
                      else:
                          logger.warning(f'{code}未获取到有效数据')
                except Exception as e:
                      logger.error(f"获取{code}数据时发生错误: {str(e)}", exc_info=True)
                      logger.info(f"继续处理下一个股票代码...")
                      continue
            except Exception as e:
                logger.error(f'处理{code}时发生错误: {str(e)}', exc_info=True)
            # 控制请求频率
            time.sleep(0.5)

        # 第二阶段：批量进行深度学习预测
        logger.info(f'开始批量处理{len(all_stock_data)}只股票的深度学习预测...')
        if all_stock_data:
            # 加载深度学习模型
            # 根据数据特征维度设置input_shape=(时间步长, 特征数量)
            model = dl_model.StockLSTMAttentionModel(input_shape=(20, 10))
            if not model.load_model():
                logger.error('深度学习模型加载失败，无法进行预测')
            else:
                for code, info in all_stock_data.items():
                    try:
                        logger.info(f'对{code} - {info['name']}进行预测...')
                        # 生成交易信号
                        signals_df = model.generate_trading_signals(info['data'])
                        # 保存信号到CSV
                        output_path = f'{code}_{info['name']}_signals.csv'
                        signals_df.to_csv(output_path, encoding='utf-8-sig')
                        logger.info(f'{code}预测完成，结果已保存至{output_path}')
                    except Exception as e:
                        logger.error(f'处理{code}预测时发生错误: {str(e)}', exc_info=True)
                        continue

    except Exception as e:
        logger.error(f'主程序执行异常: {str(e)}', exc_info=True)
    finally:
        # 确保登出
        bs.logout()
        logger.info('Baostock会话已关闭')

    logger.info('主函数执行完毕')
    print(f'所有股票处理完毕')
    print(f'去重后处理股票数量: {len(processed_codes)}')


if __name__ == '__main__':
    main()