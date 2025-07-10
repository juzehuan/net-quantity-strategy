import numpy as np
import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight
from config import DEFAULT_CONFIG
import tensorflow as tf
import pickle
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import load_model
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockLSTMAttentionModel:
    """
    LSTM+Attention混合模型用于股票收益率预测
    """
    def __init__(self, input_shape, num_classes=3, lstm_units=64, attention_units=32, dropout_rate=0.3):
        """
        初始化模型参数
        :param input_shape: 输入特征形状 (时间步长, 特征数量)
        :param num_classes: 预测类别数量
        :param lstm_units: LSTM层单元数
        :param attention_units: 注意力层单元数
        :param dropout_rate: Dropout比率
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_classif, k=10)  # 选择Top10特征

    def _build_model(self):
        """构建LSTM+Attention模型"""
        # 输入层
        inputs = Input(shape=self.input_shape)

        # LSTM层
        # 第一层双向LSTM
        lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True, recurrent_dropout=0.2))(inputs)
        lstm_out = LayerNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)

        # 第二层双向LSTM
        lstm_out = Bidirectional(LSTM(self.lstm_units//2, return_sequences=True, recurrent_dropout=0.2))(lstm_out)
        lstm_out = LayerNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)

        # 注意力层
        attention = Attention()([lstm_out, lstm_out])
        attention = LayerNormalization()(attention)

        # 第二个LSTM层
        lstm_out2 = LSTM(self.lstm_units//2, return_sequences=False)(attention)
        lstm_out2 = LayerNormalization()(lstm_out2)
        lstm_out2 = Dropout(self.dropout_rate)(lstm_out2)

        # 输出层
        outputs = Dense(self.num_classes, activation='softmax')(lstm_out2)

        # 构建模型
        model = Model(inputs=inputs, outputs=outputs)
        # 使用学习率调度器
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        logger.info(f"模型构建完成 - 输入形状: {self.input_shape}, 输出类别: {self.num_classes}")
        return model

    def prepare_data(
        self, df, feature_cols, target_col, time_steps=20
    ):
        """
        准备LSTM模型训练数据
        :param df: 包含特征和目标的DataFrame
        :param feature_cols: 特征列名列表
        :param target_col: 目标列名
        :param time_steps: 时间步长
        :return: 格式化后的X和y
        """
        # 提取特征和目标
        features = df[feature_cols].values
        target = df[target_col].values

        # 数据标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        # 特征选择
        features_selected = self.selector.fit_transform(features_scaled, y)

        # 构建时间序列样本
        X, y = [], []
        for i in range(time_steps, len(features_scaled)):
            X.append(features_selected[i-time_steps:i, :])
            y.append(target[i])

        return np.array(X), np.array(y)

    def prepare_data_for_prediction(self, df, feature_cols, time_steps=20):
        """
        准备LSTM模型预测数据
        :param df: 包含特征的DataFrame
        :param feature_cols: 特征列名列表
        :param time_steps: 时间步长
        :return: 格式化后的X
        """
        # 提取特征
        features = df[feature_cols].values

        # 使用训练时的scaler进行标准化
        features_scaled = self.scaler.transform(features)
        # 应用特征选择
        features_selected = self.selector.transform(features_scaled)

        # 构建时间序列样本
        X = []
        for i in range(time_steps, len(features_scaled) + 1):
            X.append(features_selected[i-time_steps:i, :])

        return np.array(X)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        训练模型（添加类别权重处理）
        :param X_train: 训练特征
        :param y_train: 训练目标
        :param X_val: 验证特征
        :param y_val: 验证目标
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :return: 训练历史
        """
        # 计算类别权重（解决样本不平衡）
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train.flatten())
        class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

        # 回调函数
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * (10 **(epoch / 20)) if epoch < 10 else 1e-6),
            ModelCheckpoint('best_model.keras', save_best_only=True)
        ]

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            shuffle=False  # 时间序列数据不打乱
        )

        logger.info(f"模型训练完成 - 最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
        return history

    def predict(self, X):
        """
        预测结果
        :param X: 输入特征
        :return: 预测概率和类别
        """
        pred_proba = self.model.predict(X)
        pred_class = np.argmax(pred_proba, axis=1)
        return pred_proba, pred_class

    def save_model(self, path):
        """保存模型和标准化器"""
        # 保存Keras模型
        model_path = path.replace('.h5', '.keras')
        self.model.save(model_path)
        # 保存标准化器
        scaler_path = model_path.replace('.keras', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        # 保存特征选择器
        selector_path = model_path.replace('.keras', '_selector.joblib')
        joblib.dump(self.selector, selector_path)
        logger.info(f"模型和标准化器已保存至: {model_path} 和 {scaler_path}")

    def load_model(self, path):
        """加载模型和标准化器"""
        from tensorflow.keras.models import load_model
        # 加载Keras模型
        model_path = path.replace('.h5', '.keras')
        self.model = load_model(model_path)
        # 加载标准化器
        scaler_path = model_path.replace('.keras', '_scaler.joblib')
        self.scaler = joblib.load(scaler_path)
        # 加载特征选择器
        selector_path = model_path.replace('.keras', '_selector.joblib')
        self.selector = joblib.load(selector_path)
        logger.info(f"已从{model_path}和{scaler_path}加载模型和标准化器")


def load_data_from_pkl(file_path):
    """从pkl文件加载数据
    :param file_path: pkl文件路径
    :return: 加载的DataFrame
    """
    # 数据由Baostock提供 (www.baostock.com)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 不存在")
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def train_deep_learning_strategy(data_source, feature_columns, params=None):
    """
    训练深度学习交易策略模型
    :param data_df: 包含特征和目标的DataFrame
    :param feature_columns: 特征列名列表
    :param params: 模型参数
    :return: 训练好的模型和评估结果
    """
    if params is None:
        params = {
            'time_steps': 20,
            'lstm_units': 64,
            'attention_units': 32,
            'dropout_rate': 0.3,
            'epochs': 50,
            'batch_size': 32
        }

    # 从文件路径加载数据或直接使用DataFrame
    if isinstance(data_source, str):
        data_df = load_data_from_pkl(data_source)
    else:
        data_df = data_source

    # 准备数据
    model = StockLSTMAttentionModel(
        input_shape=(params['time_steps'], len(feature_columns)),
        lstm_units=params['lstm_units'],
        attention_units=params['attention_units'],
        dropout_rate=params['dropout_rate']
    )

    X, y = model.prepare_data(
        data_df,
        feature_cols=feature_columns,
        target_col='return_category',  # 确保数据预处理阶段已创建'return_category'列作为模型训练目标
        time_steps=params['time_steps']
    )

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=params['epochs'],
            batch_size=params['batch_size']
        )

        # 评估模型
        loss, accuracy = model.model.evaluate(X_test, y_test)
        cv_scores.append(accuracy)
        logger.info(f"交叉验证准确率: {accuracy:.4f}")

    logger.info(f"平均交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return model, {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores)
    }


def generate_trading_signals(data_source):
    """
    基于MACD、大单净量指标和深度学习模型预测生成交易信号
    :param data_source: 数据文件路径或DataFrame
    :return: 包含交易信号的DataFrame
    """
    # 加载数据
    if isinstance(data_source, str):
        data_df = load_data_from_pkl(data_source)
    else:
        data_df = data_source

    # 验证关键指标列是否存在
    required_columns = ['daily_return', 'daily_return_ma5', 'open', 'high', 'low', 'close', 'volume', 'macd', 'big_order_net']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        logger.error(f'生成交易信号失败，缺少必要列: {missing_columns}')
        logger.info(f'数据中可用列: {data_df.columns.tolist()}')
        raise ValueError(f'缺少必要的技术指标列: {missing_columns}')

    # 加载训练好的模型  # 根据实际特征数量和时间步长设置input_shape
    model = StockLSTMAttentionModel(input_shape=(10, 5))  # 示例: 10个时间步, 5个特征
    model.load_model('stock_prediction_model.keras')
    logger.info('成功加载深度学习模型')

    # 准备模型输入特征
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'macd', 'big_order_net', 'daily_return', 'daily_return_ma5', 'k', 'd', 'j']
    X = model.prepare_data_for_prediction(data_df, feature_columns, time_steps=20)

    # 获取模型预测结果
    pred_proba, pred_class = model.predict(X)
    # 生成预测结果并确保长度匹配
    pred_category = pred_class
    pred_prob = pred_proba.max(axis=1)

    # 填充NaN使预测结果长度与原始数据匹配
    pad_length = len(data_df) - len(pred_category)
    # 使用-1作为缺失值标记（整数类型兼容）
    data_df['predicted_return_category'] = np.pad(pred_category, (pad_length, 0), 'constant', constant_values=-1)
    data_df['prediction_probability'] = np.pad(pred_prob, (pad_length, 0), 'constant', constant_values=np.nan)

    # 添加收益分类说明列
    category_explanations = {
        -1: '缺失数据',
        0: '下跌（当日收益率 ≤ -1%）',
        1: '持平（-1% < 当日收益率 < 1%）',
        2: '上涨（当日收益率 ≥ 1%）'
    }
    data_df['收益分类说明'] = data_df['predicted_return_category'].map(category_explanations)

    # 初始化信号为持有
    df_signals = data_df.copy()
    # 添加日期列并格式化
    df_signals['date'] = df_signals.index
    df_signals['date'] = df_signals['date'].dt.strftime('%Y-%m-%d')
    df_signals['trade_signal'] = 1  # 持有

    # 结合模型预测和技术指标生成买入信号
    # 模型预测上涨(2)且MACD<0且大单净量>阈值，同时预测概率>0.6
    buy_condition = (
        (df_signals['predicted_return_category'] == 2) &
        (df_signals['macd'] < 0) &
        (df_signals['big_order_net'] > DEFAULT_CONFIG.buy_threshold) &
        (df_signals['prediction_probability'] > 0.6)
    )
    df_signals.loc[buy_condition, 'trade_signal'] = 2  # 买入

    # 结合模型预测和技术指标生成卖出信号
    # 模型预测下跌(0)且MACD>0且大单净量<阈值，同时预测概率>0.6
    sell_condition = (
        (df_signals['predicted_return_category'] == 0) &
        (df_signals['macd'] > 0) &
        (df_signals['big_order_net'] < DEFAULT_CONFIG.sell_threshold) &
        (df_signals['prediction_probability'] > 0.6)
    )
    df_signals.loc[sell_condition, 'trade_signal'] = 0  # 卖出

    # 添加信号文字描述
    df_signals['signal_description'] = df_signals['trade_signal'].map({
        0: '卖出',
        1: '持有',
        2: '买入'
    })

    # 计算持有期间收益率
    df_signals['position'] = 0  # 0: 未持仓, 1: 持仓
    df_signals['buy_price'] = np.nan
    df_signals['hold_period_return'] = np.nan

    # 标记持仓状态和买入价格
    in_position = False
    buy_price = 0

    for i, row in df_signals.iterrows():
        if row['trade_signal'] == 2 and not in_position:  # 买入信号且未持仓
            in_position = True
            buy_price = row['close']
            df_signals.at[i, 'position'] = 1
            df_signals.at[i, 'buy_price'] = buy_price
        elif row['trade_signal'] == 0 and in_position:  # 卖出信号且持仓
            in_position = False
            sell_price = row['close']
            return_rate = round((sell_price - buy_price) / buy_price * 100, 2)
            df_signals.at[i, 'position'] = 0
            df_signals.at[i, 'hold_period_return'] = return_rate
        elif in_position:  # 持仓中
            df_signals.at[i, 'position'] = 1

    # 添加持仓状态文字描述
    df_signals['position_status'] = df_signals['position'].map({0: '观望中', 1: '持有中'})

    # 列名映射为中文
    column_mapping = {
        'date': '日期',
        'open': '开盘价',
        'high': '最高价',
        'low': '最低价',
        'close': '收盘价',
        'volume': '成交量',
        'return': '收益率(%)',
        'return_category': '收益分类',
        'macd': 'MACD',
        'big_order_net': '大单净量',
        'position_status': '持仓状态',
        'buy_price': '买入价格',
        'hold_period_return': '持有期间收益率(%)',
        'daily_return': '每日涨跌幅(%)',
        'daily_return_ma5': '5日平均涨跌幅(%)',
        'predicted_return_category': '预测收益分类',
        'prediction_probability': '预测概率',
        '收益分类说明': '收益分类说明'
    }
    df_signals = df_signals.rename(columns=column_mapping)

    # 数值列四舍五入保留两位小数
    numeric_cols = df_signals.select_dtypes(include=['float64', 'int64']).columns
    df_signals[numeric_cols] = df_signals[numeric_cols].round(2)

    return df_signals


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='股票深度学习策略执行')
    parser.add_argument('--data_path',default='./data/sh.600000_浦发银行.pkl', required=False, help='pkl数据文件路径')
    parser.add_argument('--output_path', default='./signals.csv', help='信号输出路径')
    args = parser.parse_args()

    # 检查必要参数
    if not args.data_path:
        parser.error("必须提供--data_path参数，请指定pkl数据文件路径")

    # 生成基于输入文件名的默认输出路径
    if args.output_path == './signals.csv':
        data_basename = os.path.basename(args.data_path)
        data_name, _ = os.path.splitext(data_basename)
        args.output_path = f'./{data_name}_signals.csv'

    # 训练深度学习模型
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'macd', 'big_order_net', 'daily_return', 'daily_return_ma5', 'k', 'd', 'j']
    model, evaluation = train_deep_learning_strategy(args.data_path, feature_columns)
    model.save_model('stock_prediction_model.keras')

    # 生成交易信号（基于训练好的模型）
    signals = generate_trading_signals(args.data_path)

    # 保存结果
    signals.to_csv(args.output_path, index=False)

    # 输出信号统计摘要，提高可读性
    signal_counts = signals['signal_description'].value_counts()
    print(f'交易信号已保存至 {args.output_path}')
    print('信号统计摘要:')
    print(f'买入信号: {signal_counts.get("买入", 0)} 次')
    print(f'持有信号: {signal_counts.get("持有", 0)} 次')
    print(f'卖出信号: {signal_counts.get("卖出", 0)} 次')
    print(f'总信号数: {len(signals)}')
    # 计算并打印总收益率
    total_return = signals['持有期间收益率(%)'].sum()
    print(f'总收益率: {total_return:.2f}%')