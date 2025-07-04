import pandas as pd

# 读取交易日CSV文件
trading_days_df = pd.read_excel('trading_dates.xlsx', parse_dates=True, index_col=0)

# 将交易日转换为索引
trading_days = trading_days_df.index

# 添加一列表示周数
trading_days_df['week'] = trading_days_df.index.to_series().dt.isocalendar().week

# 找到每周的最后一个交易日
last_trading_days = trading_days_df[trading_days_df['week'] != trading_days_df['week'].shift(-1)]
last_trading_days.to_csv('trading_week.csv')
print(last_trading_days)