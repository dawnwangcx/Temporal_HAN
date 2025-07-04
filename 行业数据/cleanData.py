import pandas as pd
import os


def clean_daily_data(filename):
    # 读取数据CSV文件
    data_df = pd.read_csv(filename+'.csv', parse_dates=True, index_col=0)

    # 读取交易日CSV文件
    trading_days_df = pd.read_excel('trading_dates.xlsx', parse_dates=True, index_col=0)

    # 将交易日转换为索引
    trading_days = trading_days_df.index

    # 合并A和B的日期
    all_dates = data_df.index.union(trading_days)
    data_df = data_df.reindex(all_dates)

    # 找到A文件中第一个有值的日期
    first_valid_date_in_a = data_df.first_valid_index()

    # 删除A文件中第一个有值的日期之前的所有日期
    data_df = data_df.loc[first_valid_date_in_a:]

    data_df.fillna(method='ffill', inplace=True)

    # # 过滤出交易日的数据
    filtered_data_df = data_df[data_df.index.isin(trading_days)]
    filtered_data_df.to_csv(filename+'_cleaned.csv',encoding='utf-8-sig')


def clean_weekly_data(filename):
    
    # 读取CSV文件
    df = pd.read_csv(filename+'.csv', parse_dates=True, index_col=0)

    # 读取交易日列表
    trading_days = pd.read_excel('trading_dates.xlsx', parse_dates=True, index_col=0)


    # 添加一列表示周数
    trading_days['week'] = trading_days.index.to_period('W')

    # 找到每周的最后一个交易日
    last_trading_days = trading_days[trading_days['week'] != trading_days['week'].shift(-1)]

    # # 删除周数列
    # last_trading_days = last_trading_days.drop(columns=['week'])

    # 确保每周的最后一个值放在该交易日上
    last_trading_days = last_trading_days[last_trading_days.index.isin(trading_days.index)]

    last_trading_days['date']=last_trading_days.index

    weekly_last = df.resample('W').last()
    weekly_last['week'] = weekly_last.index.to_period('W')

    cleaned = pd.merge(last_trading_days,weekly_last, on='week',how='left')
    cleaned.set_index('date', inplace=True)
    
    # 找到A文件中第一个有值的日期
    first_valid_date = cleaned.first_valid_index()

    # 删除A文件中第一个有值的日期之前的所有日期
    cleaned = cleaned.loc[first_valid_date:]

    cleaned.fillna(method='ffill', inplace=True)
    cleaned.drop(columns = 'week', inplace=True)
    cleaned.to_csv(filename+'_cleaned.csv',encoding='utf-8-sig')
    
os.chdir('D:\essay\han\dgl\行业数据')
home =  os.getcwd()
for file in os.listdir(home):
    if os.path.isfile(file) and "daily.csv" in file:
        path = os.path.join(home, file)
        print(file)
        basename = os.path.splitext(file)[0]
        clean_daily_data(basename)

    if os.path.isfile(file) and "weekly.csv" in file:
        path = os.path.join(home, file)
        print(file)
        basename = os.path.splitext(file)[0]
        clean_weekly_data(basename)