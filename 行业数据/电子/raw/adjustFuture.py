import pandas as pd


def process_csv(file_path):
    df = pd.read_csv(file_path, encoding='gbk')
    # 检查最后一行的最后一个值是否为"WIND"，如果是，则删除最后一列
    wind_index = df[df.apply(lambda row: row.astype(str).str.contains('Wind').any(), axis=1)].index
    if not wind_index.empty:
        df = df.iloc[:wind_index[0]]
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)
    return df

# 读取A和B两个CSV文件
a_df = process_csv('CZCE锰硅.csv')
a_df.fillna(method='ffill', inplace=True)
a_df.to_csv('CZCE锰硅_amended.csv')






# b_df = process_csv('Wind_南华硅铁指数.csv')

# a_price_col = a_df.columns[3]
# b_price_col = b_df.columns[0]

# # 计算B的每日涨跌幅
# b_df['pct_change'] = b_df.iloc[:,0].pct_change()

# # 找到A文件中第一个有值的日期
# first_valid_date_in_a = a_df[a_price_col].first_valid_index()

# # 删除A文件中第一个有值的日期之前的所有日期
# a_df = a_df.loc[first_valid_date_in_a:]

# # 合并A和B的日期
# all_dates = a_df.index.union(b_df.index)
# a_df_changed = a_df.reindex(all_dates)

# # 填补空值
# for date in a_df_changed.index:
#     row = a_df_changed.loc[date]
#     if pd.isna(row[3]):
#         prev_date = date - pd.Timedelta(days=1)
#         print(date)
#         # 找到前一个交易日
#         while prev_date not in a_df.index or pd.isna(a_df.at[prev_date, a_price_col]):
#             prev_date -= pd.Timedelta(days=1)
            
#         if date in b_df.index and prev_date in b_df.index:
#             # 使用前一天的价格乘以B的涨跌幅来填补
#             a_df_changed.at[date, a_price_col] = a_df_changed.at[prev_date, a_price_col] * (1 + b_df.at[date, 'pct_change'])

# # 保存处理后的数据到新的CSV文件
# a_df_changed.to_csv('A_filled.csv')