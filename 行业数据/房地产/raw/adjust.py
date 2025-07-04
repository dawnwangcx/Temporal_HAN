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

df = process_csv('Wind_二手房成交.csv')

# # 遍历每一列
# for column in df.columns:
#     # 找到该列的第一个非空值的索引
#     first_valid_index = df[column].first_valid_index()
    
#     if first_valid_index is not None:
#         # 从第一个非空值开始，将后续的空值填充为0
#         df.loc[first_valid_index:, column] = df.loc[first_valid_index:, column].fillna(0)
df['二手房成交总面积'] = df.sum(axis=1)
df[['二手房成交总面积']].to_csv('Wind_二手房成交_amended.csv')