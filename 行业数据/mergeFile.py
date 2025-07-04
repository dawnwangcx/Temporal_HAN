import os
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

def process_excel(file_path):
    # 读取Excel文件，跳过前6行
    df = pd.read_excel(file_path, skiprows=6)
    # 将第二行的第二列作为列名
    value_column_name = pd.read_excel(file_path).iloc[0, 1] #第一行是header
    print(file_path)
    df.columns = ['date', value_column_name]
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)
    return df

def merge_files(directory):
    print(directory)
    merged_df = None

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        
        if file_path.endswith('.csv'):
            df = process_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = process_excel(file_path)
        else:
            continue
        
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')
    merged_df = merged_df[merged_df.index >= '2013-01-01']
    return merged_df

os.chdir('D:\essay\han\dgl\行业数据')
home =  os.getcwd()
sectors = [dir for dir in os.listdir(home) if os.path.isdir(dir)]
print(sectors)
for sector in sectors:
    sector_dir = os.path.join(home, sector)
    for dir in os.listdir(sector_dir):
        directory = os.path.join(sector_dir, dir)
        if os.path.isdir(directory):
            print(directory)
            merged_df = merge_files(directory)
            parent_dir = os.path.basename(os.path.dirname(directory))
            curr_dir = os.path.basename(directory)
            merged_df.to_csv(f'./{parent_dir}_{curr_dir}.csv',encoding='utf-8-sig')
