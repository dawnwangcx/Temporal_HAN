import pandas as pd
import os
import shutil

def replace_files(folder_a, folder_b):
    # 获取文件夹A中的文件列表
    files_a = os.listdir(folder_a)
    for root, dirs, files in os.walk(folder_b):
        for file_name in files:
            cleaned_file_name=file_name.split('.')[0]+'_cleaned.'+file_name.split('.')[1] 
            print(cleaned_file_name)
            if file_name in files_a:
                file_a_path = os.path.join(folder_a, file_name)
                file_b_path = os.path.join(root, file_name)
            if cleaned_file_name in files_a:
                file_a_path = os.path.join(folder_a, cleaned_file_name)
                file_b_path = os.path.join(root, file_name)
                # 将文件夹A中的文件复制到文件夹B中，替换同名文件
                shutil.copy2(file_a_path, file_b_path)
                print(f"Replaced {file_b_path} with {file_a_path}")


folder_a_path = "D:\essay\han\dgl\行业数据"  # 替换为实际的文件夹路径
folder_b_path = "D:\essay\han\dgl\中证行业数据"  # 替换为实际的文件夹路径

replace_files(folder_a_path, folder_b_path)

def merge_files(directory):
    daily_merged_df = pd.DataFrame()
    monthly_merged_df = pd.DataFrame()
    weekly_merged_df = pd.DataFrame()
    for file_name in os.listdir(directory):
        print(file_name)
        file_path = os.path.join(directory, file_name)
        
        if 'daily' in file_path:
            df = pd.read_csv(file_path,index_col=0,encoding='utf-8-sig')
            if daily_merged_df is None:
                daily_merged_df = df
            else:
                daily_merged_df = pd.merge(daily_merged_df,df,left_index=True, right_index=True, how='outer')

        if 'weekly' in file_path:
            df = pd.read_csv(file_path,index_col=0,encoding='utf-8-sig')
            if weekly_merged_df is None:
                weekly_merged_df = df
            else:
                weekly_merged_df = pd.merge(weekly_merged_df,df,left_index=True, right_index=True, how='outer')

        if 'monthly' in file_path:
            df = pd.read_csv(file_path,index_col=0,encoding='utf-8-sig')
            if monthly_merged_df is None:
                monthly_merged_df = df
            else:
                monthly_merged_df = pd.merge(monthly_merged_df,df,left_index=True, right_index=True, how='outer')

        # if file_name[0]=='9':
        #     df = pd.read_csv(file_path,index_col=0)
        #     if daily_merged_df is None:
        #         daily_merged_df = df
        #     else:
        #         daily_merged_df = pd.merge(daily_merged_df,df,left_index=True, right_index=True, how='outer')
        
    if daily_merged_df.empty==False:
        daily_merged_df =daily_merged_df[daily_merged_df.index >= '2013-01-01']
        daily_merged_df.to_csv(directory.split("\\")[-1]+'_daily.csv',encoding='utf-8-sig')

    if weekly_merged_df.empty==False:
        weekly_merged_df =weekly_merged_df[weekly_merged_df.index >= '2013-01-01']
        weekly_merged_df.to_csv(directory.split("\\")[-1]+'_weekly.csv',encoding='utf-8-sig')
    
    if monthly_merged_df.empty==False:
        monthly_merged_df =monthly_merged_df[monthly_merged_df.index >= '2013-01-01']
        monthly_merged_df.to_csv(directory.split("\\")[-1]+'_monthly.csv',encoding='utf-8-sig')



os.chdir('D:\essay\han\dgl\中证行业数据')

home =  os.getcwd()

for dir in os.listdir(home):
    if os.path.isdir(dir):
        directory = os.path.join(home, dir)
        print(directory)
        merge_files(directory)

