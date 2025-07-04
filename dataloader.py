import os
import dgl
import torch
import torch.nn.functional as F
import pandas as pd
import indicators as indi
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from model_hetero import HAN
from model_dawn import TemporalAttention, Adaptive_Fusion
import sys
import argparse
def get_interval_data(df, start_date, end_date, lookback=0, lookforward=0):
        if df.index.dtype != 'datetime.datetime':
            df.index = pd.to_datetime(df.index)
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_data = df.loc[mask]
        if lookback:
            concat_df = df.loc[:df_data.index[0]].tail(lookback+1)
            df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
        if lookforward:
            concat_df = df.loc[df_data.index[-1]:].head(lookforward+1)
            df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
        df_data = df_data.astype(float)
        return df_data

def pad_features(features, target_dim):
    """
    对特征进行前后填充，使其达到目标维度
    :param features: 输入特征 (batch_size, seq_len, feature_dim)
    :param target_dim: 目标特征维度
    :return: 填充后的特征
    """
    if features is None or features.numel() == 0:
        return torch.zeros((0, target_dim))
    
    current_dim = features.shape[2]
    if current_dim == target_dim:
        return features

    padding = target_dim - current_dim

    return torch.nn.functional.pad(features, (0, padding))

def pad_tensors(tensor_list):
    max_dim = max(tensor.shape[1] for tensor in tensor_list)
    padded_tensors = [F.pad(tensor, (0, max_dim - tensor.shape[1])) for tensor in tensor_list]
    stacked_tensors = torch.stack(padded_tensors)
    mask = torch.ones_like(stacked_tensors)
    for i, tensor in enumerate(tensor_list):
        mask[i, :, tensor.shape[1]:] = 0
    return stacked_tensors, mask

class PaddedHeteroDataset(dgl.data.DGLDataset):
    def __init__(self, root_folder,start_date,end_date,n_step_ahead,daily_window_size,weekly_window_size=5,monthly_window_size=6):
        self.root_folder = root_folder

        self.start_day = start_date-relativedelta(days=window_size)
        self.start_month = start_date-relativedelta(months=monthly_window_size)
        self.start_week = start_date-relativedelta(weeks=weekly_window_size)
        self.end_date = end_date

        self.n_step_ahead = n_step_ahead
        self.daily_window_size= daily_window_size
        self.weekly_window_size =weekly_window_size
        self.monthly_window_size =monthly_window_size

        labels = []
        feats = []
        self.industry_labels = []
        self.industry_features = []
        self.daily_features = []
        self.weekly_features = []
        self.monthly_features = []

        industry_to_industry_edges = []
        industry_to_daily_edges = []
        industry_to_weekly_edges = []
        industry_to_monthly_edges = []
        
        industry_nodes = []
        daily_derivative_nodes = []
        weekly_derivative_nodes = []
        monthly_derivative_nodes = []

        industry_node_id = 0
        current_industry_node_id = 0
        current_daily_node_id = 0
        current_weekly_node_id = 0
        current_monthly_node_id = 0
        industry_id = {}

        
        # 遍历根文件夹中的每个子文件夹,先处理每个industry node
        for directory in os.listdir( self.root_folder):
            industry_folder_path = os.path.join( self.root_folder, directory)
            if os.path.isdir(industry_folder_path):
                print(directory)
                for csv_file in os.listdir(industry_folder_path):
                    # 遍历每个子文件夹中的CSV文件
                    csv_file_path = os.path.join(industry_folder_path, csv_file)
                    # 判断是否为行业量价信息
                    if csv_file.split('.')[1]=='CSI':
                        industry_id[directory] = industry_node_id
                        # 处理行业量价信息
                        industry_nodes.append(industry_node_id)
                        data = pd.read_csv(csv_file_path, index_col=0,parse_dates=True)
                        data_interval = get_interval_data(data, self.start_day, self.end_date, lookback=self.n_step_ahead,lookforward=self.n_step_ahead)
                        data = data[(data.index >=  self.start_day) & (data.index <=  self.end_date)]  # 筛选时间范围 
                        label = self.calc_close_close_return(data_interval, csv_file[:-4]+'_close')
                        labels.append(label)
                        feats.append(data)
                        industry_node_id += 1
                        assert data.shape[0]==len(label)
                        
        for industry in industry_id.keys():
            industry_feature = feats[industry_id[industry]]
            industry_label = labels[industry_id[industry]]
            daily_rolling_data = []
            weekly_rolling_data = []
            monthly_rolling_data = []
            industry_rolling = []
            label_rolling = []
            daily_data = None
            weekly_data = None
            monthly_data = None
            if os.path.exists(os.path.join(self.root_folder, industry+'_daily.csv')):
                data = pd.read_csv(os.path.join(self.root_folder, industry+'_daily.csv'), index_col=0,parse_dates=True)
                daily_data = data[(data.index >=  self.start_day) & (data.index <=  self.end_date)]  # 筛选时间范围
                daily_derivative_nodes.append(current_daily_node_id)
                # self.daily_features.append(torch.tensor(data.values, dtype=torch.float))
                # print(data)
                # print('daily',data.shape)
                industry_to_daily_edges.append((current_industry_node_id, current_daily_node_id))
                current_daily_node_id += 1
            if os.path.exists(os.path.join(self.root_folder, industry+'_weekly.csv')):
                data = pd.read_csv(os.path.join(self.root_folder, industry+'_weekly.csv'), index_col=0,parse_dates=True)
                weekly_data = data[(data.index >=  self.start_week) & (data.index <=  self.end_date)]  # 筛选时间范围
                weekly_derivative_nodes.append(current_weekly_node_id)
                # weekl_features.append(weekly_data)
                # self.weekly_features.append(torch.tensor(data.values, dtype=torch.float))
                industry_to_weekly_edges.append((current_industry_node_id, current_weekly_node_id))
                current_weekly_node_id += 1
            if os.path.exists(os.path.join(self.root_folder, industry+'_monthly.csv')):
                data = pd.read_csv(os.path.join(self.root_folder, industry+'_monthly.csv'), index_col=0,parse_dates=True)
                monthly_data = data[(data.index >=  self.start_month) & (data.index <=  self.end_date)]  # 筛选时间范围
                monthly_derivative_nodes.append(current_monthly_node_id)
                # self.monthly_features.append(torch.tensor(data.values, dtype=torch.float))
                # monthly_features.append(monthly_data)
                industry_to_monthly_edges.append((current_industry_node_id, current_monthly_node_id))
                current_monthly_node_id += 1

            for idx in range(industry_feature.shape[0] - self.daily_window_size):
                industry_feature_roll = industry_feature[idx:idx + self.daily_window_size]#日窗口长度
                label = industry_label[idx + self.daily_window_size]#日窗口长度
                industry_rolling.append(industry_feature_roll.values.tolist())
                label_rolling.append(label)

                current_end_date = industry_feature_roll.index.max()
                if daily_data is not None:
                    # 日频数据滚动窗口
                    daily_window_data = daily_data.iloc[idx:idx + self.daily_window_size]
                    # daily_window_data = pad_data(daily_window_data, max_features_daily)
                    daily_rolling_data.append(daily_window_data.values.tolist())

                if weekly_data is not None:
                    # 周频数据滚动窗口
                    weekly_end_date = current_end_date
                    # weekly_start_date = weekly_end_date - timedelta(weeks=self.weekly_window_size)
                    weekly_window_data = weekly_data[weekly_data.index <= weekly_end_date].iloc[-self.weekly_window_size:]
                    assert weekly_window_data.shape[0]==self.weekly_window_size
                    # weekly_window_data = pad_data(weekly_window_data, max_features_weekly)
                    weekly_rolling_data.append(weekly_window_data.values.tolist())

                if monthly_data is not None:
                    # 月频数据滚动窗口
                    monthly_end_date = current_end_date
                    # monthly_start_date = monthly_end_date - pd.DateOffset(months=self.monthly_window_size)
                    # monthly_window_data = monthly_data[(monthly_data.index >= monthly_start_date) & (monthly_data.index <= monthly_end_date)]
                    # #如果横跨多个月份，取最后x个月
                    # if monthly_window_data.shape[0]>self.monthly_window_size:
                    #     monthly_window_data=monthly_window_data.iloc[-self.monthly_window_size:]
                    #最后一个月份与日频对齐，向前数x个月
                    monthly_window_data = monthly_data[monthly_data.index <= monthly_end_date].iloc[-self.monthly_window_size:]
                    assert monthly_window_data.shape[0]==self.monthly_window_size
                    # monthly_window_data = pad_data(monthly_window_data, max_features_monthly)
                    monthly_rolling_data.append(monthly_window_data.values.tolist())

            self.industry_features.append(torch.tensor(industry_rolling, dtype=torch.float))
            self.industry_labels.append(torch.tensor(label_rolling, dtype=torch.float))
            self.daily_features.append(torch.tensor(daily_rolling_data, dtype=torch.float))
            self.weekly_features.append(torch.tensor(weekly_rolling_data, dtype=torch.float))
            self.monthly_features.append(torch.tensor(monthly_rolling_data, dtype=torch.float))
            #shape:[industry_id,num_days,window_size,num_features]
        print(len(self.industry_features))
        print(len(self.daily_features))
        print(len(self.weekly_features))
        print(len(self.monthly_features))
        print(self.industry_features[2].shape)
        print(self.weekly_features[2].shape)
        print(self.monthly_features[2].shape)

        # 创建行业节点之间的全连接
        for i in range(len(industry_nodes)):
            for j in range(i + 1, len(industry_nodes)):
                industry_to_industry_edges.append((industry_nodes[i], industry_nodes[j]))
                industry_to_industry_edges.append((industry_nodes[j], industry_nodes[i]))

        self.industry_to_industry_edges = (torch.tensor([e[0] for e in industry_to_industry_edges]), torch.tensor([e[1] for e in industry_to_industry_edges]))
        self.industry_to_daily_edges = (torch.tensor([e[0] for e in industry_to_daily_edges]), torch.tensor([e[1] for e in industry_to_daily_edges]))
        self.industry_to_weekly_edges = (torch.tensor([e[0] for e in industry_to_weekly_edges]), torch.tensor([e[1] for e in industry_to_weekly_edges]))
        self.industry_to_monthly_edges = (torch.tensor([e[0] for e in industry_to_monthly_edges]), torch.tensor([e[1] for e in industry_to_monthly_edges]))

        super().__init__(name='industry_hetero')

    def process(self):
    
        max_daily_dim = max(tensor.shape[2] for tensor in self.daily_features if tensor.numel()!=0)
        max_weekly_dim = max(tensor.shape[2] for tensor in self.weekly_features if tensor.numel()!=0)
        max_monthly_dim = max(tensor.shape[2] for tensor in self.monthly_features if tensor.numel()!=0)

        self.graphs = []
        self.labels = []
        
        for idx in range(self.industry_features[0].shape[0]):
            label_all = []
            industry_all = []
            daily_feat_all = []
            daily_mask_all = []
            weekly_feat_all = []
            weekly_mask_all = []
            monthly_feat_all = []
            monthly_mask_all = []
            for industry_id in range(len(self.industry_features)):
                industry_feat = self.industry_features[industry_id][idx]
                label = self.industry_labels[industry_id][idx]
                # shape: window_size, feat_num
                label_all.append(label)
                industry_all.append(industry_feat)
                if self.daily_features[industry_id].numel()!=0:
                    daily_data = self.daily_features[industry_id][idx]
                    daily_padded_tensor = torch.zeros((self.daily_window_size, max_daily_dim))
                    daily_mask = torch.zeros((self.daily_window_size, max_daily_dim), dtype=torch.bool)
                    # 填充数据
                    daily_padded_tensor[ :, :daily_data.shape[1]] = daily_data
                    daily_mask[:, :daily_data.shape[1]] = 1
                    # print(daily_padded_tensor.shape)
                    daily_feat_all.append(daily_padded_tensor)
                    daily_mask_all.append(daily_mask)
                if self.weekly_features[industry_id].numel()!=0:
                    weekly_data = self.weekly_features[industry_id][idx]
                    weekly_padded_tensor = torch.zeros((self.weekly_window_size, max_weekly_dim))
                    weekly_mask = torch.zeros((self.weekly_window_size, max_weekly_dim), dtype=torch.bool)
                    # 填充数据
                    weekly_padded_tensor[ :, :weekly_data.shape[1]] = weekly_data
                    weekly_mask[:, :weekly_data.shape[1]] = 1
                    weekly_feat_all.append(weekly_padded_tensor)
                    weekly_mask_all.append(weekly_mask)
                if self.monthly_features[industry_id].numel()!=0:
                    monthly_data = self.monthly_features[industry_id][idx]
                    monthly_padded_tensor = torch.zeros((self.monthly_window_size, max_monthly_dim))
                    monthly_mask = torch.zeros((self.monthly_window_size, max_monthly_dim), dtype=torch.bool)
                    # 填充数据
                    monthly_padded_tensor[ :, :monthly_data.shape[1]] = monthly_data
                    monthly_mask[:, :monthly_data.shape[1]] = 1 
                    monthly_feat_all.append(monthly_padded_tensor)
                    monthly_mask_all.append(monthly_mask)

            daily_feat_all = torch.stack(daily_feat_all, dim=0)
            weekly_feat_all = torch.stack(weekly_feat_all, dim=0)
            monthly_feat_all = torch.stack(monthly_feat_all, dim=0)
            daily_mask_all = torch.stack(daily_mask_all, dim=0)
            weekly_mask_all = torch.stack(weekly_mask_all, dim=0)
            monthly_mask_all = torch.stack(monthly_mask_all, dim=0)
            industry_all =  torch.stack(industry_all, dim=0)
            label_all = torch.stack(label_all, dim=0)
            # 创建异构图
            hetero_graph = dgl.heterograph({
                ('industry', 'industry_connects_industry', 'industry'): self.industry_to_industry_edges,
                ('industry', 'industry_daily_derivative', 'daily_derivative'): self.industry_to_daily_edges,
                ('industry', 'industry_weekly_derivative', 'weekly_derivative'): self.industry_to_weekly_edges,
                ('industry', 'industry_monthly_derivative', 'monthly_derivative'): self.industry_to_monthly_edges
            })

            # 为节点添加特征
            hetero_graph.nodes['industry'].data['features'] = industry_all
            hetero_graph.nodes['industry'].data['labels'] = label_all
            hetero_graph.nodes['daily_derivative'].data['features'] = daily_feat_all
            hetero_graph.nodes['daily_derivative'].data['mask'] = daily_mask_all
            hetero_graph.nodes['weekly_derivative'].data['features'] = weekly_feat_all
            hetero_graph.nodes['weekly_derivative'].data['mask'] = weekly_mask_all
            hetero_graph.nodes['monthly_derivative'].data['features'] = monthly_feat_all
            hetero_graph.nodes['monthly_derivative'].data['mask'] = monthly_mask_all

            self.graphs.append(hetero_graph)
            self.labels.append(label_all)  # 假设标签为0
        print('num graphs total:',len(self.graphs))
        print('num of labels:',len(self.labels))


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    
    def calc_close_close_return(self,df,column_name):
        df['trend_return'] = df[column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values

    def calc_open_close_return(self,df,close_column_name, open_column_name):
        df['trend_return'] = df[close_column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values

class PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder,start_date,end_date,n_step_ahead):
        self.root_folder = root_folder
        self.start_date = start_date
        self.end_date = end_date
        self.n_step_ahead = n_step_ahead
        super().__init__(name='industry_hetero')

    def process(self):

        industry_labels = []

        industry_nodes = []
        daily_derivative_nodes = []
        weekly_derivative_nodes = []
        monthly_derivative_nodes = []
        
        industry_features = []
        daily_features = []
        weekly_features = []
        monthly_features = []

        industry_to_industry_edges = []
        industry_to_daily_edges = []
        industry_to_weekly_edges = []
        industry_to_monthly_edges = []
        industry_to_industry_edges = []
        
        industry_node_id = 0
        current_industry_node_id = 0
        current_daily_node_id = 0
        current_weekly_node_id = 0
        current_monthly_node_id = 0
        industry_id = {}
        
        # 遍历根文件夹中的每个子文件夹,先处理每个industry node
        for directory in os.listdir( self.root_folder):
            industry_folder_path = os.path.join( self.root_folder, directory)
            if os.path.isdir(industry_folder_path):
                print(directory)
                for csv_file in os.listdir(industry_folder_path):
                    # 遍历每个子文件夹中的CSV文件
                    csv_file_path = os.path.join(industry_folder_path, csv_file)
                    # 判断是否为行业量价信息
                    if csv_file.split('.')[1]=='CSI':
                        features = []
                        industry_id[directory] = industry_node_id
                        # 处理行业量价信息
                        industry_nodes.append(industry_node_id)
                        data = pd.read_csv(csv_file_path, index_col=0,parse_dates=True)
                        data_interval = get_interval_data(data, self.start_date, self.end_date, lookback=self.n_step_ahead,lookforward=self.n_step_ahead)
                        data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围 
                        label = self.calc_close_close_return(data_interval, csv_file[:-4]+'_close')
                        industry_labels.append(torch.tensor(label, dtype=torch.float))
                        industry_features.append(torch.tensor(data.values, dtype=torch.float))
                        industry_node_id += 1
                        assert data.shape[0]==len(label)
                        
            
        # 遍历根文件夹中的每个子文件夹,处理衍生行情node
        for directory in os.listdir(self.root_folder):
            industry_folder_path = os.path.join(self.root_folder, directory)
            if os.path.isfile(industry_folder_path):
                industry = directory.split('_')[0]
                frequncy = directory.split('_')[1].split('.')[0]
                if industry in industry_id:
                    current_industry_node_id = industry_id[industry]
                    # 处理衍生数据
                    data = pd.read_csv(industry_folder_path, index_col=0,parse_dates=True)
                    data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围
                    if frequncy == 'daily':
                            daily_derivative_nodes.append(current_daily_node_id)
                            daily_features.append(torch.tensor(data.values, dtype=torch.float))
                            # print(data)
                            # print('daily',data.shape)
                            industry_to_daily_edges.append((current_industry_node_id, current_daily_node_id))
                            current_daily_node_id += 1
                    elif frequncy == 'weekly':
                        weekly_derivative_nodes.append(current_weekly_node_id)
                        weekly_features.append(torch.tensor(data.values, dtype=torch.float))
                        industry_to_weekly_edges.append((current_industry_node_id, current_weekly_node_id))
                        current_weekly_node_id += 1
                    elif frequncy == 'monthly':
                        monthly_derivative_nodes.append(current_monthly_node_id)
                        monthly_features.append(torch.tensor(data.values, dtype=torch.float))
                        industry_to_monthly_edges.append((current_industry_node_id, current_monthly_node_id))
                        current_monthly_node_id += 1

        # for ele in daily_features:
        #     print('ele',ele.shape)
        industry_features = torch.stack(industry_features)
        industry_labels = torch.stack(industry_labels)

        max_daily_dim = max(tensor.shape[1] for tensor in daily_features)
        daily_features_padded = [F.pad(tensor, (0, max_daily_dim - tensor.shape[1])) for tensor in daily_features]
        daily_features_padded = torch.stack(daily_features_padded)
        print(daily_features_padded.shape)
        # 创建一个mask，mask掉padding的部分
        daily_mask = torch.ones_like(daily_features_padded)
        for i, tensor in enumerate(daily_features):
            daily_mask[i, :, tensor.shape[1]:] = 0

        max_weekly_dim = max(tensor.shape[1] for tensor in weekly_features)
        weekly_features_padded = [F.pad(tensor, (0, max_weekly_dim - tensor.shape[1])) for tensor in weekly_features]
        weekly_features_padded = torch.stack(weekly_features_padded)
        # 创建一个mask，mask掉padding的部分
        weekly_mask = torch.ones_like(weekly_features_padded)
        for i, tensor in enumerate(weekly_features):#有数的地方是1，没数的地方是0
            weekly_mask[i, :, tensor.shape[1]:] = 0

        max_monthly_dim = max(tensor.shape[1] for tensor in monthly_features)
        monthly_features_padded = [F.pad(tensor, (0, max_monthly_dim - tensor.shape[1])) for tensor in monthly_features]
        monthly_features_padded = torch.stack(monthly_features_padded)
        # 创建一个mask，mask掉padding的部分
        monthly_mask = torch.ones_like(monthly_features_padded)
        for i, tensor in enumerate(monthly_features):
            monthly_mask[i, :, tensor.shape[1]:] = 0

        # 创建行业节点之间的全连接
        for i in range(len(industry_nodes)):
            for j in range(i + 1, len(industry_nodes)):
                industry_to_industry_edges.append((industry_nodes[i], industry_nodes[j]))
                industry_to_industry_edges.append((industry_nodes[j], industry_nodes[i]))

        industry_to_industry_edges = (torch.tensor([e[0] for e in industry_to_industry_edges]), torch.tensor([e[1] for e in industry_to_industry_edges]))
        industry_to_daily_edges = (torch.tensor([e[0] for e in industry_to_daily_edges]), torch.tensor([e[1] for e in industry_to_daily_edges]))
        industry_to_weekly_edges = (torch.tensor([e[0] for e in industry_to_weekly_edges]), torch.tensor([e[1] for e in industry_to_weekly_edges]))
        industry_to_monthly_edges = (torch.tensor([e[0] for e in industry_to_monthly_edges]), torch.tensor([e[1] for e in industry_to_monthly_edges]))

        # 创建异构图
        hetero_graph = dgl.heterograph({
            ('industry', 'connects', 'industry'): industry_to_industry_edges,
            ('industry', 'has_daily_derivative', 'daily_derivative'): industry_to_daily_edges,
            ('industry', 'has_weekly_derivative', 'weekly_derivative'): industry_to_weekly_edges,
            ('industry', 'has_monthly_derivative', 'monthly_derivative'): industry_to_monthly_edges
        })

        # 为节点添加特征
        hetero_graph.nodes['industry'].data['features'] = industry_features
        hetero_graph.nodes['industry'].data['labels'] = industry_labels
        if daily_features_padded.numel() > 0:
            hetero_graph.nodes['daily_derivative'].data['features'] = daily_features_padded
            hetero_graph.nodes['daily_derivative'].data['mask'] = daily_mask
        if weekly_features_padded.numel() > 0:
            hetero_graph.nodes['weekly_derivative'].data['features'] = weekly_features_padded
            hetero_graph.nodes['weekly_derivative'].data['mask'] = weekly_mask
        if monthly_features_padded.numel() > 0:
            hetero_graph.nodes['monthly_derivative'].data['features'] = monthly_features_padded
            hetero_graph.nodes['monthly_derivative'].data['mask'] = monthly_mask

        self.graph = hetero_graph
        self.label = industry_labels

    def __getitem__(self, i):
        return self.graph[i], self.label[i]

    def __len__(self):
        return len(self.graph)
    
    def calc_close_close_return(self,df,column_name):
        df['trend_return'] = df[column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values

    def calc_open_close_return(self,df,close_column_name, open_column_name):
        df['trend_return'] = df[close_column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values
    
'''
每一个节点为一个特征
'''
class FullHeteroDataset(dgl.data.DGLDataset):
    def __init__(self, root_folder,start_date,end_date,n_step_ahead):
        self.root_folder = root_folder
        self.start_date = start_date
        self.end_date = end_date
        self.n_step_ahead = n_step_ahead
        super().__init__(name='industry_hetero')

    def process(self):

        industry_labels = []

        industry_nodes = []
        daily_derivative_nodes = []
        weekly_derivative_nodes = []
        monthly_derivative_nodes = []
        
        industry_features = []
        daily_features = []
        weekly_features = []
        monthly_features = []

        industry_to_industry_edges = []
        industry_to_daily_edges = []
        industry_to_weekly_edges = []
        industry_to_monthly_edges = []
        industry_to_industry_edges = []
        
        industry_node_id = 0
        current_industry_node_id = 0
        current_daily_node_id = 0
        current_weekly_node_id = 0
        current_monthly_node_id = 0
        industry_id = {}
        
        # 遍历根文件夹中的每个子文件夹,先处理每个industry node
        for directory in os.listdir( self.root_folder):
            industry_folder_path = os.path.join( self.root_folder, directory)
            if os.path.isdir(industry_folder_path):
                print(directory)
                for csv_file in os.listdir(industry_folder_path):
                    # 遍历每个子文件夹中的CSV文件
                    csv_file_path = os.path.join(industry_folder_path, csv_file)
                    # 判断是否为行业量价信息
                    if csv_file.split('.')[1]=='CSI':
                        features = []
                        industry_id[directory] = industry_node_id
                        # 处理行业量价信息
                        industry_nodes.append(industry_node_id)
                        data = pd.read_csv(csv_file_path, index_col=0,parse_dates=True)
                        data_interval = get_interval_data(data, self.start_date, self.end_date, lookback=self.n_step_ahead,lookforward=self.n_step_ahead)
                        data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围 
                        labels = self.calc_close_close_return(data_interval, csv_file[:-4]+'_close')
                        industry_labels.append(torch.tensor(labels, dtype=torch.float))
                        industry_features.append(torch.tensor(data.values, dtype=torch.float))
                        industry_node_id += 1
                        assert data.shape[0]==len(labels)
                        
            
        # 遍历根文件夹中的每个子文件夹,处理衍生行情node
        for directory in os.listdir(self.root_folder):
            industry_folder_path = os.path.join(self.root_folder, directory)
            if os.path.isfile(industry_folder_path):
                industry = directory.split('_')[0]
                frequncy = directory.split('_')[1].split('.')[0]
                if industry in industry_id:
                    current_industry_node_id = industry_id[industry]
                    # 处理衍生数据
                    data = pd.read_csv(industry_folder_path, index_col=0,parse_dates=True)
                    data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围
                    if frequncy == 'daily':
                        for column, feature in data.items():
                            print(feature)
                            daily_derivative_nodes.append(current_daily_node_id)
                            daily_features.append(torch.tensor(feature, dtype=torch.float))
                            # print(data)
                            print('daily',feature.shape)
                            industry_to_daily_edges.append((current_industry_node_id, current_daily_node_id))
                            current_daily_node_id += 1
                    elif frequncy == 'weekly':
                        for column, feature in data.items():
                            weekly_derivative_nodes.append(current_weekly_node_id)
                            weekly_features.append(torch.tensor(feature, dtype=torch.float))
                            industry_to_weekly_edges.append((current_industry_node_id, current_weekly_node_id))
                            current_weekly_node_id += 1
                    elif frequncy == 'monthly':
                        for column, feature in data.items():
                            monthly_derivative_nodes.append(current_monthly_node_id)
                            monthly_features.append(torch.tensor(feature, dtype=torch.float))
                            industry_to_monthly_edges.append((current_industry_node_id, current_monthly_node_id))
                            current_monthly_node_id += 1

        # for ele in daily_features:
        #     print('ele',ele.shape)
        industry_features = torch.stack(industry_features)
        industry_labels = torch.stack(industry_labels)
        daily_features = torch.stack(daily_features) if daily_features else torch.tensor([])
        # for i in range(daily_features.shape[0]):
        #     print(daily_features[i].shape)
        weekly_features = torch.stack(weekly_features) if weekly_features else torch.tensor([])
        monthly_features = torch.stack(monthly_features) if monthly_features else torch.tensor([])


        # 创建行业节点之间的全连接
        for i in range(len(industry_nodes)):
            for j in range(i + 1, len(industry_nodes)):
                industry_to_industry_edges.append((industry_nodes[i], industry_nodes[j]))
                industry_to_industry_edges.append((industry_nodes[j], industry_nodes[i]))


        industry_to_industry_edges = (torch.tensor([e[0] for e in industry_to_industry_edges]), torch.tensor([e[1] for e in industry_to_industry_edges]))
        print(industry_to_industry_edges)
        sys.exit()
        industry_to_daily_edges = (torch.tensor([e[0] for e in industry_to_daily_edges]), torch.tensor([e[1] for e in industry_to_daily_edges]))
        industry_to_weekly_edges = (torch.tensor([e[0] for e in industry_to_weekly_edges]), torch.tensor([e[1] for e in industry_to_weekly_edges]))
        industry_to_monthly_edges = (torch.tensor([e[0] for e in industry_to_monthly_edges]), torch.tensor([e[1] for e in industry_to_monthly_edges]))

        # 创建异构图
        hetero_graph = dgl.heterograph({
            ('industry', 'connects', 'industry'): industry_to_industry_edges,
            ('industry', 'has_daily_derivative', 'daily_derivative'): industry_to_daily_edges,
            ('industry', 'has_weekly_derivative', 'weekly_derivative'): industry_to_weekly_edges,
            ('industry', 'has_monthly_derivative', 'monthly_derivative'): industry_to_monthly_edges
        })

        # 为节点添加特征
        hetero_graph.nodes['industry'].data['features'] = industry_features
        hetero_graph.nodes['industry'].data['labels'] = industry_labels
        if daily_features.numel() > 0:
            hetero_graph.nodes['daily_derivative'].data['features'] = daily_features
        if weekly_features.numel() > 0:
            hetero_graph.nodes['weekly_derivative'].data['features'] = weekly_features
        if monthly_features.numel() > 0:
            hetero_graph.nodes['monthly_derivative'].data['features'] = monthly_features
        self.graph = hetero_graph
        self.label = industry_labels

    def __getitem__(self, i):
        return self.graph, self.label

    def __len__(self):
        return 1
    
    def calc_close_close_return(self,df,column_name):
        df['trend_return'] = df[column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values

    def calc_open_close_return(self,df,close_column_name, open_column_name):
        df['trend_return'] = df[close_column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values



parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda')
parser.add_argument('--input', action='store', type=str, help='Input file path')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--output', action='append', help='Output file path')
parser.add_argument('--outdim', type= int, default=32)
args = parser.parse_args()


# 使用示例
root_folder = './中证行业数据/'  # 替换为你的根文件夹路径
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2021-06-30')
n_days_backward = 1
window_size=20
dataset = PaddedHeteroDataset(root_folder,start_date,end_date,n_days_backward,window_size)

# 定义一个简单的collate函数
def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    # 提取图的特征，每个图的特征已经是 (num_industry, num_feature, num_of_days)
    graphs = [g for g in graphs]
    # print(len(industry_feat))
    # daily_feat = graphs.ndata['daily_derivative']
    # weekly_feat = graphs.ndata['weekly_derivative']
    # monthly_feat = graphs.ndata['monthly_derivative']
    # batched_industry_feat = torch.stack(industry_feat)  # (N, I, F, D)
    # batched_daily_featt = torch.stack(daily_feat)  # (N, I, F, D)
    # batched_weekly_feat = torch.stack(weekly_feat)  # (N, I, F, D)
    # batched_monthly_feat = torch.stack(monthly_feat)  # (N, I, F, D)
    return graphs,labels
# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

# dataloader = GraphDataLoader(dataset,batch_size=8,shuffle=True)
g, label = dataset[0]
print(g)
print(label.size())
print(g.ntypes)
# print(g.nodes['industry'].data['labels'])
# print(g.nodes['industry'].data['features'])
print('meta_paths',g.metagraph().edges())
meta_paths = []
for src_type, e_type, dst_type  in g.canonical_etypes:
    meta_path = [e_type]
    meta_paths.append(meta_path)
    # print(src_type,e_type,dst_type)

for i, [src_type, e_type, dst_type] in enumerate(g.canonical_etypes):
    print(i)
    print(src_type,e_type,dst_type)

out_dim =32
for batched_graph, label in dataloader:
    batched_graph = dgl.batch(batched_graph)
    temporal_attention = TemporalAttention(batched_graph, meta_paths, embed_dim=args.outdim,dropout=0.2)
    graph_out = temporal_attention(batched_graph)
    print(graph_out)
    # han = HAN( meta_paths=meta_paths, in_size=args["out_dim"],   hidden_size=args["hidden_units"], out_size=2,  num_heads=1, dropout=args["dropout"]).to(args["device"])
    
    # daily_out,weekly_out,monthly_out = han(g,h)
    sys.exit()


    # batch维度：(batch_num,graph_num,time_window,num_features)
    industrys = torch.stack([g.nodes['industry'].data['features'] for g in batched_graph])
    labels = torch.stack([g.nodes['industry'].data['labels'] for g in batched_graph])


    daily_tensors = torch.stack([g.nodes['daily_derivative'].data['features'] for g in batched_graph])
    print('input daily tensor shape: ',daily_tensors.size())

    weekly_tensors = torch.stack([g.nodes['weekly_derivative'].data['features'] for g in batched_graph])
    monthly_tensors = torch.stack([g.nodes['monthly_derivative'].data['features'] for g in batched_graph])

    daily_mask = torch.stack([g.nodes['daily_derivative'].data['mask'] for g in batched_graph])
    print('input daily mask shape: ',daily_mask.size())
    weekly_mask = torch.stack([g.nodes['weekly_derivative'].data['mask'] for g in batched_graph])
    monthly_mask = torch.stack([g.nodes['monthly_derivative'].data['mask'] for g in batched_graph])



    ind_output, daily_out,weekly_out,monthly_out = temporal_attention(industrys,daily_tensors,weekly_tensors,monthly_tensors, daily_mask,weekly_mask,monthly_mask)
    print('attention out daily shape',daily_out.size())
    print("节点数:", g.number_of_nodes())

    # g.nodes['industry'].data['feat'] = ind_output
    g.nodes['daily_derivative'].data['feat'] = daily_out
    g.nodes['weekly_derivative'].data['feat'] = weekly_out
    g.nodes['monthly_derivative'].data['feat'] = monthly_out

    han = HAN( num_meta_paths=len(g), in_size=args["out_dim"],   hidden_size=args["hidden_units"],
            out_size=2,            num_heads=1,            dropout=args["dropout"]        ).to(args["device"])
    daily_out,weekly_out,monthly_out = han(g,h)

    out_1 = Adaptive_Fusion(daily_out,weekly_out)
    out_2 = Adaptive_Fusion(out_1,monthly_out)

# model = HAN(
#             meta_paths=g.metagraph().edges(),
#             in_size=features.shape[1],
#             hidden_size=64,
#             out_size=1,
#             num_heads=4,
#             dropout=0.01,
#         )

class IndustryDailyDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder,start_date,end_date,n_step_ahead):
        self.root_folder = root_folder
        self.start_date = start_date
        self.end_date = end_date
        self.n_step_ahead = n_step_ahead

        industry_labels = []

        industry_nodes = []
        daily_derivative_nodes = []
        weekly_derivative_nodes = []
        monthly_derivative_nodes = []
        
        industry_features = []
        daily_features = []
        weekly_features = []
        monthly_features = []

        industry_to_industry_edges = []
        industry_to_daily_edges = []
        industry_to_weekly_edges = []
        industry_to_monthly_edges = []
        industry_to_industry_edges = []
        
        industry_node_id = 0
        current_industry_node_id = 0
        current_daily_node_id = 0
        current_weekly_node_id = 0
        current_monthly_node_id = 0
        industry_id = {}
        
        # 遍历根文件夹中的每个子文件夹,先处理每个industry node
        for directory in os.listdir( self.root_folder):
            industry_folder_path = os.path.join( self.root_folder, directory)
            if os.path.isdir(industry_folder_path):
                print(directory)
                for csv_file in os.listdir(industry_folder_path):
                    # 遍历每个子文件夹中的CSV文件
                    csv_file_path = os.path.join(industry_folder_path, csv_file)
                    # 判断是否为行业量价信息
                    if csv_file.split('.')[1]=='CSI':
                        features = []
                        industry_id[directory] = industry_node_id
                        # 处理行业量价信息
                        industry_nodes.append(industry_node_id)
                        data = pd.read_csv(csv_file_path, index_col=0,parse_dates=True)
                        data_interval = get_interval_data(data, self.start_date, self.end_date, lookback=self.n_step_ahead,lookforward=self.n_step_ahead)
                        data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围 
                        labels = self.calc_close_close_return(data_interval, csv_file[:-4]+'_close')
                        industry_labels.append(torch.tensor(labels, dtype=torch.float))
                        industry_features.append(torch.tensor(data.values, dtype=torch.float))
                        industry_node_id += 1
                        assert data.shape[0]==len(labels)
                        
            
        # 遍历根文件夹中的每个子文件夹,处理衍生行情node
        for directory in os.listdir(self.root_folder):
            industry_folder_path = os.path.join(self.root_folder, directory)
            if os.path.isfile(industry_folder_path):
                industry = directory.split('_')[0]
                frequncy = directory.split('_')[1].split('.')[0]
                if industry in industry_id:
                    current_industry_node_id = industry_id[industry]
                    # 处理衍生数据
                    data = pd.read_csv(industry_folder_path, index_col=0,parse_dates=True)
                    data = data[(data.index >=  self.start_date) & (data.index <=  self.end_date)]  # 筛选时间范围
                    if frequncy == 'daily':
                            daily_derivative_nodes.append(current_daily_node_id)
                            daily_features.append(torch.tensor(data.values, dtype=torch.float))
                            # print(data)
                            print('daily',data.shape)
                            industry_to_daily_edges.append((current_industry_node_id, current_daily_node_id))
                            current_daily_node_id += 1
                    # elif frequncy == 'weekly':
                    #     weekly_derivative_nodes.append(current_weekly_node_id)
                    #     weekly_features.append(torch.tensor(data.values, dtype=torch.float))
                    #     industry_to_weekly_edges.append((current_industry_node_id, current_weekly_node_id))
                    #     current_weekly_node_id += 1
                    # elif frequncy == 'monthly':
                    #     monthly_derivative_nodes.append(current_monthly_node_id)
                    #     monthly_features.append(torch.tensor(data.values, dtype=torch.float))
                    #     industry_to_monthly_edges.append((current_industry_node_id, current_monthly_node_id))
                    #     current_monthly_node_id += 1

        # for ele in daily_features:
        #     print('ele',ele.shape)
        industry_features = torch.stack(industry_features)
        industry_labels = torch.stack(industry_labels)
        daily_features = torch.stack(daily_features) if daily_features else torch.tensor([])
        print(industry_features.shape)
        print(daily_features.shape)
        # for i in range(daily_features.shape[0]):
        #     print(daily_features[i].shape)
        # weekly_features = torch.stack(weekly_features) if weekly_features else torch.tensor([])
        # monthly_features = torch.stack(monthly_features) if monthly_features else torch.tensor([])


        # 创建行业节点之间的全连接
        for i in range(len(industry_nodes)):
            for j in range(i + 1, len(industry_nodes)):
                industry_to_industry_edges.append((industry_nodes[i], industry_nodes[j]))
                industry_to_industry_edges.append((industry_nodes[j], industry_nodes[i]))

        for e in industry_to_industry_edges:
            print(e[0])
            sys.exit()
        industry_to_industry_edges = (torch.tensor([e[0] for e in industry_to_industry_edges]), torch.tensor([e[1] for e in industry_to_industry_edges]))
        industry_to_daily_edges = (torch.tensor([e[0] for e in industry_to_daily_edges]), torch.tensor([e[1] for e in industry_to_daily_edges]))
        industry_to_weekly_edges = (torch.tensor([e[0] for e in industry_to_weekly_edges]), torch.tensor([e[1] for e in industry_to_weekly_edges]))
        industry_to_monthly_edges = (torch.tensor([e[0] for e in industry_to_monthly_edges]), torch.tensor([e[1] for e in industry_to_monthly_edges]))

        # 创建异构图
        hetero_graph = dgl.heterograph({
            ('industry', 'connects', 'industry'): industry_to_industry_edges,
            ('industry', 'has_daily_derivative', 'daily_derivative'): industry_to_daily_edges,
            ('industry', 'has_weekly_derivative', 'weekly_derivative'): industry_to_weekly_edges,
            ('industry', 'has_monthly_derivative', 'monthly_derivative'): industry_to_monthly_edges
        })

        # 为节点添加特征
        hetero_graph.nodes['industry'].data['features'] = industry_features
        hetero_graph.nodes['industry'].data['labels'] = industry_labels
        if daily_features.numel() > 0:
            hetero_graph.nodes['daily_derivative'].data['features'] = daily_features
        if weekly_features.numel() > 0:
            hetero_graph.nodes['weekly_derivative'].data['features'] = weekly_features
        if monthly_features.numel() > 0:
            hetero_graph.nodes['monthly_derivative'].data['features'] = monthly_features
        self.graph = hetero_graph
        self.label = industry_labels

    def __getitem__(self, index):
        # feature = self.features[index:index+self._his_window, :] # .values
        # if self.labels is not None:
        #     label = self.labels[index+self._his_window-1]
        #     return feature, label
        # else:
        #     return feature
        feature = self.features[index] # .values
        if self.labels is not None:
            label = self.labels[index]
            return feature, label
        else:
            return feature


    def __len__(self):
        return len(self.features)
    
    def calc_close_close_return(self,df,column_name):
        df['trend_return'] = df[column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values

    def calc_open_close_return(self,df,close_column_name, open_column_name):
        df['trend_return'] = df[close_column_name].pct_change(periods=self.n_step_ahead) # n_step_ahead=5 is a week
        df['trend_return'] = df['trend_return'].shift(-self.n_step_ahead)
        return df['trend_return'].iloc[self.n_step_ahead:-self.n_step_ahead].values
    

# # 使用示例
# root_folder = './中证行业数据'  # 替换为你的根文件夹路径
# start_date = pd.to_datetime('2020-01-01')
# end_date = pd.to_datetime('2021-06-30')
# n_days_backward = 1
# dataset = IndustryDailyDataset(root_folder,start_date,end_date,n_days_backward)
