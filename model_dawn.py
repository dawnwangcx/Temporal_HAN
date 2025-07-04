"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        print('z:',z.shape())
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        print('out:',(beta * z).sum(1).shape())

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size:list, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = meta_paths

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g):
        semantic_embeddings = []
        print('g:',g)
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                print('meta path:',meta_path[0])
                print( dgl.metapath_reachable_graph(g, meta_path))
                self._cached_coalesced_graph[meta_path[0]] = dgl.metapath_reachable_graph(g, meta_path)

        for i, [src_type, e_type, dst_type] in enumerate(g.canonical_etypes):
            new_g = self._cached_coalesced_graph[e_type]
            h = tuple(new_g[src_type],new_g[dst_type])
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        print(semantic_embeddings.shape())

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g):
        for gnn in self.layers:
            h = gnn(g)

        return self.predict(h)

class NodeEncoder(nn.Module):
    def __init__(self, graph, embed_dim=32):
        super().__init__()
        
        self.encoders = nn.ModuleDict({
            ntype: nn.Linear(graph.nodes[ntype].data['features'].shape[-1], embed_dim) 
            for ntype in graph.ntypes
        })
        self.ln = nn.LayerNorm(embed_dim)

        
    def forward(self, graph):
        for ntype in graph.ntypes:
            graph.nodes[ntype].data['h'] = self.ln(self.encoders[ntype](graph.nodes[ntype].data['features']))
        return graph

class HeteroTempAttention(nn.Module):
    # 不同节点进行交叉计算
    def __init__(self, graph, embed_dim=32, num_heads=4):
        super().__init__()

        self.edge_types = [f"{stype}_{connect}_{dtype}" 
                          for stype, connect, dtype in graph.canonical_etypes]
        
        # 仅创建实际存在的边类型
        self.attentions = nn.ModuleDict({
            etype: nn.MultiheadAttention(embed_dim, num_heads)
            for etype in self.edge_types
        })
        
    def forward(self, graph):
        for stype, connect, dtype in graph.canonical_etypes:
            etype = f"{stype}_{connect}_{dtype}"  # 与初始化键名一致
            src_feat = graph.nodes[stype].data['h']
            dst_feat = graph.nodes[dtype].data['h']
            print(etype)
            print(src_feat.size())
            print(dst_feat.size())
            # 维度适配：(num_nodes, embed_dim) -> (seq_len, batch, embed_dim)
            # attn_out, _ = self.attentions[etype](
            #     src_feat.unsqueeze(1),  # 添加序列维度
            #     dst_feat.unsqueeze(1),
            #     dst_feat.unsqueeze(1)
            # )
            attn_out, _ = self.attentions[etype](src_feat, dst_feat, dst_feat)
            graph.nodes[dtype].data['h'] += attn_out
        return graph
    
class SpatialAttention(nn.Module):
    # 每个节点自己时序计算
    def __init__(self, graph, embed_dim=32, num_heads=4):
        super().__init__()        
        self.attentions = nn.ModuleDict({
            ntype: nn.MultiheadAttention(embed_dim, num_heads)
            for ntype in graph.ntypes
        })

    def forward(self, graph):
        for ntype in graph.ntypes:
            feat = graph.nodes[ntype].data['h']
            attn_out, _ = self.attentions[ntype](feat, feat, feat)
            graph.nodes[ntype].data['h'] += attn_out
        return graph


class TemporalAttention(torch.nn.Module):
    def __init__(self, graph, meta_paths, out_size=2, embed_dim=32,hidden_size=32,num_heads=[4],dropout=0.2):
        super().__init__()
        self.encoder = NodeEncoder(graph)
        self.spatial_attn = SpatialAttention(graph,embed_dim,num_heads[0])
        # 初始化 HAN 层
        self.han_layer = HAN(meta_paths, embed_dim, hidden_size,out_size, num_heads, dropout)
        # 输出层
        # self.output_layer = nn.Linear(hidden_size * num_heads, out_size)
        # self.fusion = FusionLayer()
        
    def forward(self, graph):
        graph = self.encoder(graph)
        graph = self.spatial_attn(graph)
        out = self.han_layer(graph)
        # out = self.output_layer(h)

        # graph = self.attn_layer2(graph)
        # graph = self.fusion(graph)
        return out

# class TemporalAttention(torch.nn.Module):
#     def __init__(self, ind_shape, daily_shape, weekly_shape, monthly_shape, out_dim = 32, nhid=8, dropout=0.6, alpha=0.2, nheads=4):
#         super(TemporalAttention, self).__init__()

#         self.industry_linear = nn.Linear(ind_shape, out_dim)
#         self.daily_linear = nn.Linear(daily_shape,out_dim)
#         self.weekly_linear = nn.Linear(weekly_shape,out_dim)
#         self.monthly_linear = nn.Linear(monthly_shape,out_dim)

#         self.industry_attention = torch.nn.MultiheadAttention(embed_dim=out_dim, num_heads=nhid)
#         self.daily_attention = torch.nn.MultiheadAttention(embed_dim=out_dim, num_heads=nhid)
#         self.weekly_attention = torch.nn.MultiheadAttention(embed_dim=out_dim, num_heads=nhid)
#         self.monthly_attention = torch.nn.MultiheadAttention(embed_dim=out_dim, num_heads=nhid)

        
#         self.dropout = dropout


#     def forward(self,industry, daily, weekly, monthly, daily_mask=None,weekly_mask=None,monthly_mask=None):
#         print('input daily tensor shape: ',daily.shape)

#         industry = self.industry_linear(industry)
#         daily = self.daily_linear(daily)
#         weekly = self.weekly_linear(weekly)
#         monthly = self.monthly_linear(monthly)
#         print('linear out daily tensor shape: ',daily.shape)

#         ind_batch_size, ind_num_graphs, ind_window_size, ind_num_features = industry.shape
#         daily_batch_size, daily_num_graphs, daily_window_size, daily_num_features = daily.shape
#         weekly_batch_size, weekly_num_graphs, weekly_window_size, weekly_num_features = weekly.shape
#         mothly_batch_size, mothly_num_graphs, mothly_weekly_window_size, mothly_num_features = monthly.shape

#        # 调整维度以适配 MultiheadAttention 的输入要求
#         industry = industry.permute(2, 0, 1, 3).reshape(ind_window_size, ind_batch_size*ind_num_graphs, ind_num_features)
#         daily = daily.permute(2, 0, 1, 3).reshape(daily_window_size, daily_batch_size*daily_num_graphs, daily_num_features)
#         weekly = weekly.permute(2, 0, 1, 3).reshape(weekly_window_size, weekly_batch_size*weekly_num_graphs,weekly_num_features)
#         monthly = monthly.permute(2, 0, 1, 3).reshape(mothly_weekly_window_size, mothly_batch_size*mothly_num_graphs, mothly_num_features)
#         # 将遮罩调整为 (batch_size * num_stocks, window_size) 的形状
#         # daily_mask = daily_mask.view(daily_mask.shape[0] * daily_mask.shape[0], daily_mask.shape[0])
#         # weekly_mask = weekly_mask.view(weekly_mask.shape[0] * weekly_mask.shape[0], weekly_mask.shape[0])
#         # monthly_mask = monthly_mask.view(monthly_mask.shape[0] * monthly_mask.shape[0], monthly_mask.shape[0])
#         # stacked_tensors.permute(1, 0, 2), mask.permute(1, 0, 2)

#         ind_attn_output, _ = self.industry_attention(industry, industry, industry)
#         daily_attn_output, _ = self.daily_attention(daily, daily, daily)
#         daily_attn_output = F.dropout(daily_attn_output, self.dropout, training=self.training)
#         weekly_attn_output, _ = self.weekly_attention(weekly, weekly, weekly)
#         monthly_attn_output, _ = self.monthly_attention(monthly, monthly, monthly)

#         print('daily_attn_output raw shape: ',daily_attn_output.size())

#         # 调整输出维度
#         ind_output     = ind_attn_output.view(ind_window_size, ind_batch_size, ind_num_graphs, -1).permute(1, 2, 0, 3)
#         daily_output   = daily_attn_output.view(daily_window_size, daily_batch_size, daily_num_graphs, -1).permute(1, 2, 0, 3)
#         weekly_output  = weekly_attn_output.view(weekly_window_size, weekly_batch_size, weekly_num_graphs, -1).permute(1, 2, 0, 3)
#         monthly_output = monthly_attn_output.view(mothly_weekly_window_size, mothly_batch_size, mothly_num_graphs, -1).permute(1, 2, 0, 3)
#         print('daily_output shappe: ',daily_output.shape)

#         ind_output     = ind_output[:, :, -1, :]
#         daily_output   = daily_output[:, :, -1, :]
#         weekly_output  = weekly_output[:, :, -1, :]
#         monthly_output = monthly_output[:, :, -1, :]

#         print('daily_output shappe: ',daily_output.shape)

#         return ind_output, daily_output, weekly_output, monthly_output

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qlfc = FeedForward([features,features])
        self.khfc = FeedForward([features,features])
        self.vhfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, xl, xh, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        return: [B,T,N,F]
        '''

        query = self.qlfc(xl) # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh)) # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh)) # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d,-1), 0).permute(0,2,1,3) # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d,-1), 0).permute(0,2,3,1) # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d,-1), 0).permute(0,2,1,3) # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh) # [k*B,N,T,T]
        
        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attentionh).to(xl.device) # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        
        attentionh /= (self.d ** 0.5) # scaled
        attentionh = F.softmax(attentionh, -1) # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh) # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0]//self.h, 0), -1).permute(0,2,1,3) # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)