from audioop import bias
from turtle import forward
from requests import ReadTimeout
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points
# import pointnet2
from pointnet2_ops.pointnet2_utils import grouping_operation
import time
from attention import GraphAttention, MultiHeadGraphAttention

device = torch.device('cuda')
def find_neighbors(input_points=None, K=20):
    #input points dim B*N*D
    start_time = time.time()
    if len(input_points.shape) == 4:
        input_points = input_points.squeeze(-1)
    input_points = input_points.permute((0, 2, 1))
    _, neighbor_indices, _ = knn_points(input_points, input_points, K=K)
    return neighbor_indices

def knn(x, k):
    start_time = time.time()
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def generate_dilated_neighbors(neighbor_indices, dilation=4):
    return neighbor_indices[:, :, ::dilation]


def get_graph_feature(x, idx, K=20, dilation=1):
    # x = x.squeeze()
    #idx = find_neighbors(x, k=k)  # (batch_size, num_points, k)
    k = K // dilation
    idx = idx[:, :, ::dilation]
    batch_size, num_points, _ = idx.size()
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base 
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature


class ConvBN(nn.Module):
    def __init__(self, in_features: int, out_features: int, agg_fn=None) -> None:
        super(ConvBN, self).__init__()
        self.agg_fn = agg_fn
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = 1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(3)
        if self.agg_fn:
            out = self.act(self.agg_fn(self.bn(self.conv(x))))
        else:
            out = self.act(self.bn(self.conv(x)))
        return out

class ConvBNList(nn.Module):
    def __init__(self, conv_list: list, agg_fn, out_feat):
        super(ConvBNList, self).__init__()
        self.agg_fn = agg_fn
        self.layers = nn.ModuleList()
        
        if len(conv_list) > 1:
            for i in range(len(conv_list)-1):
                self.layers.append(nn.Sequential(nn.Conv2d(conv_list[i], conv_list[i+1], kernel_size=1, bias=False),
                nn.BatchNorm2d(conv_list[i+1])))
            sum_features = sum(conv_list) - conv_list[0]
            self.conv = nn.Conv2d(sum_features, out_feat, kernel_size = 1, bias=False)
        else:
            self.conv = nn.Conv2d(conv_list[0], out_feat, kernel_size = 1, bias=False)
        
        self.act = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(out_feat)
   
    def forward(self, x):
        # define your feedforward pass
        # layers_temp = self.layers
        output_list = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            # Do RELU for x, call agg_fn for x
            x =self.act(x)
            self.agg_fn(x)
            output_list.append(x)
        if len(output_list) > 0:
            x = torch.cat(output_list, dim=1) # should return 256 channels
        x = self.agg_fn(self.act(self.bn(self.conv(x))))
        return x

class DenseGCN(nn.Module):
    def __init__(self, 
                input_features, 
                output_features, 
                agg_fun_name='max'
                ):
        super(DenseGCN, self).__init__()
        if agg_fun_name == 'max':
            self.agg_fn = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        elif agg_fun_name == 'min':
            self.agg_fn = lambda x: torch.min(x, dim=-1, keepdim=True)[0]
        elif agg_fun_name == 'sum':
            self.agg_fn = lambda x: torch.sum(x, dim=-1, keepdim=True)
        elif agg_fun_name == "none":
            self.agg_fn = lambda x: x
        elif agg_fun_name == 'attention':
            print("Using Attention")
            self.agg_fn = MultiHeadGraphAttention(output_features, 3) 
        self.conv_bn = ConvBNList([input_features], self.agg_fn, output_features)  

    def forward(self, feature):
        out = self.conv_bn(feature)
        return out

class InceptionDenseGCN(nn.Module):
    def __init__(self, 
                input_features, 
                out_features, 
                dilation=[1, 1], 
                k=20):
        super(InceptionDenseGCN, self).__init__()
        self.k = k
        self.dilation = dilation
        self.find_neighbors = find_neighbors
        self.build_graph = get_graph_feature

        self.bottle_neck = ConvBN(input_features, input_features//4) # 128, 32; conv_bn(input_, input//4)
        self.GCN1 = DenseGCN(input_features//2, input_features//2, 'attention')
        self.GCN2 = DenseGCN(input_features//2, input_features//2, 'attention')
        self.global_pooling = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.decoder = ConvBN(3*input_features//2, out_features) #conv_bn(out_feat*3//4, out_feat)
        self.residual =ConvBN(input_features, out_features) # conv_bn(in_feat, out_feat)
        
    def forward(self, x):
        residual_out = self.residual(x)
        idx = self.find_neighbors(x, K=self.k) #B, N, K 
        out_bn1 = self.bottle_neck(x).squeeze(-1) #B, C//4, N, 1

        features = self.build_graph(out_bn1, idx, self.k, 1) # B, C//4, N, K
        features_2 = self.build_graph(out_bn1, idx, self.k, self.dilation[1])

        out_gcn1 = self.GCN1(features)  #B, C//4, N, 1 
        out_gcn2 = self.GCN2(features_2) #B, C//4, N, 1
        global_max = self.global_pooling(features) #B, C//4, N, 1

        all_features = torch.cat([out_gcn1, out_gcn2, global_max], dim=1)
        out_features = self.decoder(all_features)
        out = out_features + residual_out
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, 
                features_dim = [3, 128, 256], 
                k=20, 
                dilation=[1, 2]
                ):
        super(FeatureExtractor, self).__init__()
        self.k = k
        self.dilation = dilation
        self.layers = nn.ModuleList()
        self.find_neighbors = find_neighbors
        self.build_graph = get_graph_feature

        for i in range(len(features_dim)-1):
            if i == 0:
                self.layers.append(DenseGCN(2*features_dim[i], features_dim[i+1], 'attention'))
            else:
                self.layers.append(InceptionDenseGCN(features_dim[i], features_dim[i+1], self.dilation))

    def forward(self, x):
        idx = find_neighbors(x, self.k)
        initial_features = self.build_graph(x, idx, self.k, 1) # B, C, N, K
        out = None
        total_outs = []

        for i in range(len(self.layers)):
            if i == 0:
                out = self.layers[i](initial_features.to(device))
            else:
                out = self.layers[i](out) # B, C, N, 1
                total_outs.append(out)

        final_outs = torch.cat(total_outs, dim=1).squeeze(-1)
        return final_outs

if __name__=='__main__':
    # x = torch.randn((128, 3, 1024))
    # find_neighbors(x)
    # knn(x, 20)
    pass
