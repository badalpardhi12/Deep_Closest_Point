import torch
import torch.nn as nn

class GraphAttention(nn.Module):
    def __init__(self, in_dimesion) -> None:
        super(GraphAttention, self).__init__()
        self.weights = nn.Linear(2*in_dimesion, 1)
        self.act = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):  #B, C, N, K  B:batch_size, C:feature, N:num points, K: neighbors
        k = x.shape[3]
        x_reshaped = x.permute((0, 2, 1, 3)) #B, N, C, K
        center_point_feature = x_reshaped[:, :, :, 0].unsqueeze(-1) #B, N, C, 1
        center_point_feature = center_point_feature.repeat((1, 1, 1, k)) #B, N, C, K
        concat_features = torch.cat([x_reshaped, center_point_feature], dim=2) #B, N, 2*C, K
        concat_features = concat_features.permute((0, 1, 3, 2)) #B, N, K, 2*C 
        concat_features_flatten = torch.reshape(concat_features, (-1, concat_features.shape[-1])) #B*N*K, 2*C
        attention_weights = self.weights(concat_features_flatten) #B*N*K, 1
        attention_weights = torch.reshape(attention_weights, (concat_features.shape[0],
                                                              concat_features.shape[1],
                                                              1, -1)) #B*N*1*K
        attention_weights = self.act(attention_weights) #B*N*1*K
        attention_weights = self.softmax(attention_weights) #B*N*1*K
        out = x_reshaped * attention_weights #B, N, C, K
        out = torch.sum(out, dim=-1, keepdim=True)
        out = out.permute(0, 2, 1, 3) # B, C, N
        return out     

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dimension, no_of_heads=4):
        super(MultiHeadGraphAttention, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(no_of_heads):
            self.layers.append(GraphAttention(in_dimension))
        
    def forward(self, x):
        outputs = torch.empty((x.shape[0], x.shape[1], x.shape[2], 0)).cuda()
        for layer in self.layers:
            curr_out = layer(x)
            outputs = torch.cat([outputs, curr_out], dim=-1)
        outputs = torch.mean(outputs, dim=-1, keepdim=True)
        return outputs
        
if __name__ == '__main__':
    pass
    # x = torch.randn((3, 3, 1024, 20))
    # model = MultiHeadGraphAttention(3, 3)
    # out = model(x)
    # print(out.shape)