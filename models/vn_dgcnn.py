import torch
from torch import nn

from models.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool, mean_pool
import numpy as np

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

class VNDgcnn(nn.Module):
    def __init__(self, k, in_channel, mlps):
        super(VNDgcnn, self).__init__()
        self.n_knn = k
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel*2
        for out_channel in mlps[:-1]:
            self.mlp_convs.append(VNLinearLeakyReLU(last_channel, out_channel))
            last_channel = out_channel*2

        self.conv_last = VNLinearLeakyReLU(np.sum(mlps[:-1])+in_channel, mlps[-1], dim=4)
        self.pool = mean_pool

    def forward(self, x):
        x_gnn = [x]
        for i, conv in enumerate(self.mlp_convs):
            x_ = get_graph_feature(x_gnn[-1], k=self.n_knn)
            x_ = conv(x_)
            x_ = self.pool(x_)
            x_gnn.append(x_)
        x_ = torch.cat(x_gnn, dim=1)
        x_= self.conv_last(x_)

        return x_
