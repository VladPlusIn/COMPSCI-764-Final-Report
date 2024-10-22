import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np


def weight_init(layers):
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            n = layer.in_features
            y = 1.0 / np.sqrt(n)
            layer.weight.data.uniform_(-y, y)
            layer.bias.data.fill_(0)
            # nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')


class DeepFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(DeepFM, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        # nn.init.xavier_normal_(self.linear.weight)
        # self.bias = nn.Parameter(torch.zeros((output_dim,)))

        # FM embedding
        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        # MLP
        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers) 

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """

        linear_x = x

        second_x = self.feature_embedding(x)

        square_of_sum = torch.sum(second_x, dim=1) ** 2
        sum_of_square = torch.sum(second_x ** 2, dim=1)

        ix = torch.sum(square_of_sum - sum_of_square, dim=1,
                       keepdim=True) 

        out = torch.sum(self.linear(linear_x), dim=1) + ix * 0.5 + self.mlp(
            second_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)


class FNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims):
        super(FNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

    def load_embedding(self, pretrain_params):
        self.feature_embedding.weight.data.copy_(
            torch.from_numpy(
                np.array(pretrain_params['feature_embedding.weight'].cpu())
            )
        )

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x)
        out = self.mlp(embedding_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)


class DCN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(DCN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims

        deep_net_layers = list()
        neural_nums = [300, 300, 300]
        self.num_neural_layers = 5

        for neural_num in neural_nums:
            deep_net_layers.append(nn.Linear(deep_input_dims, neural_num))
            # deep_net_layers.append(nn.BatchNorm1d(neural_num))
            deep_net_layers.append(nn.ReLU())
            deep_net_layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neural_num

        weight_init(deep_net_layers)

        self.DN = nn.Sequential(*deep_net_layers)

        cross_input_dims = self.field_nums * self.latent_dims
        self.cross_net_w = nn.ModuleList([
            nn.Linear(cross_input_dims, output_dim) for _ in range(self.num_neural_layers)
        ])

        weight_init(self.cross_net_w)

        self.cross_net_b = nn.ParameterList([
            nn.Parameter(torch.zeros((cross_input_dims,))) for _ in range(self.num_neural_layers)
        ])

        self.linear = nn.Linear(neural_nums[-1] + self.field_nums * self.latent_dims, output_dim)
        # nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        embedding_x = self.feature_embedding(x).view(-1, self.field_nums * self.latent_dims)

        cn_x0, cn_x = embedding_x, embedding_x
        for i in range(self.num_neural_layers):
            cn_x_w = self.cross_net_w[i](cn_x)
            cn_x = cn_x0 * cn_x_w + self.cross_net_b[i] + cn_x
        dn_x = self.DN(embedding_x)
        x_stack = torch.cat([cn_x, dn_x], dim=1)

        out = self.linear(x_stack)

        return torch.sigmoid(out)


class AFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(AFM, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        self.row, self.col = list(), list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                self.row.append(i), self.col.append(j)

        attention_factor = self.latent_dims

        self.attention_net = nn.Linear(self.latent_dims, attention_factor) 
        n = self.attention_net.in_features
        y = 1.0 / np.sqrt(n)
        self.attention_net.weight.data.uniform_(-y, y)
        self.attention_net.bias.data.fill_(0)

        self.attention_softmax = nn.Linear(attention_factor, 1)

        self.fc = nn.Linear(self.latent_dims, output_dim)

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        embedding_x = self.feature_embedding(x)

        inner_product = torch.mul(embedding_x[:, self.row], embedding_x[:, self.col])

        attn_scores = F.relu(self.attention_net(inner_product))
        attn_scores = F.softmax(self.attention_softmax(attn_scores), dim=1)

        attn_scores = F.dropout(attn_scores, p=0.2)
        attn_output = torch.sum(torch.mul(attn_scores, inner_product), dim=1)  # shape: batch_size-latent_dims
        attn_output = F.dropout(attn_output, p=0.2)

        out = self.bias + torch.sum(self.linear(x), dim=1) + self.fc(attn_output)

        return torch.sigmoid(out)