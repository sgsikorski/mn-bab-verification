import gzip
import typing
from os.path import exists

from typing import List, Optional, Sequence, Tuple, Type

import onnx  # type: ignore[import]
import torch
import torch.nn.functional as F
from bunch import Bunch  # type: ignore[import]
import numpy as np

from torch import nn as nn

from src.concrete_layers.basic_block import BasicBlock
from src.concrete_layers.pad import Pad
from src.utilities.onnx_loader import ConvertModel

import configparser
import math

def lecture_network_small() -> nn.Sequential:
    net = nn.Sequential(
        *[
            nn.Linear(in_features=2, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1),
        ]
    )
    net[0].weight.data = torch.ones_like(net[0].weight.data)
    net[0].weight.data[1, 1] = -1.
    net[0].bias.data = torch.zeros_like(net[0].bias.data)

    net[2].weight.data = torch.ones_like(net[2].weight.data)
    net[2].bias.data[0] = -0.5
    return net


def lecture_network() -> nn.Sequential:
    net = nn.Sequential(
        *[
            nn.Linear(in_features=2, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=2)
        ]
    )

    net[0].weight.data = torch.ones_like(net[0].weight.data)
    net[0].weight.data[1, 1] = -1.
    net[0].bias.data = torch.zeros_like(net[0].bias.data)

    net[2].weight.data = torch.ones_like(net[2].weight.data)
    net[2].weight.data[1, 1] = -1.
    net[2].bias.data = torch.tensor([-0.5, 0])

    net[4].weight.data = torch.ones_like(net[4].weight.data)
    net[4].weight.data[0, 0] = -1.
    net[4].weight.data[1, 0] = 0
    net[4].bias.data = torch.tensor([3., 0])
    return net

def mnist_conv_tiny() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=8 * 2 * 2, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
        ]
    )


def mnist_conv_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 5 * 5, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def mnist_conv_sigmoid_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 5 * 5, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )


def mnist_conv_big() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def mnist_conv_super() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 18 * 18, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def mnist_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Linear(28 * 28, b), nn.ReLU()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.ReLU(),
        ]
    layers += [nn.Linear(b, 10), nn.ReLU()]
    return nn.Sequential(*layers)


def mnist_sig_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Linear(28 * 28, b), nn.Sigmoid()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.Sigmoid(),
        ]
    layers += [nn.Linear(b, 10), nn.Sigmoid()]
    return nn.Sequential(*layers)


def mnist_vnncomp_a_b(a: int, b: int) -> nn.Sequential:
    layers = [nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(28 * 28, b), nn.ReLU()]
    for __ in range(a - 1):
        layers += [
            nn.Linear(b, b),
            nn.ReLU(),
        ]
    layers += [nn.Linear(b, 10)]
    return nn.Sequential(*layers)


def cifar10_conv_small() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 6 * 6, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )

class AglStan(nn.Sequential):
    def __init__(self, nodeEmbeddings, nodeWeights=None, nodeBias=None):
        self.layers: List[nn.Module] = []
        config = configparser.ConfigParser()
        config.read(f'../AGLSTAN/data/chi/config.conf')

        self.node_num = int(config['data']['num_nodes'])
        self.dim_in = int(config['model']['input_dim'])
        self.dim_out = int(config['model']['output_dim'])
        self.hidden_size = self.node_num * self.dim_out
        self.cheb_k = int(config['model']['cheb_k'])
        self.embed_dim = int(config['model']['embed_dim'])
        self.num_layers = int(config['model']['num_layers'])
        self.filter_size = int(config['model']['filter_size'])
        self.batch_size = int(config['train']['batch_size'])
        self.window = int(config['data']['window'])
        self.horizon = int(config['data']['horizon'])
        self.head_size = 6
        self.att_size = self.hidden_size // self.head_size
        self.scale = self.att_size ** -0.5

        self.node_embeddings = nodeEmbeddings
        self.weights_pool = nodeWeights
        self.bias_pool = nodeBias

        # AGL Layers
        self.gconv_layers = [
            [nn.Linear(self.node_num, self.node_num)] 
            for _ in range(self.num_layers)
        ]

        # TSA Layers
        self.encoders = [
            # q, k, v, output, FFN
            [nn.Linear(self.hidden_size, self.head_size * self.att_size, bias=False),
            nn.Linear(self.hidden_size, self.head_size * self.att_size, bias=False),
            nn.Linear(self.hidden_size, self.head_size * self.att_size, bias=False),
            
            nn.Linear(self.head_size * self.att_size, self.hidden_size),
            
            nn.Linear(self.hidden_size, self.filter_size), 
            nn.ReLU(), 
            nn.Linear(self.filter_size, self.hidden_size)]
            for _ in range(self.num_layers)
        ]

        self.endConv = [nn.Conv2d(self.window, self.horizon, padding=(2, 2), kernel_size=(5, 5), bias=False)]

        self.layers += [nn.Linear(8, 77, bias=False)]
        for layer in range(self.num_layers):
            self.layers += self.gconv_layers[layer] + [nn.Linear(self.node_num, self.hidden_size)] + self.encoders[layer][2:] + [nn.Linear(self.hidden_size, self.node_num)]
        self.layers += self.endConv
        super(AglStan, self).__init__(*self.layers)

    def forward(self, inputs):
        out = inputs.view(1, 8, 77, 8)
        for i, (g, e) in enumerate(zip(self.gconv_layers, self.encoders)):
            # AGL Layer Forward
            batch_size = out.shape[0]
            seq_len = out.shape[1]
            x = out if out.shape[2]!=self.node_num else out.view(batch_size, seq_len, self.node_num, -1)
            out_lst = []
            for t in range(seq_len):
                support = F.softmax(F.relu(self.gconv_layers[i][0](torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)))), dim=1)
                support_set = [torch.eye(self.node_num).to('cpu'), support]
                for _  in range(2, self.cheb_k):
                    support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2])
                support = torch.stack(support_set, dim=0)
                
                w = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool[i])
                b = torch.matmul(self.node_embeddings[i], self.bias_pool[i])

                y = torch.einsum("knm,bmc->bknc", support, x[:, t, :, :])
                y = y.permute(0, 2, 1, 3)
                y = torch.einsum("bnki,nkio->bno", y, w) + b
                out_lst.append(y)

            out = torch.stack(out_lst, dim=1).view(batch_size, seq_len, -1)
            
            # TSA Forward
            q = out
            k = out
            v = out
            d_k = self.att_size
            d_v = self.att_size

            batch_size = q.size(0)
            q = self.encoders[i][0](q).view(batch_size, -1, self.head_size, d_k)
            k = self.encoders[i][1](k).view(batch_size, -1, self.head_size, d_k)
            v = self.encoders[i][2](v).view(batch_size, -1, self.head_size, d_v)

            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            k = k.transpose(1, 2).transpose(2, 3)

            q.mul_(self.scale)
            x = torch.matmul(q, k) 
            x = torch.softmax(x, dim=3)
            x = x.matmul(v)

            x = x.transpose(1, 2).contiguous()
            x = x.view(batch_size, -1, self.head_size * d_v)

            x = self.encoders[i][3](x)
            out = out + x

            y = self.encoders[i][4](out)
            y = self.encoders[i][5](y)
            y = self.encoders[i][6](y)
            out = out + y
            out = out.view(1, 8, 77, 8)
        # out = out.view(self.batch_size, self.window, self.node_num, -1)
        out = self.endConv[0](out)
        return out

def convertStateDict(original_network, state_dict, best_state_dict):
    # Assign state_dict values to cooresponding original_network layers
    state_dict['1.weight'] = best_state_dict['encoder.gconv_layers.0.gconv_layer.linear.weight']
    state_dict['1.bias'] = best_state_dict['encoder.gconv_layers.0.gconv_layer.linear.bias']
    #state_dict['1.weight'] = best_state_dict['encoder.encoders.0.self_attention.linear_q.weight']
    #state_dict['3.weight'] = best_state_dict['encoder.encoders.0.self_attention.linear_k.weight']
    state_dict['3.weight'] = best_state_dict['encoder.encoders.0.self_attention.linear_v.weight']
    state_dict['4.weight'] = best_state_dict['encoder.encoders.0.self_attention.output_layer.weight']

    state_dict['5.weight'] = best_state_dict['encoder.encoders.0.ffn.layer.0.weight']
    state_dict['5.bias'] = best_state_dict['encoder.encoders.0.ffn.layer.0.bias']
    # ReLU
    state_dict['7.weight'] = best_state_dict['encoder.encoders.0.ffn.layer.2.weight']
    state_dict['7.bias'] = best_state_dict['encoder.encoders.0.ffn.layer.2.bias']


    state_dict['9.weight'] = best_state_dict['encoder.gconv_layers.1.gconv_layer.linear.weight']
    state_dict['9.bias'] = best_state_dict['encoder.gconv_layers.1.gconv_layer.linear.bias']
    #state_dict['9.weight'] = best_state_dict['encoder.encoders.1.self_attention.linear_q.weight']
    #state_dict['10.weight'] = best_state_dict['encoder.encoders.1.self_attention.linear_k.weight']
    state_dict['11.weight'] = best_state_dict['encoder.encoders.1.self_attention.linear_v.weight']
    state_dict['12.weight'] = best_state_dict['encoder.encoders.0.self_attention.output_layer.weight']
    
    state_dict['13.weight'] = best_state_dict['encoder.encoders.1.ffn.layer.0.weight']
    state_dict['13.bias'] = best_state_dict['encoder.encoders.1.ffn.layer.0.bias']

    state_dict['15.weight'] = best_state_dict['encoder.encoders.1.ffn.layer.2.weight']
    state_dict['15.bias'] = best_state_dict['encoder.encoders.1.ffn.layer.2.bias']


    state_dict['17.weight'] = best_state_dict['encoder.gconv_layers.2.gconv_layer.linear.weight']
    state_dict['17.bias'] = best_state_dict['encoder.gconv_layers.2.gconv_layer.linear.bias']
    #state_dict['17.weight'] = best_state_dict['encoder.encoders.2.self_attention.linear_q.weight']
    #state_dict['18.weight'] = best_state_dict['encoder.encoders.2.self_attention.linear_k.weight']
    state_dict['19.weight'] = best_state_dict['encoder.encoders.2.self_attention.linear_v.weight']
    state_dict['20.weight'] = best_state_dict['encoder.encoders.0.self_attention.output_layer.weight']
    
    state_dict['21.weight'] = best_state_dict['encoder.encoders.2.ffn.layer.0.weight']
    state_dict['21.bias'] = best_state_dict['encoder.encoders.2.ffn.layer.0.bias']

    state_dict['23.weight'] = best_state_dict['encoder.encoders.2.ffn.layer.2.weight']
    state_dict['23.bias'] = best_state_dict['encoder.encoders.2.ffn.layer.2.bias']

    state_dict['25.weight'] = best_state_dict['end_conv.weight']
    return state_dict

def agl_stan() -> nn.Sequential:
    layers = []

    config = configparser.ConfigParser()
    config.read(f'../AGLSTAN/data/chi/config.conf')
    # model = AGLSTAN.AGLSTAN(config)

    node_num = int(config['data']['num_nodes'])
    dim_in = int(config['model']['input_dim'])
    dim_out = int(config['model']['output_dim'])
    hidden_size = node_num * dim_out
    cheb_k = int(config['model']['cheb_k'])
    embed_dim = int(config['model']['embed_dim'])
    num_layers = int(config['model']['num_layers'])
    filter_size = int(config['model']['filter_size'])
    # batch_size = int(config['train']['batch_size'])
    window = int(config['data']['window'])
    head_size = 6
    att_size = hidden_size // head_size
    scale = att_size ** -0.5

    layers.append(nn.Linear(window, node_num, bias=True))
    # AGL Layer
    for _ in range(num_layers):
        # encoder.gconv_layers.0.gconv_layer.weights_pool
        # encoder.gconv_layers.0.gconv_layer.bias_pool
        layers += [
                nn.Linear(node_num, node_num, bias=True), 
                nn.ReLU(), 
                nn.Linear(node_num, hidden_size, bias=True)
        ]

        # TSA Layer
        layers += [
            #nn.BatchNorm2d(hidden_size, eps=1e-6),
            # Multihead attention
            nn.Linear(hidden_size, head_size * att_size, bias=True),
            nn.ReLU(),
            #nn.Linear(hidden_size, head_size * att_size, bias=False),
            #nn.Linear(hidden_size, head_size * att_size, bias=False),
            nn.Linear(head_size * att_size, hidden_size, bias=True),

            #nn.BatchNorm2d(hidden_size, eps=1e-6),

            # Feed forward network
            nn.Linear(hidden_size, filter_size, bias=True), 
            nn.ReLU(), 
            nn.Linear(filter_size, hidden_size, bias=True),
            nn.Linear(hidden_size, node_num, bias=True),
        ]
    layers += [nn.ReLU(), nn.Linear(node_num, 8)]
    #layers += [nn.BatchNorm2d(hidden_size, eps=1e-6)]
    # _.view(batch_size, window, node_num, -1)
    # layers += [nn.Conv2d(int(config['data']['window']), int(config['data']['horizon']), padding=(2, 2), kernel_size=(5, 5), bias=False)]
    return nn.Sequential(*layers)

def cifar10_cnn_A() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=32 * 8 * 8, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        ]
    )

def cifar10_cnn_B():
    return nn.Sequential(
        Pad((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def mnist_cnn_A():
    return nn.Sequential(
        nn.Conv2d(1, 16, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def cifar10_base() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_wide() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_deep() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(8 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cifar10_2_255_simplified() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128 * 8 * 8, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cifar10_8_255_simplified() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128 * 8 * 8, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cifar10_conv_big() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


def getShapeConv(
    in_shape: Tuple[int, int, int],
    conv_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int]:
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class ResNet(nn.Sequential):
    def __init__(
        self,
        block: Type[BasicBlock],
        in_ch: int = 3,
        num_stages: int = 1,
        num_blocks: int = 2,
        num_classes: int = 10,
        in_planes: int = 64,
        bn: bool = True,
        last_layer: str = "avg",
        in_dim: int = 32,
        stride: Optional[Sequence[int]] = None,
    ):
        layers: List[nn.Module] = []
        self.in_planes = in_planes
        if stride is None:
            stride = (num_stages + 1) * [2]

        layers.append(
            nn.Conv2d(
                in_ch,
                self.in_planes,
                kernel_size=3,
                stride=stride[0],
                padding=1,
                bias=not bn,
            )
        )

        _, _, in_dim = getShapeConv(
            (in_ch, in_dim, in_dim), (self.in_planes, 3, 3), stride=stride[0], padding=1
        )

        if bn:
            layers.append(nn.BatchNorm2d(self.in_planes))

        layers.append(nn.ReLU())

        for s in stride[1:]:
            block_layers, in_dim = self._make_layer(
                block,
                self.in_planes * 2,
                num_blocks,
                stride=s,
                bn=bn,
                kernel=3,
                in_dim=in_dim,
            )
            layers.append(block_layers)

        if last_layer == "avg":
            layers.append(nn.AvgPool2d(4))
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    self.in_planes * (in_dim // 4) ** 2 * block.expansion, num_classes
                )
            )
        elif last_layer == "dense":
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(self.in_planes * block.expansion * in_dim**2, 100)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Linear(100, num_classes))
        else:
            exit("last_layer type not supported!")

        super(ResNet, self).__init__(*layers)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_layers: int,
        stride: int,
        bn: bool,
        kernel: int,
        in_dim: int,
    ) -> Tuple[nn.Sequential, int]:
        strides = [stride] + [1] * (num_layers - 1)
        cur_dim: int = in_dim
        layers: List[nn.Module] = []
        for stride in strides:
            layer = block(self.in_planes, planes, stride, bn, kernel, in_dim=cur_dim)
            layers.append(layer)
            cur_dim = layer.out_dim
            layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), cur_dim


def resnet2b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock, num_stages=1, num_blocks=2, in_planes=8, bn=bn, last_layer="dense"
    )


def resnet2b2(bn: bool = False, in_ch: int = 3, in_dim: int = 32) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=in_ch,
        num_stages=2,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2],
    )


def resnet4b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock, num_stages=2, num_blocks=2, in_planes=8, bn=bn, last_layer="dense"
    )


def resnet4b1(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[1, 1, 2, 2, 2],
    )


def resnet4b2(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 1, 1],
    )


def resnet3b2(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=3,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 2],
    )


def resnet9b(bn: bool = False) -> nn.Sequential:
    return ResNet(
        BasicBlock,
        in_ch=3,
        num_stages=3,
        num_blocks=3,
        in_planes=16,
        bn=bn,
        last_layer="dense",
    )

def ConvMedBig(dataset, bn=False, bn2=False, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch, conv_widths=[2,2,8], kernel_sizes=[3,4,4],
                 linear_sizes=[250],  strides=[1,2,2], paddings=[1, 1, 1], net_dim=None, bn=bn, bn2=bn2)

def ConvMed(dataset, bn=False, bn2=False, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch, conv_widths=[2,4], kernel_sizes=[5,4],
                 linear_sizes=[100],  strides=[2,2], paddings=[2,1], net_dim=None, bn=bn, bn2=bn2)

def ConvMed2(dataset, bn=False, bn2=False, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch, conv_widths=[2,8], kernel_sizes=[5,4],
                 linear_sizes=[250],  strides=[2,2], paddings=[2,1], net_dim=None, bn=bn, bn2=bn2)

def ConvMed_tiny(dataset, bn=False, bn2=False, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch, conv_widths=[1,2], kernel_sizes=[5,4],
                 linear_sizes=[50],  strides=[2,2], paddings=[1,1], net_dim=None, bn=bn, bn2=bn2)

class myNet(nn.Module):
    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, conv_widths=None,
                 kernel_sizes=None, linear_sizes=None, depth_conv=None, paddings=None, strides=None,
                 dilations=None, pool=False, net_dim=None, bn=False, bn2=False, max=False, scale_width=True):
        super(myNet, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3]
        if conv_widths is None:
            conv_widths = [2]
        if linear_sizes is None:
            linear_sizes = [200]
        if paddings is None:
            paddings = [1]
        if strides is None:
            strides = [2]
        if dilations is None:
            dilations = [1]
        if net_dim is None:
            net_dim = input_size

        if len(conv_widths) != len(kernel_sizes):
            kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
        if len(conv_widths) != len(paddings):
            paddings = len(conv_widths) * [paddings[0]]
        if len(conv_widths) != len(strides):
            strides = len(conv_widths) * [strides[0]]
        if len(conv_widths) != len(dilations):
            dilations = len(conv_widths) * [dilations[0]]

        self.n_class=n_class
        self.input_size=input_size
        self.input_channel=input_channel
        self.conv_widths=conv_widths
        self.kernel_sizes=kernel_sizes
        self.paddings=paddings
        self.strides=strides
        self.dilations = dilations
        self.linear_sizes=linear_sizes
        self.depth_conv=depth_conv
        self.net_dim = net_dim
        self.bn=bn
        self.bn2 = bn2
        self.max=max

        layers = []

        N = net_dim
        n_channels = input_channel
        self.dims = [(n_channels,N,N)]

        for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
            if scale_width:
                width *= 16
            N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            layers += [nn.Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
            if self.bn:
                layers += [nn.BatchNorm2d(int(width))]
            if self.max:
                layers += [nn.MaxPool2d(int(width))]
            layers += [nn.ReLU((int(width), N, N))]
            n_channels = int(width)
            self.dims += 2*[(n_channels,N,N)]

        if depth_conv is not None:
            layers += [nn.Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                       nn.ReLU((n_channels, N, N))]
            n_channels = depth_conv
            self.dims += 2*[(n_channels,N,N)]

        if pool:
            layers += [nn.GlobalAvgPool2d()]
            self.dims += 2 * [(n_channels, 1, 1)]
            N=1

        layers += [nn.Flatten()]
        N = n_channels * N ** 2
        self.dims += [(N,)]

        for width in linear_sizes:
            if width == 0:
                continue
            layers += [nn.Linear(int(N), int(width))]
            if self.bn2:
                layers += [nn.BatchNorm1d(int(width))]
            layers += [nn.ReLU(width)]
            N = width
            self.dims+=2*[(N,)]

        layers += [nn.Linear(N, n_class)]
        self.dims+=[(n_class,)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class CNN7(myNet):
    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=4, width2=8, linear_size=512,
                 net_dim=None, bn=False, bn2=False):
        super(CNN7, self).__init__(device, dataset, n_class, input_size, input_channel,
                                   conv_widths=[width1, width1, width2, width2, width2], kernel_sizes=[3, 3, 3, 3, 3],
                                   linear_sizes=[linear_size], strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1],
                                   net_dim=net_dim, bn=bn, bn2=bn2)


def CNNA(dataset, bn, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch,
                 conv_widths=[16, 32], kernel_sizes=[4, 4],
                 linear_sizes=[100], strides=[2, 2], paddings=[1, 1],
                 net_dim=None, bn=bn)


def get_dataset_info(dataset):
    if dataset == "mnist":
        return 1, 28, 10
    elif dataset == "emnist":
        return 1, 28, 10
    elif dataset == "fashionmnist":
        return 1, 28, 10
    if dataset == "svhn":
        return 3, 32, 10
    elif dataset == "cifar10":
        return 3, 32, 10
    elif dataset == "tinyimagenet":
        return 3, 56, 200
    else:
        raise ValueError(f"Dataset {dataset} not available")


def freeze_network(network: nn.Module) -> None:
    for param in network.parameters():
        param.requires_grad = False


def load_net_from(config: Bunch) -> nn.Module:
    path = config.network_path
    try:
        n_layers = config.n_layers
        n_neurons_per_layer = config.n_neurons_per_layer
    except AttributeError:
        n_layers = None
        n_neurons_per_layer = None
    return load_net(path, n_layers, n_neurons_per_layer)


def load_net(  # noqa: C901
    path: str, n_layers: Optional[int], n_neurons_per_layer: Optional[int]
) -> nn.Module:
    if path.split(".")[-1] in ["onnx", "gz"]:
        return load_onnx_model(path)[0]
    elif "AGLSTAN" in path:
        original_network = agl_stan()
        m = torch.load("../AGLSTAN/res/chi/best_model.pth", map_location=torch.device('cpu'))['state_dict']
        ws = [m[f'encoder.gconv_layers.{i}.gconv_layer.weights_pool'] for i in range(3)]
        bs = [m[f'encoder.gconv_layers.{i}.gconv_layer.bias_pool'] for i in range(3)]
        original_network = AglStan(m['node_embeddings'], ws, bs)
    elif "mnist_sig" in path and "flattened" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_sig_a_b(n_layers, n_neurons_per_layer)
    elif "mnist" in path and "flattened" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_a_b(n_layers, n_neurons_per_layer)
    elif "mnist-net" in path:
        assert n_layers is not None and n_neurons_per_layer is not None
        original_network = mnist_vnncomp_a_b(n_layers, n_neurons_per_layer)
    elif "mnist_convSmallRELU__Point" in path:
        original_network = mnist_conv_small()
    elif "mnist_SIGMOID" in path:
        original_network = mnist_conv_sigmoid_small()
    elif "mnist_convBigRELU__DiffAI" in path:
        original_network = mnist_conv_big()
    elif "mnist_convSuperRELU__DiffAI" in path:
        original_network = mnist_conv_super()
    elif "cifar10_convSmallRELU__PGDK" in path:
        original_network = cifar10_conv_small()
    elif "cifar_cnn_a" in path:
        original_network = cifar10_cnn_A()
    elif "cifar_cnn_b" in path:
        original_network = cifar10_cnn_B()
    elif "mnist_cnn_a" in path:
        original_network = mnist_cnn_A()
    elif "cifar_base_kw" in path:
        original_network = cifar10_base()
    elif "cifar_wide_kw" in path:
        original_network = cifar10_wide()
    elif "cifar_deep_kw" in path:
        original_network = cifar10_deep()
    elif "cifar10_2_255_simplified" in path:
        original_network = cifar10_2_255_simplified()
    elif "cifar10_8_255_simplified" in path:
        original_network = cifar10_8_255_simplified()
    elif "cifar10_convBigRELU__PGD" in path:
        original_network = cifar10_conv_big()
    elif "resnet_2b2" in path:
        original_network = resnet2b2(bn="bn" in path)
    elif "resnet_2b" in path:
        original_network = resnet2b()
    elif "resnet_3b2" in path:
        original_network = resnet3b2(bn="bn" in path)
    elif "resnet_4b1" in path:
        original_network = resnet4b1(bn="bn" in path)
    elif "resnet_4b2" in path:
        original_network = resnet4b2(bn="bn" in path)
    elif "resnet_4b" in path:
        original_network = resnet4b()
    elif "resnet_9b_bn" in path:
        original_network = resnet9b(bn=True)
    elif "ConvMed_tiny" in path:
        if "cifar10" in path:
            original_network = ConvMed_tiny("cifar10", bn="bn" in path)
        elif "mnist" in path:
            original_network = ConvMed_tiny("mnist", bn="bn" in path)
    elif "ConvMedBig" in path:
        if "cifar10" in path:
            original_network = ConvMedBig("cifar10")
        elif "mnist" in path:
            original_network = ConvMedBig("mnist")
    elif "ConvMed2" in path:
        if "cifar10" in path:
            original_network = ConvMed2("cifar10")
        elif "mnist" in path:
            original_network = ConvMed2("mnist")
    elif "ConvMed" in path:
        if "cifar10" in path:
            original_network = ConvMed("cifar10")
        elif "mnist" in path:
            original_network = ConvMed("mnist")
    elif "SP1" in path:
        #original_network = CNNA("fashionmnist", False, "cuda")
        if "cifar10" in path:
            original_network = CNN7("cuda", "cifar10", input_size=32, input_channel=3, bn=True)
        elif "mnist" in path:
            original_network = CNN7("cuda", "mnist", input_size=28, input_channel=1, bn=True)
        elif "tiny" in path:
            original_network = CNN7("cuda", "tinyimagenet", input_size=56, input_channel=3, n_class=200, bn=True)
    elif "CNN7" in path:
        if "no_BN" in path:
            bn = False
        else:
            bn = True
        if "cifar10" in path:
            original_network = CNN7("cuda", "cifar10", input_size=32, input_channel=3, bn=bn)
        elif "mnist" in path:
            original_network = CNN7("cuda", "mnist", input_size=28, input_channel=1, bn=bn)
        elif "tiny" in path:
            original_network = CNN7("cuda", "tinyimagenet", input_size=56, input_channel=3, n_class=200, bn=bn)
        else:
            raise NotImplementedError(
                "The network specified in the configuration, could not be loaded."
            )
    else:
        raise NotImplementedError(
            "The network specified in the configuration, could not be loaded."
        )
    
    best_model = torch.load("../AGLSTAN/res/chi/best_model.pth", map_location=torch.device('cpu'))
    best_model_state_dict = best_model['state_dict']    
    state_dict = torch.load(path, 
                            map_location=torch.device('cpu') if not torch.cuda.is_available() 
                            else None)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    if "AGLSTAN" in path:
       state_dict = convertStateDict(original_network, original_network.state_dict(), best_model_state_dict)
    original_network.load_state_dict(state_dict)
    # original_network = original_network.blocks
    freeze_network(original_network)

    return original_network

def load_onnx_model(path: str) -> Tuple[nn.Sequential, Tuple[int, ...], str]:
    onnx_model = load_onnx(path)
    return load_onnx_from_proto(onnx_model, path)


def load_onnx_from_proto(
    onnx_model: onnx.ModelProto, path: Optional[str] = None
) -> Tuple[nn.Sequential, Tuple[int, ...], str]:

    onnx_input_dims = onnx_model.graph.input[-1].type.tensor_type.shape.dim
    inp_name = onnx_model.graph.input[-1].name
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    pytorch_model = ConvertModel(onnx_model)

    if path is not None and "unet" in path:
        pytorch_structured = pytorch_model.forward_trace_to_graph_unet()
        softmax_idx = [
            i for (i, layer) in enumerate(pytorch_structured) if "Softmax" in str(layer)
        ][0]
        pytorch_structured = pytorch_structured[: softmax_idx - 1]
        pytorch_structured.append(nn.Flatten())
    elif len(onnx_shape) == 0 and path is not None and "vgg16-7" in path:
        onnx_shape = (3, 224, 224)
        pytorch_structured = pytorch_model.forward_trace_to_graph()
    elif len(onnx_shape) == 0 and path is not None and ("test_nano" in path or "test_tiny" in path or "test_small" in path):
        onnx_shape = (1,)
        pytorch_structured = pytorch_model.forward_trace_to_graph()
    else:
        pytorch_structured = pytorch_model.forward_trace_to_graph()

    return pytorch_structured, onnx_shape, inp_name


@typing.no_type_check
def load_onnx(path: str):
    # The official benchmark repo has all networks with the wrong ending
    if not exists(path) and not path.endswith(".gz"):
        path = path + ".gz"
    if path.endswith(".gz"):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model
