import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple
from module.basic_model import *

class AFFNet(nn.Module):
    def __init__(self,block=BasicBlock, num_classes=2):
        super(AFFNet, self).__init__()
        self.embedding = embed()
        self.merge1 = PatchMerging1D(input_dim=64,output_dim=128)
        self.merge2 = PatchMerging1D(input_dim=128,output_dim=256)
        self.merge3 = PatchMerging1D(input_dim=256,output_dim=512)

        self.layer1 = self._make_layers(block,64)
        self.layer2 = self._make_layers(block,128)
        self.layer3 = self._make_layers(block,256)

        self.fusion1 = Mlp(64,64,act_layer=nn.GELU,drop=0.4)
        self.fusion2 = Mlp(32,32,act_layer=nn.GELU,drop=0.4)
        self.fusion3 = Mlp(16,16,act_layer=nn.GELU,drop=0.4)

        self.decoder2 = Decoder1D(256,128)
        self.decoder1 = Decoder1D(128,64)
        self.decoder0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, num_classes, kernel_size=1, bias=False)  # 使用一维卷积
        )

    def _make_layers(self, block,planes):
        norm1 = nn.BatchNorm1d(planes)
        conv = MBConv1D(in_channels=planes,out_channels=planes,expansion_factor=4,kernel_size=3)
        norm2 = nn.BatchNorm1d(planes)
        affnet = block(hidden_size=planes,num_blocks=8,sparsity_threshold=0.01,act_layer=nn.GELU,hard_thresholding_fraction=1,hidden_size_factor=1)
        return nn.Sequential(*[norm1,conv,norm2,affnet])

    def forward(self,x1,x2):
        x = torch.cat((x1,x2),dim=1)
        x = self.embedding(x)
        feature = x
        x = self.layer1(x)
        x = self.fusion1(x,feature)

        #torch.Size([64, 64, 64])

        skip1 = x
        x = self.merge1(x)
        feature = x
        x = self.layer2(x)
        x = self.fusion2(x,feature)

        #torch.Size([64, 128, 32])

        skip2 = x
        x = self.merge2(x)
        feature = x
        x = self.layer3(x)
        x = self.fusion3(x,feature)

        #torch.Size([64, 256, 16])

        x = self.decoder2(x,skip2)
        #torch.Size([64, 128, 32])
        x = self.decoder1(x,skip1)
        #torch.Size([64, 64, 64])
        x = self.decoder0(x)
        #torch.Size([64, 2, 128])
        x = x.mean(dim=2)  # 在最后一个维度上计算均值

        #print(x.shape)

        return x

