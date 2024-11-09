import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple
from module.basic_model import *

class AFFTrans(nn.Module):
    def __init__(self,block=BasicBlock, num_classes=2):
        super(AFFTrans, self).__init__()
        self.embedding = embed()
        self.merge1 = PatchMerging1D(input_dim=64,output_dim=128)
        self.merge2 = PatchMerging1D(input_dim=128,output_dim=256)

        self.layer1 = self._make_layers(block,64)
        self.layer2 = self._make_layers(block,128)
        self.layer3 = self._make_layers(block,256)

        self.fusion1 = Mlp(64,64,act_layer=nn.GELU,drop=0.4)
        self.fusion2 = Mlp(32,32,act_layer=nn.GELU,drop=0.4)
        self.fusion3 = Mlp(16,16,act_layer=nn.GELU,drop=0.4)

        self.decoder2 = Decoder1D2(256,128)
        self.decoder1 = Decoder1D2(128,64)
        self.decoder01 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, num_classes, kernel_size=1, bias=False)  # 使用一维卷积
        )
        self.year_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, 31, kernel_size=1, bias=False)  # 使用一维卷积
        )
        self.month_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, 12, kernel_size=1, bias=False)  # 使用一维卷积
        )
        self.day_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, 12, kernel_size=1, bias=False)  # 使用一维卷积
        )
        self.day_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 使用 'linear' 模式进行一维上采样
            nn.Conv1d(64, 31, kernel_size=1, bias=False)  # 使用一维卷积
        )
        #如果有时间的话可以试着比较一下Conv和Transformer的区别，但是我感觉还是过拟合的问题

        self.ecg_embedding = ecg_embed()
        self.attention = AttentionAddConv(64)
        self.ecg_down1 = down_conv1d(64,128)
        self.ecg_down2 = down_conv1d(128,256)
        self.ecg_fusion = ecg_fusion(256,256,act_layer=nn.GELU,drop=0.4)

        self.ecg_up2 = up_conv1d(256,128)
        self.ecg_up1 = up_conv1d(128,64)
        self.plm = PixLevelModule1D(64)

    def _make_layers(self, block,planes):
        norm1 = nn.BatchNorm1d(planes)
        conv = MBConv1D(in_channels=planes,out_channels=planes,expansion_factor=4,kernel_size=3)
        norm2 = nn.BatchNorm1d(planes)
        affnet = block(hidden_size=planes,num_blocks=8,sparsity_threshold=0.01,act_layer=nn.GELU,hard_thresholding_fraction=1,hidden_size_factor=1)
        return nn.Sequential(*[norm1,conv,norm2,affnet])

    def forward(self,x1,x2,x_ecg):
        x = torch.cat((x1,x2),dim=1)
        x = self.embedding(x)
        feature = x
        x = self.layer1(x)
        x = self.fusion1(x,feature)

        # torch.Size([64, 64, 64])
        x_ecg = x_ecg.permute(0,2,1)
        x_ecg = self.ecg_embedding(x_ecg)
        x_ecg = self.attention(x_ecg)
        x_ecg1 = self.ecg_down1(x_ecg,x)

        skip1 = x
        x = self.merge1(x)
        feature = x
        x = self.layer2(x)
        x = self.fusion2(x,feature)

        # torch.Size([64, 128, 32])

        x_ecg2 = self.ecg_down2(x_ecg1,x)

        skip2 = x
        x = self.merge2(x)
        feature = x
        x = self.layer3(x)
        x = self.fusion3(x,feature)

        # torch.Size([64, 256, 16])
        x_ecg3 = self.ecg_fusion(x_ecg2,x)

        x = self.decoder2(x,skip2,x_ecg3)

        x_ecg3 = self.ecg_up2(x_ecg3,x_ecg1)
        x = self.decoder1(x,skip1,x_ecg3)
        # print(x.shape)
        x_ecg3 = self.ecg_up1(x_ecg3,x_ecg)
        x = self.plm(x,x_ecg3)
        x_date = x
        x = self.decoder01(x)

        return x,x_date

    def train_model_label(self, base,life,ecg_data):
            x,_ = self.forward(base,life,ecg_data)
            x = x.mean(dim=2)  # 在最后一个维度上计算均值
            # print(x.shape)

            return x

    def train_model_date(self, base,life,ecg_data):
            _,x_date = self.forward(base,life,ecg_data)
            x_year = self.year_decoder(x_date)
            x_month = self.month_decoder(x_date)
            x_day = self.day_decoder(x_date)
            x_year = x_year.mean(dim=2)
            x_month = x_month.mean(dim=2)
            x_day = x_day.mean(dim=2)
            return x_year,x_month,x_day
