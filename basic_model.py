import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class embed(nn.Module):
    def __init__(self, in_chans= 61,embed_dim=64, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim


        self.embedding = nn.Embedding(2100,embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(in_chans)
        else:
            self.norm = None
        self.conv = nn.Sequential(nn.Conv1d(in_chans,64,kernel_size=1),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU6(inplace=True),)

    def forward(self, x):
        #B, C, H, W = x.shape
        # print(x.shape)
        x = self.embedding(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.conv(x)
        return x

class PatchMerging1D(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = norm_layer(input_dim)
        self.reduction = nn.Conv1d(input_dim, output_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features*2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x1,x2):
        x = self.fc1(x1)
        x = self.act(x)
        x = self.drop(torch.cat((x,x2),dim=-1))
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, act_layer=nn.GELU,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act = act_layer()
        self.act2 = act_layer()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()

        B, C, L = x.shape  # 假设输入形状为 [批次大小, 通道数, 长度]
        # print(x.shape)
        # print(x.dtype)
        # 一维傅里叶变换
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        origin_ffted = x

        x = x.reshape(B, self.num_blocks, self.block_size, -1)  # 调整形状
        o1_real = self.act(
            torch.einsum('bkih,kio->bkoh', x.real, self.w1[0]) - \
            torch.einsum('bkih,kio->bkoh', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkih,kio->bkoh', x.imag, self.w1[0]) +
            torch.einsum('bkih,kio->bkoh', x.real, self.w1[1]) +
            self.b1[1, :, :, None]
        )

        o2_real = (
                torch.einsum('bkih,kio->bkoh', o1_real, self.w2[0]) -
                torch.einsum('bkih,kio->bkoh', o1_imag, self.w2[1]) +
                self.b2[0, :, :, None]
        )

        o2_imag = (
                torch.einsum('bkih,kio->bkoh', o1_imag, self.w2[0]) +
                torch.einsum('bkih,kio->bkoh', o1_real, self.w2[1]) +
                self.b2[1, :, :, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, -1)  # 调整形状以匹配输出

        x = x * origin_ffted
        x = torch.fft.irfft(x, n=L, dim=2, norm="ortho")  # 一维逆傅里叶变换
        x = x.type(dtype)

        return x + bias


class MBConv1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_factor: int,
                 kernel_size: int,
                 stride: int = 1,
                 use_se: bool = False,
                 drop_rate: float = 0.0):
        super(MBConv1D, self).__init__()

        # Expansion phase
        self.expansion = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * expansion_factor, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True)  # Use ReLU6 for better performance in mobile networks
        )

        # Depthwise convolution phase
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm1d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True)
        )

        # Pointwise linear convolution phase
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels * expansion_factor, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # Squeeze-and-Excitation layer (optional)
        self.se = None
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels * expansion_factor, in_channels * expansion_factor // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels * expansion_factor // 4, in_channels * expansion_factor, kernel_size=1),
                nn.Sigmoid()
            )

        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        #print(x.shape)

        # Expansion phase
        x = self.expansion(x)

        # Depthwise convolution phase
        x = self.depthwise(x)

        # Squeeze-and-Excitation phase
        if self.se is not None:
            se = self.se(x)
            x = x * se

        # Pointwise convolution phase
        x = self.pointwise(x)

        # Dropout layer (if specified)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        # Residual connection
        if self.use_residual:
            x += identity

        return x

class Decoder1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder1D, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x

class Decoder1D2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder1D2, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.plm = PixLevelModule1D(in_channels)

    def forward(self, x1, x2, x_ecg):
        x1 = self.plm(x1,x_ecg)
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x

class ecg_embed(nn.Module):

    def __init__(self, in_chans=12, embed_dim=64, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.in_chans = in_chans  # 输入的导联数
        self.embed_dim = embed_dim  # 嵌入维度

        self.embedding = nn.Embedding(15000, embed_dim)  # 嵌入层
        if norm_layer is not None:
            self.norm = norm_layer(in_chans)  # 标准化层
        else:
            self.norm = None

        # 卷积层，kernel_size=1以保持信号的时间序列长度
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=1),  # 输入通道数为嵌入维度
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        B,S,L = x.shape
        # x 的形状为 (batch_size, 12, 600)
        # 首先将 x 转换为 (batch_size * 12 * 600) 的形状以适应嵌入层
        x = x.contiguous().view(-1, 1)  # 将 x 变为 (batch_size * 12 * 600, 1)

        # 通过嵌入层
        x = self.embedding(x)  # 输出形状为 (batch_size * 12 * 600, embed_dim)

        # 将 x 的形状恢复为 (batch_size, 12, 500, embed_dim)
        x = x.contiguous().view(-1, self.in_chans, 600, self.embed_dim)  # 变为 (batch_size, 12, 600, embed_dim)

        # 交换维度以适应卷积层 (batch_size, embed_dim, 12, 600)
        x = x.contiguous().permute(0, 3, 1, 2)  # 变为 (batch_size, embed_dim, 12, 600)

        x = x.contiguous().view(B,self.embed_dim,S*L)
        # 进行卷积
        x = self.conv(x)  # 输出形状为 (batch_size, 64, 12* 600)

        x = x.contiguous().view(B,self.embed_dim,S,L)
        # 将输出的维度恢复到 (batch_size, 12, 600, 64)
        x = x.permute(0, 2, 3, 1)  # 变为 (batch_size, 12, 600, 64)

        return x

class AttentionAddConv(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionAddConv, self).__init__()
        self.embed_dim = embed_dim
        # 定义线性层用于计算查询、键和值
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.conv = nn.Conv1d(600, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        # x 的形状是 (batch_size, 12, 600, embed_dim)
        batch_size, num_channels, seq_length, embed_dim = x.size()

        # 将 x 变形为 (batch_size * num_channels, seq_length, embed_dim) 以便计算
        x = x.contiguous().view(-1, seq_length, embed_dim)

        # 计算查询、键和值
        Q = self.query(x)  # (batch_size * num_channels, seq_length, embed_dim)
        K = self.key(x)    # (batch_size * num_channels, seq_length, embed_dim)
        V = self.value(x)  # (batch_size * num_channels, seq_length, embed_dim)

        # 计算注意力权重
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (batch_size * num_channels, seq_length, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size * num_channels, seq_length, seq_length)

        # 计算加权值
        attention_output = torch.bmm(attention_weights, V)  # (batch_size * num_channels, seq_length, embed_dim)

        # 返回到原来的形状
        attention_output = attention_output.view(batch_size, num_channels, seq_length, embed_dim)  # (batch_size, 12, 600, embed_dim)

        # 计算加权值的按位相加
        weighted_sum = attention_output.sum(dim=1)  # 在通道维度上进行求和，得到 (batch_size, 500, embed_dim)

        weighted_sum = self.conv(weighted_sum)  # (batch_size, 64, embed_dim)
        weighted_sum = self.bn(weighted_sum)  # (batch_size, 64, embed_dim)

        return weighted_sum


class down_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_conv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=2,stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1,x2):
        x = x1 + x2
        x = self.conv(x)
        return x

class up_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv1d, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1,x2):
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        return x

class ecg_fusion(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features*16, hidden_features*16)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features*16 * 2, out_features*16)
        self.drop = nn.Dropout(drop)

    def forward(self, x1, x2):
        B,L,S = x1.shape
        x = torch.flatten(x1,start_dim=1)

        x = self.fc1(x)
        x = self.act(x)
        x2 = torch.flatten(x2,start_dim=1)
        x = self.drop(torch.cat((x, x2), dim=-1))
        x = self.fc2(x)
        x = self.drop(x)
        x = torch.reshape(x,(B,L,S))
        return x


class PixLevelModule1D(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule1D, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv1d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv1d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):
        # 假设 x 的形状为 (batch_size, in_channels, seq_length)
        x = x1 + x2
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1, keepdim=True)  # (batch_size, 1, seq_length)

        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1)[0].unsqueeze(dim=1)  # (batch_size, 1, seq_length)

        x_out = x_max + x_avg  # 按位相加
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)  # 在通道维度拼接

        # 需要调整形状以适应瓶颈层
        batch_size, num_channels, seq_length = x_output.size()
        x_output = x_output.view(batch_size, -1, 3)  # (batch_size, 3, seq_length)

        x_output = self.bottleneck(x_output)  # 应用瓶颈层
        y = x_output.view(batch_size, -1, seq_length) * x  # 逐元素相乘

        return y
