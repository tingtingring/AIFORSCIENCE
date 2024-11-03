import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as Data
import random
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import time
from termcolor import colored
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from module.AFFNet import AFFNet
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


def generate_data(file):

    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    lines = lines[1:]

    if len(lines) > 2400:
        first_2400_lines = lines[:2400]
    else:
        print("Warning: The file has less than 2400 rows. Using all available rows.")
        first_2400_lines = lines

        # 从2400行数据中随机抽取800行
    random.shuffle(first_2400_lines)
    random_800_lines = first_2400_lines[:800]

    # 从整个数据集中取出后800个数据
    if len(lines) > 800:
        last_800_lines = lines[-800:]
    else:
        print("Warning: The file has less than 800 rows. Using all available rows.")
        last_800_lines = lines

    lines = random_800_lines + last_800_lines

    bases = []
    labels = []
    lifes = []
    for row in lines:

        row = row.split(",")
        print(row)
        label = int(row[2])  # 第一列作为标签
        base = [int(x.strip()) for x in row[4:10]]
        life = [int(x.strip()) for x in row[10:]]

        labels.append(label)
        bases.append(torch.tensor(base))
        lifes.append(torch.tensor(life))
    print(labels)
    print(bases)
    print(lifes)
    return  torch.tensor(labels),bases, lifes

# base = ["f.31.0.0","f.34.0.0","f.21022.0.0","f.21001.0.0","f.4079.0.0","f.4080.0.0"]
# life = ["f.2907.0.0","f.2907.0.1","f.2907.0.2","f.2907.0.3","f.2907.0.4","f.2907.0.5","f.2907.0.6","f.2907.0.7",f.2907.0.8	f.2907.0.9	f.2907.0.10	f.2907.0.11	f.2907.0.12	f.2907.0.13	f.2907.0.14	f.2907.0.15	f.2907.0.16	f.2907.0.17	f.2907.0.18	f.2907.0.19	f.2907.0.20	f.2907.0.21	f.2907.0.22	f.2907.0.23	f.2907.0.24	f.2907.0.25	f.2907.0.26	f.2907.0.27	f.2907.0.28	f.2907.0.29	f.2907.0.30	f.2907.0.31	f.2907.0.32	f.2907.0.33	f.2907.0.34	f.2907.0.35	f.2907.0.36	f.2907.0.37	f.2907.0.38	f.2907.0.39	f.2907.0.40	f.2907.0.41	f.2907.0.42	f.2907.0.43	f.2907.0.44	f.2907.0.45	f.2907.0.46	f.2907.0.47	f.2907.0.48	f.2907.0.49	f.2907.0.50	f.2907.0.51	f.2907.0.52	f.2907.0.53	f.2907.0.54	f.2907.0.55	f.2907.0.56	f.2907.0.57	f.2907.0.58	f.2907.0.59	f.2907.0.60	f.2907.0.61	f.2907.0.62	f.2907.0.63	f.2907.0.64	f.2907.0.65	f.2907.0.66	f.2907.0.67
# ]

class t2dDataset(Dataset):
    def __init__(self, file):
        self.datas = pd.read_csv(file)
        self.labels = np.array(self.datas['T2D'])
        self.bases = np.array(self.datas.iloc[:,4:10])
        self.lifes = np.array(self.datas.iloc[:,10:])

    def __getitem__(self, index):
        return self.labels[index], self.bases[index], self.lifes[index]

    def __len__(self):
        return len(self.labels)

# class ExamPle(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden_dim = 128
#         self.batch_size = 128
#         self.dropout = 0.4
#         self.conv = nn.Embedding(2100,512)
#
#         self.fusion = feature_fusion()
#
#         self.encoder_layer_life = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         self.encoder_layer_base = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#
#         # Pass the sequence information and secondary structure information through different transformer encoders
#         self.transformer_encoder_life = nn.TransformerEncoder(self.encoder_layer_life, num_layers=1)
#         self.transformer_encoder_base = nn.TransformerEncoder(self.encoder_layer_base, num_layers=1)
#
#         # self.gru_life = nn.GRU(512, self.hidden_dim, num_layers=2,
#         #                       bidirectional=True, dropout=self.dropout)
#         # self.gru_base = nn.GRU(512, self.hidden_dim, num_layers=2,
#         #                      bidirectional=True, dropout=self.dropout)
#
#         # Reduce the dimension of the embedding
#         # self.block_life = nn.Sequential(nn.Linear(14592, 2048),
#         #                                nn.BatchNorm1d(2048),
#         #                                nn.LeakyReLU(),
#         #                                 nn.Dropout(self.dropout),
#         #                                nn.Linear(2048, 1024),
#         #                                 nn.BatchNorm1d(1024))
#         #
#         # self.block_base = nn.Sequential(nn.Linear(131584, 2048),
#         #                               nn.BatchNorm1d(2048),
#         #                               nn.LeakyReLU(),
#         #                                 nn.Dropout(self.dropout),
#         #                               nn.Linear(2048, 1024),
#         #                                 nn.BatchNorm1d(1024))
#         #
#         # self.block1 = nn.Sequential(nn.Linear(2048, 1024),
#         #                             nn.BatchNorm1d(1024),
#         #                             nn.LeakyReLU(),
#         #                             nn.Dropout(self.dropout),
#         #                             nn.Linear(1024, 256),
#         #                             nn.BatchNorm1d(256))
#         #
#         # self.block2 = nn.Sequential(nn.Linear(256, 8),
#         #                             nn.ReLU(),
#         #                             nn.Dropout(self.dropout),
#         #                             nn.Linear(8, 2))
#
#     def forward(self, x_base,x_life):
#         #batch,64,64
#         x_base = self.fusion(x_base, x_life)
#         x_life = self.conv(x_life)
#
#         output = self.transformer_encoder_life(x_life).permute(1, 0, 2)
#         # output, hn = self.gru_life(output)
#         # output = output.permute(1, 0, 2)
#         # hn = hn.permute(1, 0, 2)
#         # output = output.reshape(output.shape[0], -1)
#         # hn = hn.reshape(output.shape[0], -1)
#         # output = torch.cat([output, hn], 1)
#         # #print(output.shape)
#         # output = self.block_life(output)
#
#         # Process the secondary structure information
#         #batch,64,64
#
#         base_output = self.transformer_encoder_base(x_base).permute(1, 0, 2)
#         # base_output, base_hn = self.gru_base(base_output)
#         # base_output = base_output.permute(1, 0, 2)
#         # base_hn = base_hn.permute(1, 0, 2)
#         # base_output = base_output.reshape(base_output.shape[0], -1)
#         # base_hn = base_hn.reshape(base_output.shape[0], -1)
#         # base_output = torch.cat([base_output, base_hn], 1)
#         # #print(base_output.shape)
#         # base_output = self.block_base(base_output)
#
#         #Fusion of features
#         representation = torch.cat([output, base_output], dim=1)
#         #representation = output
#
#         return self.block2(self.block1(representation))
#
#     # def train_model(self, x_base, x_life):
#     #     with torch.no_grad():
#     #         output = self.forward(x_base, x_life)
#     #     return self.block2(output)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # 添加Dropout
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.MaxPool1d(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck)
        dec4 = dec4+ enc4  # Skip connection
        dec3 = self.decoder3(dec4)
        dec3 = dec3 + enc3  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = dec2 + enc2  # Skip connection
        dec1 = self.decoder1(dec2)

        return dec1

class ExamPle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 64
        self.dropout = 0.4
        self.conv = nn.Embedding(2100, 128)

        self.fusion = feature_fusion()
        self.base_unet = UNet(in_channels=128, out_channels=128)  # 使用U-Net
        self.life_unet = UNet(in_channels=55, out_channels=128)

        self.encoder_layer_life = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.encoder_layer_base = nn.TransformerEncoderLayer(d_model=128, nhead=8)

        self.transformer_encoder_life = nn.TransformerEncoder(self.encoder_layer_life, num_layers=1)
        self.transformer_encoder_base = nn.TransformerEncoder(self.encoder_layer_base, num_layers=1)

        self.block1 = nn.Sequential(
            nn.Linear(256*128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.block2 = nn.Sequential(
            nn.Linear(512, 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(8, 2)
        )

    def forward(self, x_base, x_life):
        x_base = self.fusion(x_base, x_life)
        x_life = self.conv(x_life)

        output = self.transformer_encoder_life(x_life)
        base_output = self.transformer_encoder_base(x_base)

        # 使用U-Net处理输出
        output = self.life_unet(output)
        base_output = self.base_unet(base_output)

        representation = torch.cat([output, base_output], dim=1)
        representation = torch.flatten(representation, start_dim=1)
        return self.block2(self.block1(representation))


class feature_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2100, 128)
        self.conv1 = nn.Conv1d(61, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(61, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(61, 16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(61, 16, kernel_size=9, stride=1, padding=4)
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

        self.fc1 = nn.Linear(128, 256)  # 修改输入维度
        self.action = nn.LeakyReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.eca = eca_layer1d(128)

    def forward(self, x_base, x_life):
        x_cat = torch.cat([x_life, x_base], dim=1)
        x_cat = self.embedding(x_cat)

        x = []
        for conv in self.convs:
            x.append(conv(x_cat))

        x = torch.cat(x, dim=1)
        x = self.eca(x)
        # 将输入和输出相加

        x = self.fc1(x)
        x = self.action(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.action(x)
        x = self.eca(x)
        return x

class eca_layer1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.channel_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_features = self.global_avg_pooling(x)
        x_features = x_features.reshape((-1, 1, self.in_channels))
        x_features = self.channel_conv(x_features)
        x_features = x_features.reshape((-1, self.in_channels))
        channel_weights = self.sigmoid(x_features)
        x = x * channel_weights.unsqueeze(-1) + x
        return x

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    base1_ls = []
    base2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    life1_ls = []
    life2_ls = []

    batch_size = len(batch)

    for i in range(int(batch_size / 2)):
        label1, base1, life1 = batch[i][0], batch[i][1], batch[i][2]
        label2, base2, life2 = batch[i + int(batch_size / 2)][0], \
            batch[i + int(batch_size / 2)][1], \
            batch[i + int(batch_size / 2)][2]

        # 确保 base 和 label 都是 PyTorch Tensor
        base1 = torch.tensor(base1, dtype=torch.long) if not isinstance(base1, torch.Tensor) else base1
        base2 = torch.tensor(base2, dtype=torch.long) if not isinstance(base2, torch.Tensor) else base2
        label1 = torch.tensor(label1, dtype=torch.long) if not isinstance(label1, torch.Tensor) else label1
        label2 = torch.tensor(label2, dtype=torch.long) if not isinstance(label2, torch.Tensor) else label2
        life1 = torch.tensor(life1, dtype=torch.long) if not isinstance(life1, torch.Tensor) else life1
        life2 = torch.tensor(life2, dtype=torch.long) if not isinstance(life2, torch.Tensor) else life2

        # 计算 label 的异或（^）
        label = (label1 ^ label2)

        # 添加维度并添加到列表中
        base1_ls.append(base1.unsqueeze(0))
        base2_ls.append(base2.unsqueeze(0))
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
        life1_ls.append(life1.unsqueeze(0))
        life2_ls.append(life2.unsqueeze(0))

    # 将列表中的张量拼接为一个批次，并移动到设备上
    base1 = torch.cat(base1_ls).to(device)  # [batch_size/2, ...]
    base2 = torch.cat(base2_ls).to(device)  # [batch_size/2, ...]
    life1 = torch.cat(life1_ls).to(device)  # [batch_size/2, ...]
    life2 = torch.cat(life2_ls).to(device)  # [batch_size/2, ...]
    label = torch.cat(label_ls).to(device)  # [batch_size/2]
    label1 = torch.cat(label1_ls).to(device)  # [batch_size/2]
    label2 = torch.cat(label2_ls).to(device)  # [batch_size/2]

    return base1, base2, label, label1, label2, life1, life2

def collate1(batch):
    base_ls = []
    label_ls = []
    life_ls = []

    for item in batch:
        label,base, life = item[0], item[1], item[2]  # 获取数据、标签和其他特征

        base = torch.tensor(base,dtype=torch.long) if not isinstance(base, torch.Tensor) else base
        label = torch.tensor(label,dtype=torch.long) if not isinstance(label, torch.Tensor) else label
        life = torch.tensor(life,dtype=torch.long) if not isinstance(life, torch.Tensor) else life

        base_ls.append(base.unsqueeze(0))  # 添加维度以便后续拼接
        label_ls.append(label.unsqueeze(0))
        life_ls.append(life.unsqueeze(0))

    # 将列表中的张量拼接为一个批次，并移动到设备上
    base = torch.cat(base_ls).to(device)  # [batch_size, ...]
    label = torch.cat(label_ls).to(device)  # [batch_size]
    life = torch.cat(life_ls).to(device)  # [batch_size, ...]

    return label,base,life

device = torch.device("cuda", 1)
#device = 'cpu'

# train_data, train_label, train_ss = generate_data("./dataset/train/merged_data_task1.csv")
#
#
# test_data, test_label, test_ss = generate_data("./dataset/test/merged_data_task1.csv")

# train_dataset = Data.TensorDataset(train_data, train_label, train_ss)
# test_dataset = Data.TensorDataset(test_data, test_label, test_ss)

train_dataset = t2dDataset("./dataset/train/merged_data_task1.csv")
test_dataset = t2dDataset("./dataset/test/merged_data_task1.csv")

batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True,collate_fn=collate1)#, collate_fn=collate

def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    for label,base,life in data_iter:
        label, base, life = label.to(device), base.to(device), life.to(device)
        base = torch.tensor(base, dtype=torch.long) if not isinstance(base, torch.Tensor) else base
        label = torch.tensor(label, dtype=torch.long) if not isinstance(label, torch.Tensor) else label
        life = torch.tensor(life, dtype=torch.long) if not isinstance(life, torch.Tensor) else life
        #outputs = net.train_model(base, life)
        outputs = net(base, life)
        outputs_cpu = outputs.cpu()
        label_cpu = label.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + label_cpu.tolist()
    performance, roc_data, prc_data = calculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data


def calculate_metric(pred_prob, label_pred, label_real):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(label_real, label_pred).ravel()

    # 准确率 (Accuracy)
    ACC = accuracy_score(label_real, label_pred)

    # 召回率 (Recall)
    Recall = recall_score(label_real, label_pred)

    # 精确率 (Precision)，处理 UndefinedMetricWarning
    Precision = precision_score(label_real, label_pred, zero_division=0)

    # F1-score
    F1 = f1_score(label_real, label_pred, zero_division=0)

    # 特异性 (Specificity)
    Specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # MCC (Matthews 相关系数)
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0

    # ROC 和 AUC
    FPR, TPR, _ = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)

    # PRC 和 AP
    precision, recall, _ = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob)

    # 返回性能指标
    performance = [ACC, Precision, Recall, F1, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def to_log(log):
    with open("./results/ExamPle_Log.log", "a+") as f:
        f.write(log + '\n')


kf = KFold(n_splits=5, shuffle=True, random_state=42)

net = AFFNet().to(device)
lr = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion_model = nn.CrossEntropyLoss(reduction='sum')
best_acc = 0
EPOCH = 100
patience = 5  # 设置忍耐值
test_results = []  # 用于存储每一折的测试结果
best_model_path = None  # 存储最佳模型路径
best_val_performance = float('-inf')  # 初始化最佳验证性能
best_val_epoch = -1  # 初始化最佳验证轮次

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    log_message = f"第 {fold + 1} 折 / {kf.n_splits} 折"
    print(log_message)
    to_log(log_message)

    # 划分数据集
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate1)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate1)

    net = AFFNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    current_best_val_performance = float('-inf')  # 当前折的最佳验证性能

    model_dir = './Task1Model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()

        for label, base, life in train_loader:
            output = net(base, life)
            loss = criterion_model(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, _, _ = evaluate(train_loader, net)
            val_performance, _, _ = evaluate(val_loader, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}\n'
        results += '=' * 16 + ' 验证性能. Epoch[{}] '.format(epoch + 1) + '=' * 16 + '\n' + \
                   '[ACC,\tPrecision,\tRecall,\tF1,\tAUC,\tMCC]\n' + \
                   '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}\n'.format(
                       val_performance[0], val_performance[1], val_performance[2],
                       val_performance[3], val_performance[4], val_performance[5]) + '=' * 60

        print(results)
        to_log(results)

        # 使用验证损失进行判断
        current_val_loss = val_performance[0]  # 假设 val_performance[0] 是验证损失
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            early_stopping_counter = 0  # 重置计数器
            # 保存当前折的最佳模型
            filename = f'best_model_fold_{fold + 1}_acc_{val_performance[0]:.4f}.pt'
            save_path_pt = os.path.join(model_dir, filename)
            torch.save(net.state_dict(), save_path_pt)

            # 更新最佳验证性能
            if val_performance[0] > best_val_performance:
                best_val_performance = val_performance[0]
                best_model_path = save_path_pt  # 记录最佳模型路径
                best_val_epoch = epoch + 1  # 记录最佳验证轮次

        else:
            early_stopping_counter += 1

        # 更新当前折的最佳验证性能
        current_best_val_performance = max(current_best_val_performance, val_performance[0])

        # 检查是否达到忍耐值
        if early_stopping_counter >= patience:
            print(f"提前停止，忍耐值已达到 {patience}。")
            to_log(f"提前停止，忍耐值已达到 {patience}。")
            break

    # 输出当前折的最佳验证效果
    best_fold_results = f"第 {fold + 1} 折 当前最佳验证效果 (ACC): {current_best_val_performance:.4f}"
    print(best_fold_results)
    to_log(best_fold_results)

    # 在每一折结束后进行测试验证
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate1)
    with torch.no_grad():
        test_performance, _, _ = evaluate(test_loader, net)

    results = (
        "=" * 40 + "\n" +
        colored("         测试结果", "red") + "\n" +
        "=" * 40 + "\n" +
        f"准确率 (ACC):      {colored(f'{test_performance[0]:.4f}', 'red')}\n" +
        f"精准率 (Precision): {colored(f'{test_performance[1]:.4f}', 'red')}\n" +
        f"召回率 (Recall):    {colored(f'{test_performance[2]:.4f}', 'red')}\n" +
        f"F1 值:           {colored(f'{test_performance[3]:.4f}', 'red')}\n" +
        f"AUC 值:          {colored(f'{test_performance[4]:.4f}', 'red')}\n" +
        f"MCC 值:          {colored(f'{test_performance[5]:.4f}', 'red')}\n" +
        "=" * 40
    )

    print(results)
    to_log(results)
    test_results.append(test_performance)  # 存储测试结果

# 打印每一折的测试结果
for i, result in enumerate(test_results):
    test_result_message = (
        f"第 {i + 1} 折测试结果: [ACC: {result[0]:.4f}, Precision: {result[1]:.4f}, Recall: {result[2]:.4f}, "
        f"F1: {result[3]:.4f}, AUC: {result[4]:.4f}, MCC: {result[5]:.4f}]"
    )
    print(test_result_message)
    to_log(test_result_message)

# 输出最佳模型路径
if best_model_path:
    best_model_message = f"最佳模型保存为: {best_model_path}"
    print(best_model_message)
    to_log(best_model_message)