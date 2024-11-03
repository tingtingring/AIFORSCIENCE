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
from module.AFFTrans import AFFTrans
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from torch.utils.data import Dataset


class t2dDataset(Dataset):
    def __init__(self, file, ecg_folder):
        self.datas = pd.read_csv(file)
        self.labels = np.array(self.datas['T2D'])
        self.dates = np.array(self.datas['date'])
        self.bases = np.array(self.datas.iloc[:, 4:10])
        self.lifes = np.array(self.datas.iloc[:, 10:])
        self.ecg_folder = ecg_folder  # 存储 ECG 文件夹路径

    def __getitem__(self, index):
        label = self.labels[index]
        base = self.bases[index]
        life = self.lifes[index]

        # 获取 date 并转换为一维向量
        date_str = self.dates[index]
        date_vector = self.date_to_vector(date_str)  # 调用转换函数

        # 获取 ID
        data_id = self.datas.iloc[index]['f.eid']  # 假设 'f.eid' 列存在

        # 加载 ECG 数据
        ecg_data = self.load_ecg_data(data_id)

        return label, base, life, ecg_data  # 返回 date_vector

    def __len__(self):
        return len(self.labels)

    def load_ecg_data(self, data_id):
        # 以 id_20205_2.csv 命名
        ecg_file = os.path.join(self.ecg_folder, f"{data_id}_20205_2_0.csv")  # 根据实际文件名格式调整

        # 检查文件是否存在
        if os.path.exists(ecg_file):
            ecg_data = pd.read_csv(ecg_file)
            return np.array(ecg_data)  # 转换为 NumPy 数组
        else:
            raise FileNotFoundError(f"ECG file for ID {data_id} not found at {ecg_file}.")

    def date_to_vector(self, date_str):
        """将日期字符串转换为大小为 (1, 3) 的向量"""
        # 解析日期字符串
        if '/' in date_str:
            year, month, day = map(int, date_str.split('/'))
        elif '-' in date_str:
            year, month, day = map(int, date_str.split('-'))
        else:
            raise ValueError(f"日期格式不正确: {date_str}")

        return np.array([year, month, day])  # 返回一维数组

def collate1(batch):
    base_ls = []
    label_ls = []
    life_ls = []
    ecg_ls = []

    for item in batch:
        label,base, life,ecg_data = item[0], item[1], item[2] ,item[3] # 获取数据、标签和其他特征

        base = torch.tensor(base,dtype=torch.long) if not isinstance(base, torch.Tensor) else base
        label = torch.tensor(label,dtype=torch.long) if not isinstance(label, torch.Tensor) else label
        life = torch.tensor(life,dtype=torch.long) if not isinstance(life, torch.Tensor) else life
        ecg_data = torch.tensor(ecg_data,dtype=torch.long) if not isinstance(ecg_data, torch.Tensor) else ecg_data

        base_ls.append(base.unsqueeze(0))  # 添加维度以便后续拼接
        label_ls.append(label.unsqueeze(0))
        life_ls.append(life.unsqueeze(0))
        ecg_ls.append(ecg_data.unsqueeze(0))
    # 将列表中的张量拼接为一个批次，并移动到设备上
    base = torch.cat(base_ls).to(device)  # [batch_size, ...]
    label = torch.cat(label_ls).to(device)  # [batch_size]
    life = torch.cat(life_ls).to(device)  # [batch_size, ...]
    ecg_data = torch.cat(ecg_ls).to(device)  # [batch_size, ...]
    return label,base,life,ecg_data

device = torch.device("cuda", 1)
#device = 'cpu'

ecg_folder_train = "./dataset/train/ecg_data_without_norm"
ecg_folder_test = "./dataset/test/ecg_data_without_norm"

train_dataset = t2dDataset("./dataset/train/data_task2.csv",ecg_folder_train)
test_dataset = t2dDataset("./dataset/test/data_task2.csv",ecg_folder_test)

batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True,collate_fn=collate1)#, collate_fn=collate


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []

    for label, base, life, ecg_data in data_iter:
        label, base, life, ecg_data = label.to(device), base.to(device), life.to(device), ecg_data.to(device)
        base = torch.tensor(base, dtype=torch.long) if not isinstance(base, torch.Tensor) else base
        label = torch.tensor(label, dtype=torch.long) if not isinstance(label, torch.Tensor) else label
        life = torch.tensor(life, dtype=torch.long) if not isinstance(life, torch.Tensor) else life
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32) if not isinstance(ecg_data, torch.Tensor) else ecg_data

        _,_ = net(base, life, ecg_data)
        outputs = net.train_model_label(base,life,ecg_data)
        outputs_cpu = outputs.cpu()
        label_cpu = label.cpu()

        pred_prob += outputs_cpu[:, 1].tolist()
        label_pred += outputs.argmax(dim=1).tolist()
        label_real += label_cpu.tolist()


    performance, roc_data, prc_data = calculate_metric(pred_prob, label_pred, label_real)

    return performance, roc_data, prc_data


def calculate_metric(pred_prob, label_pred, label_real):
    tn, fp, fn, tp = confusion_matrix(label_real, label_pred).ravel()
    ACC = accuracy_score(label_real, label_pred)
    Recall = recall_score(label_real, label_pred)
    Precision = precision_score(label_real, label_pred, zero_division=0)
    F1 = f1_score(label_real, label_pred, zero_division=0)
    Specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    FPR, TPR, _ = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    precision, recall, _ = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob)

    performance = [ACC, Precision, Recall, F1, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def calculate_date_metric(date_pred, date_real):
    mse = mean_squared_error(date_real, date_pred)
    mae = mean_absolute_error(date_real, date_pred)
    return mse, mae  # 返回 MSE 和 MAE


def to_log(log):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./results/AFFTrans_Log_{current_time}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    with open(log_filename, "a+") as f:
        f.write(log + '\n')


kf = KFold(n_splits=5, shuffle=True, random_state=42)


lr = 0.0001

criterion_model = nn.CrossEntropyLoss(reduction='sum')
criterion_date = nn.MSELoss()
best_acc = 0
EPOCH = 100
patience = 6
test_results = []
best_model_path = None
best_val_performance = float('-inf')
best_val_epoch = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):

    net = AFFTrans().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    log_message = f"第 {fold + 1} 折 / {kf.n_splits} 折"
    print(log_message)
    to_log(log_message)

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate1)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate1)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    current_best_val_performance = float('-inf')

    model_dir = './Task2Model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()

        for label, base, life, ecg in train_loader:
            label = label.to(device)
            base = base.to(device)
            life = life.to(device)
            ecg = ecg.to(device)


            _,_ = net(base, life, ecg)

            outputs = net.train_model_label(base,life,ecg)
            loss = criterion_model(outputs, label)


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
                       val_performance[3], val_performance[4], val_performance[5]) + \
                   '=' * 60

        print(results)
        to_log(results)

        current_val_loss = val_performance[0]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            early_stopping_counter = 0
            filename = f'best_model_fold_{fold + 1}_acc_{val_performance[0]:.4f}.pt'
            save_path_pt = os.path.join(model_dir, filename)
            torch.save(net.state_dict(), save_path_pt)

            if val_performance[0] > best_val_performance:
                best_val_performance = val_performance[0]
                best_model_path = save_path_pt
                best_val_epoch = epoch + 1

        else:
            early_stopping_counter += 1

        current_best_val_performance = max(current_best_val_performance, val_performance[0])

        if early_stopping_counter >= patience:
            print(f"提前停止，忍耐值已达到 {patience}。")
            to_log(f"提前停止，忍耐值已达到 {patience}。")
            break

    best_fold_results = f"第 {fold + 1} 折 当前最佳验证效果 (ACC): {current_best_val_performance:.4f}"
    print(best_fold_results)
    to_log(best_fold_results)

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
        f"MCC 值:          {colored(f'{test_performance[5]:.4f}', 'red')}\n"
    )

    print(results)
    to_log(results)
    test_results.append(test_performance)

for i, result in enumerate(test_results):
    test_result_message = (
        f"第 {i + 1} 折测试结果: [ACC: {result[0]:.4f}, Precision: {result[1]:.4f}, Recall: {result[2]:.4f}, "
        f"F1: {result[3]:.4f}, AUC: {result[4]:.4f}, MCC: {result[5]:.4f}]"
    )
    print(test_result_message)
    to_log(test_result_message)

if best_model_path:
    best_model_message = f"最佳模型保存为: {best_model_path}"
    print(best_model_message)
    to_log(best_model_message)