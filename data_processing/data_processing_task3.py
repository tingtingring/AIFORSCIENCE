import os
import xml.etree.ElementTree as ET
import torch
import pandas as pd
from tqdm import tqdm

def normalize_signal(signal):
    """对信号进行归一化处理"""
    if len(signal) == 0:
        return signal
    return [x+5000 for x in signal]

def ECG_process_and_save(path, data_type='median', output_path='../dataset/train/ecg_data_without_norm'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"指定路径不存在: {path}")

    if data_type not in ['strip', 'median']:
        raise ValueError("data_type must be either 'strip' or 'median'.")

    ecg_list = []
    files = [f for f in os.listdir(path) if f.endswith(".xml")]

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(files, desc="Processing ECG XML files", colour="blue"):
        file_path = os.path.join(path, filename)

        try:
            # 解析XML文件
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            print(f"解析文件时出错: {file_path}, 错误信息: {e}")
            continue

        ecg_data = {lead: [] for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]}

        # 根据指定的数据类型选择读取的节点
        data_nodes = root.findall(".//StripData/WaveformData") if data_type == 'strip' else root.findall(".//MedianSamples/WaveformData")

        for waveform in data_nodes:
            lead_name = waveform.attrib.get('lead')
            if lead_name in ecg_data:
                try:
                    signal = list(map(int, waveform.text.strip().split(',')))
                    if signal:  # 确保信号数据不为空
                        # 进行归一化
                        normalized_signal = normalize_signal(signal)
                        ecg_data[lead_name].extend(normalized_signal)
                except ValueError as ve:
                    print(f"信号数据转换失败: {filename}, 错误信息: {ve}")

        # 检查每个导联是否有数据
        leads_data = [ecg_data[lead] for lead in ecg_data]
        if all(len(data) > 0 for data in leads_data):  # 确保每个导联的数据不为空
            ecg_tensor = torch.tensor(leads_data)
            ecg_list.append(ecg_tensor)

            # 保存为CSV文件，使用XML文件名
            base_filename = os.path.splitext(filename)[0]  # 获取不带扩展名的文件名
            output_file = os.path.join(output_path, f"{base_filename}.csv")
            df = pd.DataFrame(ecg_tensor.numpy().T, columns=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
            df.to_csv(output_file, index=False)
            print(f"保存文件: {output_file}")

    # 最终将所有ECG数据组合成3D张量，形状为 [文件数, 12, 心电信号序列长度]
    if ecg_list:  # 确保至少有一个有效的张量
        ecg_tensor = torch.stack(ecg_list)
        return ecg_tensor
    else:
        print("没有有效的ECG数据被处理。")
        return None

# 示例调用
train_path = "../dataset/train/ecg/"
ecg_data_tensor = ECG_process_and_save(train_path)
if ecg_data_tensor is not None:
    print("最终张量形状:", ecg_data_tensor.shape)

if ecg_data_tensor is not None:
    max_value = torch.max(ecg_data_tensor)
    print("ECG数据张量中的最大值:", max_value.item())
    min_value = torch.min(ecg_data_tensor)
    print("ECG数据张量中的最小值:", min_value.item())