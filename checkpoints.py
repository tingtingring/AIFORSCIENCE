import torch

checkpoint = torch.load('../Model/best_model_fold_5.pt')
print(checkpoint.keys())  # 查看包含的键