import torch
import torch.nn as nn

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    G = torch.bmm(features, features.permute(0, 2, 1))
    return G.div(ch * h * w)

def add_mean_std(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    m = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)

def subtract_mean_std(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    m = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor - m) / s
    return tensor