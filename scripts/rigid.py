import numpy as np 
import random
import os
import sys
from sklearn import metrics

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF

from ARIAdataset import buildARIA
from AiArtBench import buildArtBenchDataset as buildArtBench


seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def sim_auc(similarities, datasets):
    if len(similarities) != len(datasets):
        raise ValueError("Number of similarities arrays must match number of dataset names")

    if len(similarities) < 2:
        raise ValueError("At least 2 datasets (ID and OOD) are required")

    similarities = np.array(similarities, dtype=object)
    id_confi = similarities[0]

    auc_scores = []
    fpr_scores = []

    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        auroc, fpr_95 = calculate_auc_metrics(id_confi, ood_confi)
        auc_scores.append(auroc)
        fpr_scores.append(fpr_95)
        print(f"Dataset: {dataset:<25} | AUC: {auroc:.4f} | FPR95: {fpr_95:.4f}")
    
    avg_auc = np.mean(auc_scores)
    avg_fpr = np.mean(fpr_scores)
    
    print("-" * 60)
    print(f"Average AUC: {avg_auc:.4f} | Average FPR95: {avg_fpr:.4f}")
    
    return avg_auc, avg_fpr


def sim_ap(similarities, datasets):
    if len(similarities) != len(datasets):
        raise ValueError("Number of similarities arrays must match number of dataset names")

    if len(similarities) < 2:
        raise ValueError("At least 2 datasets (ID and OOD) are required")

    similarities = np.array(similarities, dtype=object)
    id_confi = similarities[0]

    ap_scores = []

    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        aver_p = calculate_average_precision(id_confi, ood_confi)
        ap_scores.append(aver_p)
        print(f"Dataset: {dataset:<25} | AP: {aver_p:.4f}")
    
    avg_ap = np.mean(ap_scores)
    print("-" * 40)
    print(f"Average AP: {avg_ap:.4f}")
    
    return avg_ap


# =========================
# ✅ ADDED: ACCURACY SUPPORT
# =========================

def calculate_accuracy(id_scores, ood_scores):
    all_scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])

    fpr, tpr, thresholds = metrics.roc_curve(labels, all_scores)

    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]

    preds = (all_scores >= best_thresh).astype(int)

    acc = metrics.accuracy_score(labels, preds)

    return acc


def sim_acc(similarities, datasets):
    similarities = np.array(similarities, dtype=object)
    id_scores = similarities[0]

    acc_scores = []

    for ood_scores, dataset in zip(similarities[1:], datasets[1:]):
        acc = calculate_accuracy(id_scores, ood_scores)
        acc_scores.append(acc)
        print(f"Dataset: {dataset:<25} | Accuracy: {acc:.4f}")
    
    avg_acc = np.mean(acc_scores)
    print("-" * 40)
    print(f"Average Accuracy: {avg_acc:.4f}")
    
    return avg_acc


def calculate_auc_metrics(id_conf, ood_conf):
    all_conf = np.concatenate([id_conf, ood_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)
    auroc = metrics.auc(fpr, tpr)
    
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    if np.any(valid_indices):
        fpr_at_95 = fpr[np.argmax(valid_indices)]
    else:
        fpr_at_95 = fpr[-1]
    
    return auroc, fpr_at_95


def calculate_average_precision(id_predictions, ood_predictions):
    all_predictions = np.concatenate([id_predictions, ood_predictions])
    labels = np.concatenate([np.ones(len(id_predictions)), np.zeros(len(ood_predictions))])
    return metrics.average_precision_score(labels, all_predictions)


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class RIGID_Detector():
    def __init__(self, lamb=0.05, percentile=5):
        self.lamb = lamb
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
        self.model.eval()
        
    @torch.no_grad()
    def calculate_sim(self, data):
        features = self.model(data)
        noise = torch.randn_like(data).to(data.device)
        trans_data = data + noise * self.lamb
        trans_features = self.model(trans_data)
        sim_feat = F.cosine_similarity(features, trans_features, dim=-1)
        return sim_feat

    @torch.no_grad()
    def detect(self, data):
        return self.calculate_sim(data)


transform_RIGID = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
])


test_datasets = ["AIArtBench-train", "AIArtBench-test"]
noise_intensity = 0.05
batch_size = 256

rigid_detector = RIGID_Detector(lamb=noise_intensity)

with torch.no_grad():

    sim_datasets = []

    for dataset in test_datasets:

        if dataset == "AIArtBench-test": 
            artBench = buildArtBench("test")
            artBench.transform = transform_RIGID
            data_loader = DataLoader(artBench, batch_size=batch_size, shuffle=True, num_workers=2)
        elif dataset == "AIArtBench-train": 
            artBench = buildArtBench("train")
            artBench.transform = transform_RIGID
            data_loader = DataLoader(artBench, batch_size=batch_size, shuffle=True, num_workers=2)

        else:
            aria = buildARIA()
            aria.transform = transform_RIGID
            trainSize = int(len(aria) * 0.8)
            testSize = len(aria) - trainSize
            
            ariaTrain, ariaTest = random_split(aria, [trainSize, testSize])
            data_loader = DataLoader(ariaTest, batch_size=batch_size, shuffle=True, num_workers=2)
        
        sim_feat = []
        total_num = 0 

        for i, (samples, _) in enumerate(data_loader):
            samples = samples.cuda()
            total_num += len(samples)

            sim = rigid_detector.calculate_sim(samples)
            sim_feat.append(sim)

            if total_num >= 500:
                break
        
        sim_feat = torch.cat(sim_feat, dim=0)
        print(f'{dataset}, Image number: {sim_feat.shape[0]}, similarity is {sim_feat.mean().item()}')

        sim_datasets.append(sim_feat.cpu().numpy())

    print("Detection Results AUC:")
    sim_auc(sim_datasets, test_datasets)

    print("Detection Results AP:")
    sim_ap(sim_datasets, test_datasets)

    # =========================
    # ✅ ADDED OUTPUT
    # =========================
    print("Detection Results Accuracy:")
    sim_acc(sim_datasets, test_datasets)