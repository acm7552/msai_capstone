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

# code freely available at https://github.com/IBM/RIGID/blob/main/RIGID.ipynb

# ways to load the needed datasets
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
    """
    Calculate AUC and FPR95 for multiple OOD datasets against ID dataset.
    
    Args:
        similarities (list): List of similarity arrays, first one is ID dataset
        datasets (list): List of dataset names
        
    Returns:
        tuple: (average_auc, average_fpr95)
    """
    if len(similarities) != len(datasets):
        raise ValueError("Number of similarities arrays must match number of dataset names")
    
    if len(similarities) < 2:
        raise ValueError("At least 2 datasets (ID and OOD) are required")

    similarities = np.array(similarities, dtype=object)  # Use object dtype for arrays of different lengths
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
    """
    Calculate Average Precision for multiple OOD datasets against ID dataset.
    
    Args:
        similarities (list): List of similarity arrays, first one is ID dataset
        datasets (list): List of dataset names
        
    Returns:
        float: average AP score
    """
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


def calculate_auc_metrics(id_conf, ood_conf):
    """
    Calculate AUROC and FPR at 95% TPR for binary classification.
    
    Args:
        id_conf (np.ndarray): Confidence scores for ID (in-distribution) samples
        ood_conf (np.ndarray): Confidence scores for OOD (out-of-distribution) samples
        
    Returns:
        tuple: (auroc, fpr_at_95_tpr)
    """
    # Combine predictions and create labels
    all_conf = np.concatenate([id_conf, ood_conf])
    # ID samples are positive (1), OOD samples are negative (0)
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    
    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)
    
    # Calculate AUROC
    auroc = metrics.auc(fpr, tpr)
    
    # Calculate FPR at 95% TPR
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    if np.any(valid_indices):
        fpr_at_95 = fpr[np.argmax(valid_indices)]
    else:
        fpr_at_95 = fpr[-1]
        print(f"Warning: 95% TPR not achievable. Max TPR: {tpr[-1]:.3f}")
    
    return auroc, fpr_at_95


def calculate_average_precision(id_predictions, ood_predictions):

    # Combine predictions and create labels
    all_predictions = np.concatenate([id_predictions, ood_predictions])
    # ID samples are positive (1), OOD samples are negative (0)
    labels = np.concatenate([np.ones(len(id_predictions)), np.zeros(len(ood_predictions))])
    
    # Calculate Average Precision
    average_precision = metrics.average_precision_score(labels, all_predictions)
    
    return average_precision


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
        sim = self.calculate_sim(data)
        return sim


transform_RIGID = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])

# Test Datasets 
# You can test multiple datasets
# But make sure the real image dataset is in the first one to facilitate the calculation of AUC or AP.
#test_datasets = ['Imagenet', 'Imagenet256-ADM', 'Imagenet256-ADMG', 'Imagenet256-LDM', 'Imagenet256-DiT-XL-2', 
#                     'Imagenet256-BigGAN', 'Imagenet256-GigaGAN', 'Imagenet256-StyleGAN-XL', 'Imagenet256-RQ-Transformer', 'Imagenet256-Mask-GIT']

test_datasets = ["AIArtBench", "AIArtBench"]
#test_datasets = ["ARIA", "AIArtBench"]
noise_intensity = 0.05
batch_size = 256

rigid_detector = RIGID_Detector(lamb=noise_intensity)

with torch.no_grad():

        sim_datasets = []
        for dataset in test_datasets:

            
            #dataset_folder = datasets.ImageFolder(root=f'./gen_images/{dataset}',  transform=transform_RIGID)
            if dataset == "AIArtBench": 
                
                artBench = buildArtBench("test")
                artBench.transform = transform_RIGID
                data_loader = DataLoader(artBench, batch_size=batch_size, shuffle=True, num_workers=2)

            else:
                aria = buildARIA()
                aria.transform = transform_RIGID
                trainSize = int(len(aria) * 0.8)
                testSize = len(aria) - trainSize
                
                # ARIA needs the seed to ensure reproducible train test split.
                ariaTrain, ariaTest = random_split(
                    aria, [trainSize, testSize] 
                )
                data_loader = DataLoader(ariaTest, batch_size=batch_size, shuffle=True, num_workers=2)
                
            sim_feat = []
            total_num = 0 
            for i, (samples, _) in enumerate(data_loader):
                
                samples = samples.cuda()
                samples_num = len(samples)
                total_num += samples_num

                sim = rigid_detector.calculate_sim(samples)
                sim_feat.append(sim)

                desired_total = 2000
                #if total_num % 500 == 0: print(f"{total_num}/{desired_total}")
                    
                if total_num >= desired_total: break
            
            sim_feat = torch.cat(sim_feat, dim=0)
            print(f'{dataset}, Image number: {sim_feat.shape[0]}, similarity is {sim_feat.mean().item()}')

            sim_datasets.append(sim_feat.cpu().numpy())
        
        print("Detection Results AUC:")
        sim_auc(sim_datasets, test_datasets)

        print("Detection Results AP:")
        sim_ap(sim_datasets, test_datasets)
            