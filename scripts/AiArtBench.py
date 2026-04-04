import os
# import pandas as pd
import torch
#import matplotlib.pyplot as plt
import numpy as np
#import argparse

from datetime import datetime
from torchvision import datasets, transforms

#from torchvision.models import resnet50, ResNet50_Weights
#from torch.utils.data import DataLoader, random_split
#from torch.utils.tensorboard import SummaryWriter


#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Helper functions to handle the creation of the datasets so I don't have to keep calling it in every file.

artBenchTransform = transforms.Compose([
        transforms.Resize((256, 256)), # real images should be this resolution, AI images will need resizing
        transforms.ToTensor(),
        #transforms.Normalize( # resnet50 uses ImageNet normalization values
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)
])


# this takes a while since the full dataset is 180k images. 
def buildArtBenchDataset(testOrTrain):
    print(f"retrieving Artbench {testOrTrain} dataset")
    # ImageFolder is a livesaver here
    #testOrTrain = 'train'
    #dataPath = os.path.join(script_dir, "..", f"Real_AI_SD_LD_Dataset/{testOrTrain}")
    #rootPath = f"/data/"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get the parent directory (go up one level, indicated by '..') and join the sibling folder name.
    dataPath = os.path.join(script_dir, os.pardir, f"data/Real_AI_SD_LD_Dataset/{testOrTrain}")
    
    # normalize the path to remove the '..' component.
    dataPath = os.path.normpath(dataPath)
    AiArtBench = datasets.ImageFolder(root=dataPath, transform=artBenchTransform)
    
    # need binary label mapping. by default InageFolder will make every folder into a class. I dont want that
    # this checks the name of the folder. if it starts with "AI_", it goes in the AI class.
    print("mapping labels")
    ai_class_indices = [
        idx for cls, idx in  AiArtBench.class_to_idx.items()
        if "AI_" in cls
    ]

    print("replacing labels")
    # replace the labels now
    new_samples = []
    for path, original_label in  AiArtBench.samples:
        if original_label in ai_class_indices:
            new_label = 1  # AI
        else:
            new_label = 0  # Real
        new_samples.append((path, new_label))

    print("replacing class values and indices")
    # then replace everything else
    AiArtBench.samples = new_samples
    AiArtBench.targets = [label for _, label in new_samples]
    AiArtBench.classes = ["Real", "AI"]
    AiArtBench.class_to_idx = {"Real": 0, "AI": 1}
    
    #dataLoader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # print("Classes:", image_dataset.classes)
    #print("Class to index mapping:", image_dataset.class_to_idx)
    # print(f"Number of images found: {len(image_dataset)}")
    
    # for images, labels in dataloader:
    #    print("Batch images shape:", images.shape)
    #    print("Batch labels:", labels)
    #    break
    print("done setting up dataset")
    return AiArtBench
