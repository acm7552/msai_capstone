import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from datetime import datetime

#from torchvision import datasets, transforms
#from torchvision.models import resnet50, ResNet50_Weights
#from torch.utils.data import DataLoader, random_split
#from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, roc_curve, auc

import matplotlib.pyplot as plt

# ways to load the needed datasets
# from Datasets.ARIAdataset import buildARIA
# from Datasets.AiArtBench import buildArtBenchDataset as buildArtBench


#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# Uses some example code from PyTorch CNN samples.
# testOrTrain = 'train'
#rootPath = "/home/stu14/s16/acm7552/.cache/kagglehub/datasets/ravidussilva/real-ai-art/versions/5/Real_AI_SD_LD_Dataset/test"
#rootPath = "/home/stu14/s16/acm7552/"

#artBenchTransform = transforms.Compose([
#    transforms.Resize((256, 256)), # real images should be this resolution, AI images will need resizing
#    transforms.ToTensor(),
#    transforms.Normalize( # resnet50 sues ImageNet normalization values
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225]
#    )
#])


# the test set has 30k images. 
# def buildArtBenchDataset():
#    print("retrieving dataset")
    # ImageFolder is a livesaver here
#    AiArtBench = datasets.ImageFolder(root=rootPath, transform=artBenchTransform)
    
    # need binary label mapping. by default InageFolder will make every folder into a class. I dont want that
    # this checks the name of the folder. if it starts with "AI_", it goes in the AI class.
 #   print("mapping labels")
 #   ai_class_indices = [
 #       idx for cls, idx in  AiArtBench.class_to_idx.items()
 #       if "AI_" in cls
 #   ]

  #  print("replacing labels")
  #  # replace the labels now
  #  new_samples = []
  #  for path, original_label in  AiArtBench.samples:
  #      if original_label in ai_class_indices:
  #          new_label = 1  # AI
  #      else:
  #          new_label = 0  # Real
  #      new_samples.append((path, new_label))

  #  print("replacing class values and indices")
    # then replace everything else
  #  AiArtBench.samples = new_samples
  #  AiArtBench.targets = [label for _, label in new_samples]
  #  AiArtBench.classes = ["Real", "AI"]
  #  AiArtBench.class_to_idx = {"Real": 0, "AI": 1}
    
    #dataLoader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # print("Classes:", image_dataset.classes)
    #print("Class to index mapping:", image_dataset.class_to_idx)
    # print(f"Number of images found: {len(image_dataset)}")
    
    # for images, labels in dataloader:
    #    print("Batch images shape:", images.shape)
    #    print("Batch labels:", labels)
    #    break
   # print("done setting up dataset")
   # return AiArtBench
def evaluate_baseline(model, dataset):

    
def evaluate(args):

    # step 1: generate the dataset
    
    testAiArtBench = buildArtBenchDataset()
    batchSize = args.batchSize
  
    testLoader = DataLoader(
        testAiArtBench,
        batch_size=batchSize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    print(f"test set length: {len(testLoader.dataset)}")
    
    # step 2: instantiate model

    # SavedModels/resnet50_20260222_114208_10
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    # recreate classifier head
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # move model to device before loading
    model = model.to(device)
    
    # load saved weights
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # set to evaluation mode
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    # i like having it print every so often so i can see it progress thru the dataset.
    printEvery = 2500

    all_labels = []
    all_preds  = []
    all_probs  = []


    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of AI class
            preds = outputs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if total % printEvery < labels.size(0):
                print(f"Processed {total}/{len(testLoader.dataset)} images")
    
    avg_loss = running_loss / len(testLoader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy : {accuracy:.4%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Real", "AI"]
    ))


    # saves the ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – AI Image Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("resnet50_artbench_ROC.png", dpi=300, bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset", help="(string) the dataset to use (ArtBrain, or Aria)", type=str)
    #parser.add_argument("--epochs", default=10, help="(int) number of training epochs", type=int)
    parser.add_argument("--batchSize", default=64, help="(int, power of 2) batch size", type=int)
    parser.add_argument("model", help="(string) model path", type=str)
    #parser.add_argument("--seed", default=42, help="(int) random seed", type=int)
    #parser.add_argument("optimizer", help="(string) type of optimizer to use", type=str)
    #parser.add_argument("Beta 1",  help="(float) print debug statements", type=float)
    #parser.add_argument("debug",  help="(bool) print debug statements", type=bool)
    args = parser.parse_args()

    evaluate(args)
    print("All done")
    




