import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from datetime import datetime

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

# ways to load the needed datasets
from ARIAdataset import buildARIA
from AiArtBench import buildArtBenchDataset as buildArtBench
#from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt

# Uses some example code from PyTorch CNN samples.

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# getting proper folder paths for saving figures, models, and other outputs
script_dir = os.path.dirname(os.path.abspath(__file__))
    
figPath = os.path.join(script_dir, os.pardir, f"figures")
figPath = os.path.normpath(figPath)

modelPath = os.path.join(script_dir, os.pardir, f"models")
modelPath = os.path.normpath(modelPath)

resultPath = os.path.join(script_dir, os.pardir, f"results")
resultPath = os.path.normpath(resultPath)

#print(figPath)
#print(modelPath)
#print(resultPath)

RESNET50_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize( # resnet50 uses ImageNet normalization values.
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def trainingLoop(args):

    # step 1: generate the dataset
    # specify if its aria, otherwise do artbench
    if args.dataset.lower() == "aria":
        
        aria = buildARIA()
        
        trainSize = int(len(aria) * 0.8)
        testSize = len(aria) - trainSize
        
        # ARIA needs the seed to ensure reproducible train test split.
        dataset, testDataset = random_split(
            aria, [trainSize, testSize], generator=torch.manual_seed(args.seed) 
        )
        
    else: 
        # ArtBench is already split. It doesn't require the seed here.
        dataset = buildArtBench("train")
        testDataset = buildArtBench("test")


    # assign the model's needed transform
    dataset.transform = RESNET50_transform
    testDataset.transform = RESNET50_transform
    
    #aiArtBench = buildArtBenchDataset("train")
    
    # split the training set further into 80% train, 20% validate
    # will use the held out test set during inference
    
    trainSize = int(len(dataset) * 0.8)
    valSize = len(dataset) - trainSize
    
    # perform split with seed
    trainDataset, valDataset = random_split(
        dataset, [trainSize, valSize], generator=torch.manual_seed(args.seed) 
    )
    
    batchSize = args.batchSize
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        #sampler = sampler,
        shuffle=True, #remove this when using samplers
        num_workers=8,       # 4-8 usually saturates CPU without freezing system
        pin_memory=True,     # speeds up transfer to GPU
        prefetch_factor=4,   # number of batches to preload per worker
        persistent_workers=True  # keeps workers alive between epochs
    )

    # validation set
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # test set
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    print(f"train length: {len(trainLoader)*batchSize}")
    print(f"val length: {len(valLoader)*batchSize}")
    
    # step 2: instantiate model

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # replace classification head
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # ensure classifier is trainable
    for param in model.fc.parameters():
        param.requires_grad = True
 
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    # adam optimizer only for unfrozen layers
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    total_images = len(trainLoader.dataset)
    
    def single_epoch(currEpoch):
        
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        
        for i, (inputs, labels) in enumerate(trainLoader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # Zero your gradients for every batch!
            optimizer.zero_grad()
    
            # Make predictions for this batch
            outputs = model(inputs)
    
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
    
            # Adjust learning weights
            optimizer.step()
    
            # Gather data and report
            running_loss += loss.item()
            
            # Print every X batches
            if (i + 1) % 20 == 0:
                last_loss = running_loss / 20
                images_done = (i + 1) * trainLoader.batch_size
                images_done = min(images_done, total_images)  # avoid overshooting
                print(f"({images_done}/{total_images}) Batch {i+1}, Loss: {last_loss:.4f}")
                running_loss = 0.0
        return last_loss

    # nice to have some timing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #writer = SummaryWriter('runs/restnet50_{}'.format(timestamp))
    currEpoch = 0
    
    EPOCHS = args.epochs
    
    best_vloss = 1_000_000.
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = single_epoch(currEpoch) #, writer)
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
    
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valLoader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
    
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
        # log running loss averaged per batch for both training and validation
        #writer.add_scalars('Training vs. Validation Loss',
        #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                epoch_number + 1)
        #writer.flush()
    
        # track best performance, and save the model state dict
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            if args.save:
                print("saving model...")
                model_path = 'resnet50baseline_{}_{}_{}'.format(args.dataset, timestamp, epoch+1)
                final_model_path = f'{modelPath}/{model_path}'
                torch.save(model.state_dict(), final_model_path)
                print("saved model.")


    # finally, perform inference for performance on the test set if desired
    if args.eval: 
        # set to evaluation mode
        model.eval()
        #loss_fn = torch.nn.CrossEntropyLoss()
    
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
        plt.savefig(f"{figPath}/resnet50_{args.dataset}_ROC.png", dpi=300, bbox_inches="tight")
        plt.close()

        # confusion matrix
        print("generating confusion matrix")
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Resnet50 on {args.dataset}")
        plt.colorbar()
        
        classes = ["Real", "AI"]
        ticks = np.arange(len(classes))
        
        plt.xticks(ticks, classes)
        plt.yticks(ticks, classes)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha="center", va="center")
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        plt.savefig(
            f"{figPath}/resnet50_{args.dataset}_confusion_matrix_counts.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # and a normalized one

        print("generating normalized confusion matrix")
        
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6,5))
        plt.imshow(cm_norm, interpolation="nearest")
        plt.title(f"Normalized Resnet50 on {args.dataset}")
        plt.colorbar()
        
        # classes = ["Real", "AI"]
        # ticks = np.arange(len(classes))
        
        plt.xticks(ticks, classes)
        plt.yticks(ticks, classes)
        
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center")
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        plt.savefig(
            f"{figPath}/resnet50_{args.dataset}_confusion_matrix_normalized.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artbench", help="(string) the dataset to use (ArtBench, or Aria)", type=str)
    parser.add_argument("--epochs", default=10, help="(int) number of training epochs", type=int)
    parser.add_argument("--batchSize", default=64, help="(int, power of 2) batch size", type=int)
    parser.add_argument("--seed", default=42, help="(int) random seed", type=int)
    parser.add_argument("--eval", default=True, help="(bool) whether or not to run inference", type=bool)
    parser.add_argument("--save", default=False, help="(bool) whether to save best model", type=bool)
    #parser.add_argument("optimizer", help="(string) type of optimizer to use", type=str)
    #parser.add_argument("Beta 1",  help="(float) print debug statements", type=float)
    #parser.add_argument("debug",  help="(bool) print debug statements", type=bool)
    args = parser.parse_args()

    trainingLoop(args)
    print("All done")
    




