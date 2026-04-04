import torch
import clip
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ways to load the needed datasets
from ARIAdataset import buildARIA
from AiArtBench import buildArtBenchDataset as buildArtBench

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def zeroshot_CLIP(args):
    # load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # load dataset(s)

    if args.dataset.lower() == "aria":

        print("Loading ARIA dataset")
        
        aria = buildARIA()
        
        trainSize = int(len(aria) * 0.8)
        testSize = len(aria) - trainSize
        
        # ARIA needs the seed to ensure reproducible train test split.
        dataset, testDataset = random_split(
            aria, [trainSize, testSize], generator=torch.manual_seed(args.seed) 
        )
        dataset = testDataset
    else: 
        # ArtBench is already split. It doesn't require the seed here.
        print("loading ArtBench")
        dataset = buildArtBench("test")
    
    # replace transform with CLIP transform loaded earlier
    dataset.transform = preprocess
    
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    print("Dataset size:", len(dataset))
    
    # zero-shot prompts
    prompts = [
        "human created artwork",
        "AI generated artwork"
    ]
    
    text_tokens = clip.tokenize(prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # inference
    all_labels = []
    all_preds = []
    
    print("Running zero-shot inference")
    
    with torch.no_grad():
        for images, labels in loader:
    
            images = images.to(device)
    
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
    
            similarity = image_features @ text_features.T
            preds = similarity.argmax(dim=1)
    
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\nZero-shot CLIP results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\nClassification Report")
    print(classification_report(all_labels, all_preds, target_names=["Real", "AI"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, help="(int) random seed", type=int)
    parser.add_argument("--dataset", default="artbench", help="(string) the dataset to use (ArtBench, or Aria)", type=str)
    args = parser.parse_args()
    
    zeroshot_CLIP(args)
    print("all done")
    
    