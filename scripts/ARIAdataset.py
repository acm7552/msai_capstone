import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms


# default
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class FolderImageDataset(Dataset):
    def __init__(self, folder_path, label, transform=None, extensions=(".png", ".jpg", ".jpeg")):
        self.folder_path = folder_path
        self.transform = transform
        self.label = label
        self.image_paths = []

        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, f))

        self.targets = [label] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.label


def buildARIA():
    print("loading ARIA images")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "data/ARIA_dataset")
    )

    all_datasets = []
    all_targets = []

    # real images
    real_folder = os.path.join(root_dir, "Real")

    real_ds = FolderImageDataset(
        real_folder,
        label=0,
        transform=transform
    )

    all_datasets.append(real_ds)
    all_targets.extend(real_ds.targets)

    # ai images
    ai_sources = ["DALL-E", "Midjourney", "DreamStudio", "StarryAI"]

    for source in ai_sources:
        for method in ["IT2I", "T2I"]:
            method_folder = os.path.join(root_dir, source, method)

            if not os.path.exists(method_folder):
                continue

            ai_ds = FolderImageDataset(
                method_folder,
                label=1,
                transform=transform
            )

            all_datasets.append(ai_ds)
            all_targets.extend(ai_ds.targets)

    # combine into one dataset
    full_dataset = ConcatDataset(all_datasets)
    full_dataset.targets = all_targets
    full_dataset.datasets = all_datasets

    print("done loading ARIA")
    return full_dataset



if __name__ == "__main__":
    aria = buildARIA()

    print("Total images:", len(aria))

    loader = DataLoader(aria, batch_size=64, shuffle=True, num_workers=4)

    for imgs, labels in loader:
        print("Batch images shape:", imgs.shape)
        print("Batch labels:", labels)
        break