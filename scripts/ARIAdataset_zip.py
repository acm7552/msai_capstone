import os
from zipfile import ZipFile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms


# default transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -------------------- Dataset for images inside a zip --------------------
class ZipImageDataset(Dataset):
    def __init__(self, zip_path, label, transform=None, extensions=(".png", ".jpg", ".jpeg")):
        self.zip_path = zip_path
        self.transform = transform
        self.extensions = extensions
        self.label = label

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]

        with ZipFile(zip_path) as z:
            self.image_names = [
                name for name in z.namelist()
                if name.lower().endswith(self.extensions)
                and name.startswith(f"{zip_name}/")
                and not name.startswith("__MACOSX/")
            ]

        self.targets = [label] * len(self.image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        with ZipFile(self.zip_path) as z:
            with z.open(image_name) as f:
                img = Image.open(f).convert("RGB")
                if self.transform:
                    img = self.transform(img)
        return img, self.label


# -------------------- Dataset for folders --------------------
class FolderImageDataset(Dataset):
    def __init__(self, folder_path, label, transform=None, extensions=(".png", ".jpg", ".jpeg")):
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
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


# -------------------- Build ARIA --------------------
def buildARIA():
    print("loading ARIA images")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figPath    = os.path.join(script_dir, os.pardir, "data/ARIA_dataset")
    root_dir   = os.path.normpath(figPath)

    all_datasets = []
    all_targets = []   # global label index

    # real
    real_zip_folder = os.path.join(root_dir, "Real")
    for zip_file in os.listdir(real_zip_folder):
        if zip_file.lower().endswith(".zip"):
            zip_path = os.path.join(real_zip_folder, zip_file)
            ds = ZipImageDataset(zip_path, label=0, transform=transform)
            all_datasets.append(ds)
            all_targets.extend(ds.targets)

    # ai
    ai_sources = ["DALL-E", "Midjourney", "DreamStudio", "StarryAI"]

    for source in ai_sources:
        for method in ["IT2I", "T2I"]:
            method_folder = os.path.join(root_dir, source, method)
            if not os.path.exists(method_folder):
                continue

            for zip_file in os.listdir(method_folder):
                if zip_file.lower().endswith(".zip"):
                    zip_path = os.path.join(method_folder, zip_file)
                    ds = ZipImageDataset(zip_path, label=1, transform=transform)
                    all_datasets.append(ds)
                    all_targets.extend(ds.targets)

    # concatenate
    full_dataset = ConcatDataset(all_datasets)

    # attach  labels
    full_dataset.targets = all_targets

    print("done loading ARIA")
    return full_dataset


if __name__ == "__main__":
    aria = buildARIA()

    print("Total images:", len(aria))

    dataloader = DataLoader(aria, batch_size=64, shuffle=True, num_workers=4)

    for imgs, labels in dataloader:
        print("Batch images shape:", imgs.shape)
        print("Batch labels:", labels)
        break