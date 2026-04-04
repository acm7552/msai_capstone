import torch
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset, Subset
from torchvision import transforms
import numpy as np

from ARIAdataset import buildARIA
from AiArtBench import buildArtBenchDataset as buildArtBench
from bit_patch import bit_patch as bit_patch_process


# -------------------- Preprocessing --------------------
def create_preprocessing_pipeline(options):
    if options.isPatch:
        transform_func = transforms.Lambda(
            lambda img: bit_patch_process(
                img, options.img_height, options.bit_mode,
                options.patch_size, options.patch_mode
            )
        )
    else:
        transform_func = transforms.Resize((options.img_height, options.img_height))

    return transforms.Compose([
        transform_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def apply_preprocessing(image, options):
    pipeline = create_preprocessing_pipeline(options)
    return pipeline(image)


# -------------------- Wrapper Dataset --------------------
class LOTAWrapperDataset(Dataset):
    def __init__(self, base_dataset, options):
        self.dataset = base_dataset
        self.options = options

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # If tensor already (ArtBench / ARIA), avoid re-ToTensor
        if isinstance(img, torch.Tensor):
            if img.shape[-1] != self.options.img_height:
                img = transforms.functional.resize(
                    img, (self.options.img_height, self.options.img_height)
                )
        else:
            img = apply_preprocessing(img, self.options)

        return img, torch.tensor(label, dtype=torch.float32)


# -------------------- Helper: Split Dataset --------------------
def split_by_label(dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))

    labels = np.array(base.targets)[indices]

    real_idx = np.where(labels == 0)[0]
    ai_idx = np.where(labels == 1)[0]

    return real_idx, ai_idx


# -------------------- Training Loader --------------------
def create_training_loader(options):

    datasets = []

    # ARIA
    aria = buildARIA()
    
    trainSize = int(len(aria) * 0.8)
    testSize = len(aria) - trainSize
        
    # ARIA needs the seed to ensure reproducible train test split.
    aria, testAria = random_split(
        aria, [trainSize, testSize], generator=torch.manual_seed(42) 
    )

    aria = LOTAWrapperDataset(aria, options)
    datasets.append(aria)
    print("Train dataset: ARIA")

    # ArtBench (train split)
    artbench = buildArtBench("train")
    
    artbench = LOTAWrapperDataset(artbench, options)
    datasets.append(artbench)
    print("Train dataset: ArtBench")

    combined_dataset = ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=options.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


# -------------------- Validation Loader --------------------
def setup_validation_loaders(options):
    loaders = []

    # ---------- ARIA ----------
    aria = buildARIA()
    
    trainSize = int(len(aria) * 0.8)
    testSize = len(aria) - trainSize
        
    # ARIA needs the seed to ensure reproducible train test split.
    ariaTrain, ariaTest = random_split(
        aria, [trainSize, testSize], generator=torch.manual_seed(42) 
    )


    real_idx, ai_idx = split_by_label(ariaTest)

    aria_real = Subset(ariaTest, real_idx)
    aria_ai = Subset(ariaTest, ai_idx)

    aria_real_wrap = LOTAWrapperDataset(aria_real, options)
    aria_ai_wrap = LOTAWrapperDataset(aria_ai, options)

    

    loaders.append({
        'name': 'ARIA',
        'val_ai_loader': DataLoader(
            aria_ai_wrap,
            batch_size=options.val_batchsize,
            shuffle=False,
            num_workers=4
        ),
        'ai_size': len(aria_ai),
        'val_nature_loader': DataLoader(
            aria_real_wrap,
            batch_size=options.val_batchsize,
            shuffle=False,
            num_workers=4
        ),
        'nature_size': len(aria_real),
    })

    # ---------- ArtBench ----------
    artbench = buildArtBench("test")
    
    real_idx, ai_idx = split_by_label(artbench)

    art_real = Subset(artbench, real_idx)
    art_ai = Subset(artbench, ai_idx)

    artbench_real_wrap = LOTAWrapperDataset(art_real, options)
    artbench_ai_wrap   = LOTAWrapperDataset(art_ai, options)

    loaders.append({
        'name': 'ArtBench',
        'val_ai_loader': DataLoader(
            artbench_real_wrap,
            batch_size=options.val_batchsize,
            shuffle=False,
            num_workers=4
        ),
        'ai_size': len(art_ai),
        'val_nature_loader': DataLoader(
            artbench_ai_wrap,
            batch_size=options.val_batchsize,
            shuffle=False,
            num_workers=4
        ),
        'nature_size': len(art_real),
    })

    return loaders


# -------------------- API --------------------
def get_loader(opt):
    return create_training_loader(opt)


def get_val_loader(opt):
    return setup_validation_loaders(opt)


def get_single_loader(opt, image_dir, is_real):
    raise NotImplementedError("Not used in this setup")