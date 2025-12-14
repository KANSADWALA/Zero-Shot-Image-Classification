import os
import torch
import multiprocessing as mp
import albumentations as A

# -------------------
# CONFIG
# ------------------- 

# For CIFAR-100 dataset
DATASET_NAME = "cifar-100-python"  
OUTPUT_DIR = f"{DATASET_NAME}_processed"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val") 
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Create directories
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)

RESULTS_DIR = "results_zero_shot_research"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

# Performance optimization settings
BATCH_SIZE = 16  # Increased batch size for GPU efficiency
MAX_WORKERS = min(12, mp.cpu_count())  # Threading for I/O operations
CHUNK_SIZE = 100  # Process images in chunks
PREFETCH_FACTOR = 2  # DataLoader prefetch factor

# Prompt templates from PDF requirements
PROMPT_TEMPLATES = [
    # Default
    "a photo of a {}",
    
    # Visual Context 
    "a blurry image of a {}",
    "a microscopic image of a {}",
    
    # Descriptive Adjective 
    "a high resolution image of a {}",
    "a low-light photo of a {}",
    "a close-up photo of a {}",
    "a cropped image of a {}",
    "a bright photo of a {}",
    "a black and white photo of a {}",
    
    # Domain-Specific - adapt based on your dataset
    "a satellite image of {}",  # for EuroSAT
    "an X-ray image of {}",     # for medical datasets
    "a food photo of {}",       # for Food-101
    
    # Additional variations
    "a painting of a {}",
]

# Corruption types from PDF
CORRUPTIONS = {
    # Blur
    "gaussian_blur": lambda s: A.GaussianBlur(blur_limit=(1 + s*2, 3 + s*2), p=1.0),
    "motion_blur": lambda s: A.MotionBlur(blur_limit=3 + s*2, p=1.0),
    
    # Noise
    "gaussian_noise": lambda s: A.GaussNoise(var_limit=(10*(s**2), 30*(s**2)), p=1.0),
    "salt_pepper_noise": lambda s: A.Compose([
        A.CoarseDropout(max_holes=s*10, max_height=1, max_width=1, p=1.0, fill_value=0),
        A.CoarseDropout(max_holes=s*10, max_height=1, max_width=1, p=1.0, fill_value=255)
    ]),
    
    # Occlusion - Fixed the cutout corruption
    "random_occlusion": lambda s: A.CoarseDropout(max_holes=1+s, max_height=16*s, max_width=16*s, p=1.0),
    "cutout": lambda s: A.CoarseDropout(max_holes=s, max_h_size=16*s, max_w_size=16*s, p=1.0, fill_value=0),  # Fixed: using CoarseDropout instead of Cutout
    
    # Lighting
    "brightness": lambda s: A.RandomBrightnessContrast(brightness_limit=0.1*s, contrast_limit=0.0, p=1.0),
    "contrast": lambda s: A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.1*s, p=1.0),
    
    # Compression
    "jpeg": lambda s: A.ImageCompression(quality_lower=100 - 15*s, quality_upper=100 - 10*s, p=1.0),
}

SEVERITY_LEVELS = [1, 3, 5] #SEVERITY_LEVELS = [1, 2, 3, 4, 5]  # Using ImageNet-C levels as specified

# -------------------
# CUSTOM COLLATE FUNCTION 
# -------------------
def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL Images in DataLoader batches.
    
    Converts a batch of items from the dataset into dictionaries while keeping
    images as a list of PIL Image objects instead of converting to tensors.
    
    Args:
        batch (list): List of items from the dataset, each containing 'image',
                     'class_idx', and 'class_name' keys.
    
    Returns:
        dict: Dictionary with keys:
            - 'image': List of PIL Image objects
            - 'class_idx': Tensor of shape (batch_size,) with class indices
            - 'class_name': List of class name strings
    """
    images = [item['image'] for item in batch]
    class_indices = torch.tensor([item['class_idx'] for item in batch])
    class_names = [item['class_name'] for item in batch]
    
    return {
        'image': images,  # Keep as list of PIL Images
        'class_idx': class_indices,
        'class_name': class_names
    }