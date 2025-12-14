import os
import shutil    
import concurrent.futures    
from tqdm import tqdm           
import random
from torchvision import datasets
from trials.config import MAX_WORKERS, TRAIN_DIR, TEST_DIR, VAL_DIR

def setup_cifar100():
    """
    Download and prepare CIFAR-100 dataset organized by class directories.
    
    Downloads CIFAR-100 from torchvision, extracts images into class-organized
    directory structure (train/test/val splits), and creates validation split
    from training data (20% of training images). Uses parallel processing for
    efficiency. Skips download if dataset already exists.
    
    Returns:
        list: Sorted list of 100 class names
    
    Creates directory structure:
        cifar-100-python_processed/
        ├── train/
        │   ├── apple/
        │   ├── aquarium_fish/
        │   └── ...
        ├── test/
        │   └── ...
        └── val/
            └── ...
    """

    # CHECK IF DATASET ALREADY EXISTS
    if (os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR) and os.path.exists(VAL_DIR) 
        and len(os.listdir(TRAIN_DIR)) > 0 and len(os.listdir(TEST_DIR)) > 0 and len(os.listdir(VAL_DIR)) > 0):
        print("Dataset already exists, skipping setup...")
        # Get class names from existing structure
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=False)
        return train_dataset.classes
    
    # Download CIFAR-100
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
    
    # Get class names
    classes = train_dataset.classes
    
    # Create directory structure in parallel
    def create_class_dirs(split_dir):
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(create_class_dirs, [TRAIN_DIR, TEST_DIR, VAL_DIR])
    
    # Save images with parallel processing
    def save_dataset_split(dataset_info):
        dataset, split_dir, split_name = dataset_info
        
        def save_batch(batch_info):
            start_idx, end_idx = batch_info
            for idx in range(start_idx, end_idx):
                if idx >= len(dataset):
                    break
                    
                image, label = dataset[idx]
                class_name = classes[label]
                image_path = os.path.join(split_dir, class_name, f"{idx:05d}.png")
                image.save(image_path)
        
        # Process in chunks
        chunk_size = 1000
        batch_ranges = [(i, min(i + chunk_size, len(dataset))) 
                       for i in range(0, len(dataset), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(tqdm(executor.map(save_batch, batch_ranges), 
                     desc=f"Saving {split_name}", total=len(batch_ranges)))
    
    # Save train and test splits
    dataset_splits = [
        (train_dataset, TRAIN_DIR, "train"),
        (test_dataset, TEST_DIR, "test")
    ]
    
    for dataset_info in dataset_splits:
        save_dataset_split(dataset_info)
    
    # Create validation split from training data
    print("Creating validation split...")
    
    def move_validation_images(class_name):
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        
        if not os.path.exists(train_class_dir):
            return
            
        train_images = [f for f in os.listdir(train_class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        val_count = max(1, len(train_images) // 5)
        val_images = random.sample(train_images, min(val_count, len(train_images)))
        
        for img_name in val_images:
            try:
                shutil.move(
                    os.path.join(train_class_dir, img_name),
                    os.path.join(val_class_dir, img_name)
                )
            except Exception as e:
                print(f"Error moving {img_name}: {e}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(move_validation_images, classes), 
                 desc="Creating validation split", total=len(classes)))
    
    return classes