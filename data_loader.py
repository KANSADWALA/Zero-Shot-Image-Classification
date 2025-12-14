
import os
import concurrent.futures
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from trials.config import MAX_WORKERS

class ImageDataset(Dataset):
    """
    Optimized PyTorch Dataset for loading and processing images with optional corruptions.
    
    Handles loading images from disk, applying augmentations/corruptions, and providing
    structured access to image data with corresponding labels. Includes error handling
    for failed image loads.
    
    Attributes:
        image_list (list): List of dicts with keys 'path', 'class', 'class_idx'
        corruption_fn (callable, optional): Function to apply image corruption
        transform (callable, optional): Albumentations or torchvision transforms
    """
    def __init__(self, image_list, corruption_fn=None, transform=None):
        """
        Initialize the ImageDataset.
        
        Args:
            image_list (list): List of image metadata dictionaries containing:
                - 'path' (str): File path to the image
                - 'class' (str): Class name
                - 'class_idx' (int): Numeric class index
            corruption_fn (callable, optional): Function to apply corruptions to images.
                Should accept image as numpy array and return dict with 'image' key.
            transform (callable, optional): Additional transforms to apply (torchvision/albumentations)
        """
        self.image_list = image_list
        self.corruption_fn = corruption_fn
        self.transform = transform
        
    def __len__(self):
        """
        Return the total number of images in the dataset.
        
        Returns:
            int: Number of images in image_list
        """
        return len(self.image_list)
    
    def __getitem__(self, idx):
        """
        Load and return a single image with its label.
        
        Loads image from disk, converts to RGB, applies optional corruption,
        and returns as dictionary. Includes fallback for failed loads.
        
        Args:
            idx (int): Index of the image to load
        
        Returns:
            dict: Contains keys:
                - 'image': PIL Image object (RGB)
                - 'class_idx': Integer class index
                - 'class_name': String class name
        """
        item = self.image_list[idx]
        
        # Load image
        try:
            img = Image.open(item["path"]).convert("RGB")
            img_np = np.array(img)
            
            # Apply corruption if specified
            if self.corruption_fn:
                img_np = self.corruption_fn(image=img_np)["image"]
            
            img = Image.fromarray(img_np)
            
            return {
                'image': img,
                'class_idx': item["class_idx"],
                'class_name': item["class"]
            }
        except Exception as e:
            print(f"Error loading image {item['path']}: {e}")
            # Return a dummy black image if loading fails
            dummy_img = Image.new('RGB', (224, 224), color='black')
            return {
                'image': dummy_img,
                'class_idx': item["class_idx"],
                'class_name': item["class"]
            }

def load_images_parallel(data_dir, max_workers=MAX_WORKERS):
    """
    Load dataset from disk structure using parallel processing.
    
    Scans a directory structure organized as data_dir/class_name/image_files
    and loads metadata for all images. Uses ThreadPoolExecutor for parallel
    I/O operations to speed up directory traversal.
    
    Args:
        data_dir (str): Root directory containing class subdirectories
        max_workers (int): Number of parallel threads for I/O operations
    
    Returns:
        tuple: (all_images, classes) where:
            - all_images (list): List of dicts with 'path', 'class', 'class_idx' keys
            - classes (list): Sorted list of unique class names
    
    Example:
        images, classes = load_images_parallel('data/train', max_workers=8)
    """
    def load_class_images(class_name):
        """
        Helper function to load all images from a single class directory.
        
        Args:
            class_name (str): Name of the class directory
        
        Returns:
            tuple: (class_name, image_filenames) where image_filenames is a list
        """
        class_dir = os.path.join(data_dir, class_name)
        images = []
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(img_name)
        return class_name, images
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Parallel loading of class information
    all_images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_class = {executor.submit(load_class_images, class_name): class_name for class_name in classes}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_class), total=len(classes), desc="Loading classes"):
            class_name, img_names = future.result()
            class_idx = classes.index(class_name)
            
            for img_name in img_names:
                all_images.append({
                    'path': os.path.join(data_dir, class_name, img_name),
                    'class': class_name,
                    'class_idx': class_idx
                })
    
    return all_images, classes