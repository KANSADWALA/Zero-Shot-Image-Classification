import os
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pathlib import Path    
from PIL import Image
import concurrent.futures
import torch
from torch.utils.data import DataLoader 
from trials.config import MAX_WORKERS, PREFETCH_FACTOR, CORRUPTIONS, SEVERITY_LEVELS
from trials.data_loader import ImageDataset
from trials.data_loader import custom_collate_fn


def zero_shot_eval(clipwrap, dataset, classes, templates, topk=(1, 5)):
    """
    Perform zero-shot classification evaluation with multiple prompt templates.
    
    Evaluates CLIP's ability to classify images using text prompts without
    fine-tuning. Tests multiple templates and computes top-k accuracies.
    Uses batch processing for GPU efficiency.
    
    Args:
        clipwrap (CLIPWrapper): Initialized CLIP wrapper instance
        dataset (ImageDataset): Dataset containing images and labels
        classes (list): List of class names
        templates (list): List of prompt templates with {} placeholder for class name
        topk (tuple): Top-k accuracies to compute (default: (1, 5))
    
    Returns:
        pd.DataFrame: Results with columns 'prompt', 'top1', 'top5', 'total_samples'
    
    Example:
        templates = ['a photo of a {}', 'a picture of a {}']
        results_df = zero_shot_eval(clipwrap, dataset, classes, templates)
    """
    results = []
    
    # Create DataLoader for efficient batching
    dataloader = DataLoader(dataset, batch_size=clipwrap.batch_size, 
                      shuffle=False, num_workers=MAX_WORKERS, 
                      pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
                      collate_fn=custom_collate_fn)
    
    for template in tqdm(templates, desc="Evaluating prompts"):
        # Create text embeddings for all classes (vectorized)
        text_list = [template.format(class_name) for class_name in classes]
        txt_emb = clipwrap.batch_text_embeddings(text_list)
        
        predictions = []
        true_labels = []
        
        # Process images in batches
        for batch in tqdm(dataloader, desc=f"Processing batches - {template[:30]}...", leave=False):
            images = batch['image']
            labels = batch['class_idx'].numpy()
            
            # Get batch image embeddings
            img_emb_batch = clipwrap.batch_image_embeddings(images)
            
            # Compute similarities for entire batch
            similarities_batch = np.dot(img_emb_batch, txt_emb.T)
            top_indices_batch = np.argsort(-similarities_batch, axis=1)
            
            predictions.extend(top_indices_batch)
            true_labels.extend(labels)
        
        # Calculate accuracies
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        
        accuracies = {}
        for k in topk:
            # Vectorized accuracy calculation
            correct_mask = np.any(predictions[:, :k] == true_labels.reshape(-1, 1), axis=1)
            accuracies[f"top{k}"] = np.mean(correct_mask)
        
        results.append({
            "prompt": template,
            **accuracies,
            "total_samples": len(true_labels)
        })
    
    return pd.DataFrame(results)

def robustness_test(clipwrap, image_list, classes, templates, corruption_name, severities=SEVERITY_LEVELS):
    """
    Test CLIP robustness against image corruptions at various severity levels.
    
    Applies different corruption types (blur, noise, occlusion, etc.) at multiple
    severity levels and measures the performance drop compared to clean images.
    Includes early stopping if baseline accuracy is too low.
    
    Args:
        clipwrap (CLIPWrapper): Initialized CLIP wrapper instance
        image_list (list): List of image metadata dicts with 'path', 'class', 'class_idx'
        classes (list): List of class names
        templates (list): List of prompt templates to test
        corruption_name (str): Name of corruption type (key in CORRUPTIONS dict)
        severities (list): Severity levels to test (default: [1, 3, 5])
    
    Returns:
        pd.DataFrame: Results with columns for corruption type, severity, prompt,
                     clean/corrupted accuracy, and robustness drops
    
    Example:
        df = robustness_test(clipwrap, val_images, classes, templates, 'gaussian_blur')
    """
    results = []
    
    # Get clean baseline first (cached)
    print(f"Computing clean baseline for {corruption_name}...")
    clean_dataset = ImageDataset(image_list)
    clean_results = {}
    
    for template in templates:
        text_list = [template.format(class_name) for class_name in classes]
        txt_emb = clipwrap.batch_text_embeddings(text_list)
        
        dataloader = DataLoader(clean_dataset, batch_size=clipwrap.batch_size, 
                      shuffle=False, num_workers=MAX_WORKERS, pin_memory=True,
                      collate_fn=custom_collate_fn)
        
        correct_top1, correct_top5 = 0, 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc=f"Clean baseline - {template[:30]}...", leave=False):
            images = batch['image']
            labels = batch['class_idx'].numpy()
            
            img_emb_batch = clipwrap.batch_image_embeddings(images)
            similarities_batch = np.dot(img_emb_batch, txt_emb.T)
            top_indices_batch = np.argsort(-similarities_batch, axis=1)
            
            # Vectorized accuracy calculation
            correct_top1 += np.sum(top_indices_batch[:, 0] == labels)
            correct_top5 += np.sum(np.any(top_indices_batch[:, :5] == labels.reshape(-1, 1), axis=1))
            total_samples += len(labels)
        
        clean_results[template] = {
            "top1": correct_top1 / total_samples,
            "top5": correct_top5 / total_samples
        }
    
    # ADD THIS EARLY STOPPING CHECK HERE:
    if clean_results[templates[0]]["top1"] < 0.3:  # If baseline is too low
        print(f"Skipping {corruption_name} - baseline too low ({clean_results[templates[0]]['top1']:.3f})")
        return pd.DataFrame()
    
    # Test corruptions in parallel
    def process_corruption_severity(args):
        severity, template = args
        corruption_fn = CORRUPTIONS[corruption_name](severity)
        
        # Create corrupted dataset
        corrupted_dataset = ImageDataset(image_list, corruption_fn=corruption_fn)
        dataloader = DataLoader(corrupted_dataset, batch_size=clipwrap.batch_size, 
                      shuffle=False, num_workers=4,
                      collate_fn=custom_collate_fn)  # Reduced workers for memory
        
        text_list = [template.format(class_name) for class_name in classes]
        txt_emb = clipwrap.batch_text_embeddings(text_list)
        
        correct_top1, correct_top5 = 0, 0
        total_samples = 0
        
        for batch in dataloader:
            images = batch['image']
            labels = batch['class_idx'].numpy()
            
            img_emb_batch = clipwrap.batch_image_embeddings(images)
            similarities_batch = np.dot(img_emb_batch, txt_emb.T)
            top_indices_batch = np.argsort(-similarities_batch, axis=1)
            
            correct_top1 += np.sum(top_indices_batch[:, 0] == labels)
            correct_top5 += np.sum(np.any(top_indices_batch[:, :5] == labels.reshape(-1, 1), axis=1))
            total_samples += len(labels)
        
        corrupted_top1 = correct_top1 / total_samples
        corrupted_top5 = correct_top5 / total_samples
        
        return {
            "corruption": corruption_name,
            "severity": severity,
            "prompt": template,
            "clean_top1": clean_results[template]["top1"],
            "corrupted_top1": corrupted_top1,
            "robustness_drop_top1": clean_results[template]["top1"] - corrupted_top1,
            "clean_top5": clean_results[template]["top5"],
            "corrupted_top5": corrupted_top5,
            "robustness_drop_top5": clean_results[template]["top5"] - corrupted_top5,
        }
    
    # Create argument combinations
    severity_template_combinations = [(sev, temp) for sev in severities for temp in templates]
    
    # Process corruptions with limited parallelism to manage GPU memory
    batch_results = []
    batch_size_parallel = 2  # Process 2 combinations at a time to manage memory
    
    for i in range(0, len(severity_template_combinations), batch_size_parallel):
        batch_args = severity_template_combinations[i:i + batch_size_parallel]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            batch_futures = [executor.submit(process_corruption_severity, args) for args in batch_args]
            
            for future in tqdm(concurrent.futures.as_completed(batch_futures), 
                             desc=f"Processing {corruption_name}", 
                             total=len(batch_futures), leave=False):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing corruption: {e}")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return pd.DataFrame(batch_results)