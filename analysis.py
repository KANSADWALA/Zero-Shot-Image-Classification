import os   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from trials.config import SEVERITY_LEVELS, PROMPT_TEMPLATES, CORRUPTIONS
from trials.zeroshot_eval import zero_shot_eval
from trials.clip_wrapper import CLIPWrapper
from trials.data_loader import ImageDataset, custom_collate_fn


def generate_confusion_matrix(predictions, true_labels, classes, save_path):
    """
    Generate and save a confusion matrix visualization.
    
    Creates a heatmap visualization of classification results for the first 20
    classes (for readability). Saves the figure to disk.
    
    Args:
        predictions (np.ndarray): Predicted class indices of shape (N,)
        true_labels (np.ndarray): Ground truth class indices of shape (N,)
        classes (list): List of all class names
        save_path (str): File path to save the figure (e.g., 'confusion_matrix.png')
    
    Returns:
        None (saves figure to disk)
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=classes[:20], yticklabels=classes[:20])
    plt.title("Confusion Matrix (First 20 Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_prompt_robustness(df_robust, save_dir):
    """
    Analyze and rank prompts by their robustness to corruptions.
    
    Computes average performance drop across all corruption types for each prompt,
    generates a bar chart, and returns rankings (lower drop = more robust).
    
    Args:
        df_robust (pd.DataFrame): Robustness test results with 'prompt' and
                                 'robustness_drop_top1' columns
        save_dir (str): Directory to save visualization
    
    Returns:
        pd.Series: Prompts indexed by name, sorted by average robustness drop
    
    Example:
        resilience = analyze_prompt_robustness(df_robustness, 'results/')
        print(resilience.index[0])  # Most robust prompt
    """
    prompt_robustness = df_robust.groupby('prompt')['robustness_drop_top1'].mean().sort_values()
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(prompt_robustness)), prompt_robustness.values)
    plt.yticks(range(len(prompt_robustness)), prompt_robustness.index)
    plt.xlabel("Average Robustness Drop (Top-1)")
    plt.title("Prompt Resilience to Corruptions (Lower is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prompt_resilience.png"))
    plt.close()
    
    return prompt_robustness

def plot_comprehensive_results(df_clean, df_robust, save_dir):
    """
    Generate comprehensive visualization of all evaluation results.
    
    Creates four types of plots:
    1. Bar chart of clean zero-shot accuracy by prompt
    2. Line plots of performance vs severity for each corruption
    3. Heatmap of prompt × corruption robustness drops
    4. Additional corruption-specific analysis
    
    Args:
        df_clean (pd.DataFrame): Clean evaluation results with 'prompt', 'top1', 'top5'
        df_robust (pd.DataFrame): Robustness test results
        save_dir (str): Directory to save all generated plots
    
    Returns:
        None (saves figures to disk in save_dir)
    """
    
    # 1. Prompt performance plot
    plt.figure(figsize=(12, 6))
    df_clean_sorted = df_clean.sort_values("top1", ascending=True)
    plt.barh(df_clean_sorted['prompt'], df_clean_sorted['top1'])
    plt.xlabel("Top-1 Accuracy")
    plt.title("Zero-Shot Accuracy by Prompt Template")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prompt_performance.png"))
    plt.close()
    
    # 2. Corruption severity plots
    for corruption in df_robust['corruption'].unique():
        sub_df = df_robust[df_robust['corruption'] == corruption]
        severity_performance = sub_df.groupby('severity')['corrupted_top1'].mean()
        
        plt.figure()
        plt.plot(severity_performance.index, severity_performance.values, marker='o')
        plt.xlabel("Severity Level")
        plt.ylabel("Top-1 Accuracy")
        plt.title(f"Performance vs Severity - {corruption}")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"severity_{corruption}.png"))
        plt.close()
    
    # 3. Prompt-robustness heatmap
    pivot_data = df_robust.groupby(['prompt', 'corruption'])['robustness_drop_top1'].mean().unstack()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title("Prompt × Corruption Robustness Drop Heatmap")
    plt.xlabel("Corruption Type")
    plt.ylabel("Prompt Template")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prompt_corruption_heatmap.png"))
    plt.close()

def generate_enhanced_summary_report(df_clean, df_robustness, summary_report, prompt_resilience):
    """
    Generate comprehensive analysis report with detailed resilience metrics.
    
    Creates an enhanced summary report containing:
    - Prompt rankings by resilience score
    - Corruption-specific resilience analysis
    - Statistical insights across all tests
    - Actionable recommendations for prompt selection
    
    Args:
        df_clean (pd.DataFrame): Clean evaluation results
        df_robustness (pd.DataFrame): Robustness test results
        summary_report (dict): Base summary report to enhance
        prompt_resilience (pd.Series): Prompt resilience scores from analyze_prompt_robustness
    
    Returns:
        dict: Enhanced report with additional 'prompt_resilience_analysis' section
    
    Report includes:
        - overall_overview: Best/worst prompts
        - complete_prompt_ranking: All prompts ranked by resilience
        - corruption_specific_resilience: Analysis per corruption type
        - statistical_summary: Aggregated metrics
        - recommendations: Actionable suggestions
    """
    
    enhanced_report = summary_report.copy()
    
    if not df_robustness.empty and len(prompt_resilience) > 0:
        # Calculate detailed resilience metrics
        prompt_rankings = []
        for i, (prompt, avg_drop) in enumerate(prompt_resilience.items()):
            prompt_rankings.append({
                "rank": i + 1,
                "prompt": prompt,
                "average_robustness_drop": float(avg_drop),
                "resilience_score": float(1.0 - avg_drop),
                "resilience_category": "High" if avg_drop < 0.1 else "Medium" if avg_drop < 0.2 else "Low"
            })
        
        # Corruption-specific resilience
        corruption_resilience = {}
        for corruption in df_robustness['corruption'].unique():
            corruption_data = df_robustness[df_robustness['corruption'] == corruption]
            corruption_prompt_performance = corruption_data.groupby('prompt')['robustness_drop_top1'].mean().sort_values()
            
            corruption_resilience[corruption] = {
                "most_resilient_prompt": corruption_prompt_performance.index[0],
                "best_resilience_drop": float(corruption_prompt_performance.iloc[0]),
                "worst_resilience_drop": float(corruption_prompt_performance.iloc[-1]),
                "ranking": [
                    {"prompt": prompt, "robustness_drop": float(drop)}
                    for prompt, drop in corruption_prompt_performance.items()
                ]
            }
        
        # Top resilient prompts
        top_resilient_prompts = prompt_rankings[:min(5, len(prompt_rankings))]
        
        # Statistical insights
        statistical_insights = {
            "total_prompts_tested": len(prompt_resilience),
            "total_corruptions_tested": len(df_robustness['corruption'].unique()),
            "average_clean_accuracy": float(df_clean['top1'].mean()),
            "average_robustness_drop_all_prompts": float(df_robustness['robustness_drop_top1'].mean()),
            "std_robustness_drop": float(df_robustness['robustness_drop_top1'].std()),
            "best_worst_gap": float(prompt_resilience.iloc[-1] - prompt_resilience.iloc[0])
        }
        
        # Generate recommendations
        recommendations = []
        if prompt_rankings:
            best_prompt = prompt_rankings[0]
            recommendations.append({
                "type": "best_overall",
                "recommendation": f"Use '{best_prompt['prompt']}' for general robustness",
                "rationale": f"Shows lowest average robustness drop of {best_prompt['average_robustness_drop']:.3f}"
            })
            
            for corruption, data in corruption_resilience.items():
                recommendations.append({
                    "type": "corruption_specific",
                    "corruption": corruption,
                    "recommendation": f"For {corruption}, use '{data['most_resilient_prompt']}'",
                    "rationale": f"Best performance with robustness drop of {data['best_resilience_drop']:.3f}"
                })
        
        # Add to enhanced report
        enhanced_report["prompt_resilience_analysis"] = {
            "overview": {
                "most_resilient_prompt": top_resilient_prompts[0]["prompt"] if top_resilient_prompts else "N/A",
                "best_resilience_score": top_resilient_prompts[0]["resilience_score"] if top_resilient_prompts else 0,
                "least_resilient_prompt": prompt_rankings[-1]["prompt"] if prompt_rankings else "N/A",
                "worst_resilience_score": prompt_rankings[-1]["resilience_score"] if prompt_rankings else 0
            },
            "top_resilient_prompts": top_resilient_prompts,
            "complete_prompt_ranking": prompt_rankings,
            "corruption_specific_resilience": corruption_resilience,
            "statistical_summary": statistical_insights,
            "recommendations": recommendations
        }
    
    return enhanced_report

# -------------------
# ADDITIONAL OPTIMIZATION UTILITIES
# -------------------

# Fix 1: Correct the memory_efficient_batch_processing function
def memory_efficient_batch_processing(clipwrap, batch_size=None):
    """
    Automatically adjust batch size based on available GPU memory.
    
    Implements heuristic batch size reduction for systems with limited GPU memory.
    Helps prevent out-of-memory errors while maximizing throughput.
    
    Args:
        clipwrap (CLIPWrapper): CLIP wrapper instance
        batch_size (int, optional): Override batch size (if None, uses clipwrap's setting)
    
    Returns:
        int: Adjusted batch size safe for current GPU
    
    Notes:
        - GPU with <4GB: batch_size // 4
        - GPU with <8GB: batch_size // 2
        - GPU with >=8GB: batch_size unchanged
    """
    if batch_size is None:
        batch_size = clipwrap.batch_size
    
    # Auto-adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        # Heuristic: reduce batch size for lower memory GPUs
        if gpu_memory < 4e9:  # Less than 4GB
            batch_size = max(1, batch_size // 4)
        elif gpu_memory < 8e9:  # Less than 8GB
            batch_size = max(1, batch_size // 2)
    
    return batch_size

def progressive_corruption_testing(clipwrap, image_list, classes, templates, 
                                 corruption_name, max_samples_per_severity=500):
    """
    Test corruptions progressively with early stopping on severe degradation.
    
    Tests corruption severity levels sequentially. Limits samples for efficiency
    and stops early if accuracy falls below 10%. Tests only the best performing
    template to save computation time.
    
    Args:
        clipwrap (CLIPWrapper): Initialized CLIP wrapper
        image_list (list): Images to test
        classes (list): Class names
        templates (list): Prompt templates (best one selected automatically)
        corruption_name (str): Type of corruption to test
        max_samples_per_severity (int): Max images to test per severity level
    
    Returns:
        pd.DataFrame: Results with columns 'corruption', 'severity', 'accuracy', 'samples_tested'
    
    Early stopping occurs when accuracy < 10% (indicates severe distortion)
    """
    results = []
    
    # Start with lowest severity and adapt
    for severity in SEVERITY_LEVELS:
        print(f"Testing {corruption_name} severity {severity}...")
        
        # Limit samples for higher severities to save time
        sample_size = min(len(image_list), max_samples_per_severity)
        test_images = random.sample(image_list, sample_size)
        
        corruption_fn = CORRUPTIONS[corruption_name](severity)
        corrupted_dataset = ImageDataset(test_images, corruption_fn=corruption_fn)
        
        # Use smaller batch size for corrupted images to manage memory
        batch_size = memory_efficient_batch_processing(clipwrap)
        dataloader = DataLoader(corrupted_dataset, batch_size=batch_size, 
                      shuffle=False, num_workers=4,
                      collate_fn=custom_collate_fn)
        
        # Test with best performing template only for efficiency
        best_template = templates[0]  # Assume first is best from previous results
        text_list = [best_template.format(class_name) for class_name in classes]
        txt_emb = clipwrap.batch_text_embeddings(text_list)
        
        correct_top1 = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc=f"Severity {severity}", leave=False):
            try:
                images = batch['image']
                labels = batch['class_idx'].numpy()
                
                img_emb_batch = clipwrap.batch_image_embeddings(images)
                similarities_batch = np.dot(img_emb_batch, txt_emb.T)
                predictions = np.argmax(similarities_batch, axis=1)
                
                correct_top1 += np.sum(predictions == labels)
                total_samples += len(labels)
                
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue
        
        if total_samples > 0:
            accuracy = correct_top1 / total_samples
            results.append({
                "corruption": corruption_name,
                "severity": severity,
                "accuracy": accuracy,
                "samples_tested": total_samples
            })
            
            # Early stopping if performance drops too much
            if accuracy < 0.1:  # Less than 10% accuracy
                print(f"Early stopping for {corruption_name} at severity {severity}")
                break
        
        # Clear cache after each severity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

def quick_benchmark_run(clipwrap, val_images, classes, num_samples=1000):
    """
    Quick performance benchmark to measure system throughput.
    
    Tests model performance on a small subset of images and prompts to estimate
    processing speed (samples per second). Results inform adaptive testing decisions.
    
    Args:
        clipwrap (CLIPWrapper): Initialized CLIP wrapper
        val_images (list): Validation image list
        classes (list): Class names
        num_samples (int): Number of images to benchmark with (default: 1000)
    
    Returns:
        tuple: (benchmark_results_df, processing_time_seconds, samples_per_second)
    
    Example:
        df_bench, time_sec, speed = quick_benchmark_run(clipwrap, val_images, classes)
        print(f"Speed: {speed:.1f} samples/second")
    """
    print("🏃 Running quick benchmark...")
    
    # Use subset of images and prompts
    benchmark_images = random.sample(val_images, min(num_samples, len(val_images)))
    benchmark_templates = PROMPT_TEMPLATES[:3]  # Top 3 prompts only
    
    benchmark_dataset = ImageDataset(benchmark_images)
    
    start_time = time.time()
    df_benchmark = zero_shot_eval(clipwrap, benchmark_dataset, classes, benchmark_templates)
    end_time = time.time()
    
    processing_time = end_time - start_time
    samples_per_second = (len(benchmark_images) * len(benchmark_templates)) / processing_time
    
    print(f"⚡ Benchmark Results:")
    print(f"   • Processed {len(benchmark_images)} images × {len(benchmark_templates)} prompts")
    print(f"   • Total time: {processing_time:.2f}s")
    print(f"   • Speed: {samples_per_second:.1f} samples/second")
    
    return df_benchmark, processing_time, samples_per_second

def adaptive_testing_pipeline(clipwrap, val_images, classes):
    """
    Adaptive testing pipeline that adjusts scope based on system performance.
    
    Runs a quick benchmark to measure system throughput, then adapts the testing
    configuration:
    - High performance (>100 samples/sec): Full evaluation
    - Medium performance (>50 samples/sec): Balanced evaluation
    - Lower performance: Efficient evaluation with reduced scope
    
    Args:
        clipwrap (CLIPWrapper): Initialized CLIP wrapper
        val_images (list): Validation images to choose from
        classes (list): Class names
    
    Returns:
        tuple: (templates_to_test, corruptions_to_test, test_images, benchmark_results)
    
    Returned values:
        - templates_to_test (list): Selected prompts for testing
        - corruptions_to_test (list): Selected corruption types
        - test_images (list): Selected image subset
        - benchmark_results: Performance metrics from quick benchmark
    """
    print("🔄 Starting adaptive testing pipeline...")
    
    # 1. Quick benchmark to assess system performance
    benchmark_results, benchmark_time, speed = quick_benchmark_run(clipwrap, val_images, classes)
    
    # 2. Adapt testing strategy based on performance (MODIFIED THRESHOLDS)
    if speed > 100:  # High performance system
        print("🚀 High performance detected - full evaluation")
        templates_to_test = PROMPT_TEMPLATES
        corruptions_to_test = list(CORRUPTIONS.keys())
        samples_per_test = len(val_images)
    elif speed > 50:  # Medium performance
        print("⚡ Medium performance detected - balanced evaluation")
        templates_to_test = PROMPT_TEMPLATES[:10]  # Increased from 8
        corruptions_to_test = list(CORRUPTIONS.keys())[:7]  # Increased from 6
        samples_per_test = min(3500, len(val_images))  # Increased from 3000
    else:  # Lower performance - LESS RESTRICTIVE
        print("🐌 Lower performance detected - efficient evaluation")
        templates_to_test = PROMPT_TEMPLATES[:8]  # Increased from 5 to 8
        corruptions_to_test = list(CORRUPTIONS.keys())[:6]  # Increased from 4 to 6
        samples_per_test = min(2000, len(val_images))  # Increased from 1500 to 2000
    
    # 3. Limit images for testing
    test_images = random.sample(val_images, samples_per_test)
    
    return templates_to_test, corruptions_to_test, test_images, benchmark_results
