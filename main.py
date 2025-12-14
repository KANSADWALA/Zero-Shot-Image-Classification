# Importing dependencies
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import json
import torch

from trials.config import (MAX_WORKERS, DEVICE, BATCH_SIZE, DATASET_NAME, 
                           TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR, PROMPT_TEMPLATES, CORRUPTIONS)
from trials.data_setup import setup_cifar100
from trials.data_loader import load_images_parallel, ImageDataset
from trials.clip_wrapper import CLIPWrapper
from trials.zeroshot_eval import zero_shot_eval, robustness_evaluation as robustness_test
from trials.analysis import adaptive_testing_pipeline, progressive_corruption_testing, plot_comprehensive_results, analyze_prompt_robustness, generate_enhanced_summary_report

# -------------------
# ENHANCED MAIN WITH ADAPTIVE FEATURES
# -------------------
def main(adaptive_mode=True, force_full_evaluation=False):
    """
    Main execution function for comprehensive zero-shot robustness analysis.
    
    Orchestrates the complete pipeline:
    1. Dataset setup and loading (CIFAR-100)
    2. CLIP model initialization
    3. Zero-shot evaluation on clean images
    4. Robustness testing with multiple corruptions
    5. Analysis and report generation
    
    Args:
        adaptive_mode (bool): Enable adaptive testing that adjusts scope based on
                            system performance (default: True)
        force_full_evaluation (bool): Use all prompts/corruptions if True,
                                    regardless of system performance (default: False)
    
    Returns:
        tuple: (enhanced_summary_dict, df_clean, df_robustness) or (None, None, None) on error
    
    Output files created in RESULTS_DIR:
        - clean_performance.csv: Zero-shot results per prompt
        - robustness_results.csv: Corruption robustness test results
        - {mode}_summary.json: Execution summary
        - {mode}_comprehensive_resilience_analysis.json: Detailed resilience analysis
        - *.png: Visualizations (prompt performance, severity curves, heatmaps)
    
    Example:
        summary, df_clean, df_robust = main(adaptive_mode=False, force_full_evaluation=True)
    """
    import time
    
    print("Starting Optimized Zero-Shot Vision-Language Research...")
    print(f"Device: {DEVICE}")
    print(f"Adaptive mode: {'Enabled' if adaptive_mode else 'Disabled'}")
    
    start_total_time = time.time()
    
    # System setup with error handling
    try:
        # Set multiprocessing start method for CUDA compatibility
        if torch.cuda.is_available():
            torch.multiprocessing.set_start_method('spawn', force=True)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Batch size: {BATCH_SIZE}")
            print(f"Max workers: {MAX_WORKERS}")
        
        # 1. Dataset setup
        print("Setting up dataset...")
        classes = setup_cifar100()
        print(f"Dataset prepared with {len(classes)} classes")
        
        # 2. Load data with parallel processing
        print("Loading datasets...")
        train_images, _ = load_images_parallel(TRAIN_DIR)
        val_images, _ = load_images_parallel(VAL_DIR)
        test_images, _ = load_images_parallel(TEST_DIR)
        
        print(f"Loaded: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")
        
        # 3. Initialize optimized CLIP
        print("Initializing optimized CLIP...")
        clipwrap = CLIPWrapper()
        print("CLIP model loaded and optimized")
        
        # 4. Determine testing strategy (adaptive or standard)
        if adaptive_mode and not force_full_evaluation:
            print("Running adaptive testing pipeline...")
            templates_to_test, corruptions_to_test, test_images, benchmark_results = \
                adaptive_testing_pipeline(clipwrap, val_images, classes)
            
            print(f"Adaptive configuration:")
            print(f"  Templates to test: {len(templates_to_test)}")
            print(f"  Corruptions to test: {len(corruptions_to_test)}")
            print(f"  Images for testing: {len(test_images)}")
        else:
            print("Using standard/full evaluation mode...")
            templates_to_test = PROMPT_TEMPLATES
            corruptions_to_test = list(CORRUPTIONS.keys())
            
            if force_full_evaluation:
                test_images = val_images[:500]  # Use only first 3000 images instead of all 33,626
                print(f"Full evaluation with {len(test_images)} images")
            else:
                # Use subset for faster testing
                test_images = val_images[:2000]
                corruptions_to_test = corruptions_to_test[:4]  # Test first 4 corruptions
                print(f"Standard evaluation with {len(test_images)} images and {len(corruptions_to_test)} corruptions")
            
            benchmark_results = None
        
        # 5. Zero-shot evaluation
        print("Evaluating zero-shot performance...")
        val_dataset = ImageDataset(test_images)
        df_clean = zero_shot_eval(clipwrap, val_dataset, classes, templates_to_test)
        df_clean.to_csv(os.path.join(RESULTS_DIR, "clean_performance.csv"), index=False)
        print("Clean evaluation completed")
        
        # 6. Robustness testing
        print("Starting robustness testing...")
        
        # Get top performing prompts for efficiency
        num_top_prompts = min(3, len(templates_to_test))
        top_prompts = df_clean.nlargest(num_top_prompts, 'top1')['prompt'].tolist()

        #top_prompts = templates_to_test  # Use all available prompts

        robust_dfs = []
        
        for corruption_name in corruptions_to_test:
            print(f"Testing {corruption_name}...")
            try:
                if adaptive_mode:
                    # Use progressive testing for adaptive mode
                    df_corr = progressive_corruption_testing(clipwrap, test_images, classes, 
                                                           top_prompts, corruption_name)
                else:
                    # Use standard robustness testing
                    df_corr = robustness_test(clipwrap, test_images, classes, 
                                                      top_prompts, corruption_name)
                
                if not df_corr.empty:
                    robust_dfs.append(df_corr)
                
                # Clear GPU cache after each corruption
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error testing {corruption_name}: {e}")
                continue
        
        # Combine robustness results
        if robust_dfs:
            df_robustness = pd.concat(robust_dfs, ignore_index=True)
            df_robustness.to_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"), index=False)
            print("Robustness testing completed")
        else:
            print("No robustness results generated")
            df_robustness = pd.DataFrame()
        
        # 7. Generate comprehensive analysis
        print("Generating comprehensive analysis...")
        
        # Plot results
        if not df_robustness.empty:
            plot_comprehensive_results(df_clean, df_robustness, RESULTS_DIR)
            
            # Prompt resilience analysis
            prompt_resilience = analyze_prompt_robustness(df_robustness, RESULTS_DIR)
        else:
            prompt_resilience = pd.Series()
        
        # 8. Generate final comprehensive report
        print("Generating comprehensive report...")
        
        total_time = time.time() - start_total_time
        
        summary_report = {
            "execution_info": {
                "total_time_minutes": total_time / 60,
                "adaptive_mode": adaptive_mode,
                "force_full_evaluation": force_full_evaluation,
                "templates_tested": len(templates_to_test),
                "corruptions_tested": len(corruptions_to_test),
                "images_per_test": len(test_images)
            },
            "optimization_settings": {
                "batch_size": BATCH_SIZE,
                "max_workers": MAX_WORKERS,
                "device": DEVICE,
                "mixed_precision": DEVICE == "cuda"
            },
            "dataset_info": {
                "name": DATASET_NAME,
                "num_classes": len(classes),
                "train_samples": len(train_images),
                "val_samples": len(val_images),
                "test_samples": len(test_images)
            },
            "best_clean_performance": {
                "prompt": df_clean.loc[df_clean['top1'].idxmax(), 'prompt'],
                "top1_accuracy": df_clean['top1'].max(),
                "top5_accuracy": df_clean.loc[df_clean['top1'].idxmax(), 'top5']
            }
        }

        # Add benchmark results if available
        if benchmark_results is not None and len(benchmark_results) >= 3:
            _, processing_time, samples_per_second = benchmark_results
            summary_report["benchmark_info"] = {
                "samples_per_second": samples_per_second,
                "benchmark_time": processing_time
            }
        elif benchmark_results is not None:
            summary_report["benchmark_info"] = {
                "samples_per_second": "N/A",
                "benchmark_time": "N/A"
            }

        # Add robustness analysis if available
        if not df_robustness.empty:
            summary_report.update({
                "robustness_analysis": {
                    "most_robust_prompt": prompt_resilience.index[0] if len(prompt_resilience) > 0 else "N/A",
                    "avg_robustness_drop": df_robustness['robustness_drop_top1'].mean(),
                    "tested_corruptions": corruptions_to_test
                },
                "corruption_ranking": df_robustness.groupby('corruption')['robustness_drop_top1'].mean().sort_values().to_dict()
            })
        
        # Generate enhanced summary with resilience analysis
        enhanced_summary = generate_enhanced_summary_report(df_clean, df_robustness, summary_report, prompt_resilience)
        
        # 9. Save results
        mode_suffix = "adaptive" if adaptive_mode else "standard"
        
        with open(os.path.join(RESULTS_DIR, f"{mode_suffix}_summary.json"), 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(os.path.join(RESULTS_DIR, f"{mode_suffix}_comprehensive_resilience_analysis.json"), 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        # 10. Print final report
        print("\n" + "="*70)
        print("OPTIMIZED ZERO-SHOT RESEARCH COMPLETED!")
        print("="*70)
        print(f"Execution time: {total_time/60:.1f} minutes")
        print(f"Mode: {'Adaptive' if adaptive_mode else 'Standard'}")
        print(f"Best Clean Performance: {summary_report['best_clean_performance']['top1_accuracy']:.3f}")
        print(f"Best Prompt: {summary_report['best_clean_performance']['prompt'][:50]}...")
        
        if not df_robustness.empty and "prompt_resilience_analysis" in enhanced_summary:
            resilience_info = enhanced_summary["prompt_resilience_analysis"]["overview"]
            print(f"Most Resilient Prompt: {resilience_info['most_resilient_prompt'][:50]}...")
            print(f"Best Resilience Score: {resilience_info['best_resilience_score']:.3f}")
            print(f"Total Prompts Analyzed: {enhanced_summary['prompt_resilience_analysis']['statistical_summary']['total_prompts_tested']}")
        
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Detailed analysis: {mode_suffix}_comprehensive_resilience_analysis.json")
        print("="*70)
        
        return enhanced_summary, df_clean, df_robustness
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# -------------------
# EXECUTION
# -------------------
if __name__ == "__main__":
    import time
    
    # Configuration options - MODIFIED FOR COMPREHENSIVE TESTING
    ADAPTIVE_MODE = False         # Disable adaptive mode for full testing
    FORCE_FULL_EVAL = True        # Force comprehensive evaluation
    
    print("Starting execution...")
    
    if FORCE_FULL_EVAL:
        print("Running FULL evaluation mode (all prompts, all corruptions)...")
        results = main(adaptive_mode=False, force_full_evaluation=True)
    elif ADAPTIVE_MODE:
        print("Running ADAPTIVE mode (performance-based optimization)...")
        results = main(adaptive_mode=True, force_full_evaluation=False)
    else:
        print("Running STANDARD optimized mode...")
        results = main(adaptive_mode=False, force_full_evaluation=False)
    
    print("Execution completed!")