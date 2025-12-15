# Zero Shot Image Classification with Vision-Language Pre-trained models - Prompt Engineering & Robustness Analysis

## 📋 Overview

<strong>Problem Statement:</strong>
This research investigates the impact of prompt engineering and image robustness factors on the zero-shot image classification performance of the CLIP vision–language model. The study analyzes how different natural language prompt templates influence classification accuracy and examines the sensitivity of CLIP’s zero-shot predictions to common real-world image corruptions, including blur, noise, occlusion, and lighting variations. By systematically evaluating prompt design choices and corruption severity levels, this work aims to provide insights into optimizing prompt formulations and deploying CLIP-based systems in noisy and unconstrained visual environments.


## 📌 Prompt Templates Used

| **Category** | **Prompt Template(s)** |
|-------------|------------------------|
| **Baseline** | `a photo of a {}` |
| **Visual Context** | `a blurry image of a {}`<br>`a microscopic image of a {}` |
| **Descriptive Adjectives** | `a high resolution image of a {}`<br>`a low-light photo of a {}`<br>`a close-up photo of a {}`<br>`a cropped image of a {}`<br>`a bright photo of a {}`<br>`a black and white photo of a {}` |
| **Domain-Specific** | `a satellite image of {}`<br>`an X-ray image of {}`<br>`a food photo of {}` |
| **Additional Variations** | `a painting of a {}` |


## 🧪 Image Corruptions used

| **Corruption Name** | **What It Does (Simple Explanation)** |
|-------------------|----------------------------------------|
| **Gaussian Blur** | Makes the image soft and unfocused |
| **Motion Blur** | Simulates camera movement or fast object motion |
| **Gaussian Noise** | Adds random grain, making the image noisy |
| **Salt & Pepper Noise** | Introduces random white and black pixels |
| **Random Occlusion** | Blocks large regions of the image, hiding object parts |
| **Cutout** | Removes square patches from the image |
| **Brightness Change** | Makes the image darker or brighter |
| **Contrast Change** | Alters the difference between light and dark regions |
| **JPEG Compression** | Adds compression artifacts, making the image blocky or pixelated |


## 🌟 Features

<ol> 

<li>Zero-Shot Image Classification: 
  <ul>
    <li>Implements zero-shot classification using the CLIP (ViT-B/32) vision–language model without any task-specific fine-tuning.</li>
  </ul>
</li>


<li>Prompt Engineering Analysis: 
  <ul>
    <li>Evaluates 13 natural language prompt templates to study their impact on zero-shot classification accuracy.</li>
    <li>Supports descriptive, contextual, and domain-inspired prompt variations.</li>
  </ul>
</li>


<li>Robustness Evaluation under Image Corruptions: 
  <ul>
    <li>Tests model robustness against real-world 9 image corruptions and 3 Severity Levels[1, 3 ,5].</li>
    <li>Evaluates performance across multiple corruption severity levels.</li>
  </ul>
</li>

<li>Top-K Accuracy Metrics:
  <ul>
    <li>Computes Top-1 and Top-5 accuracy for comprehensive performance assessment.</li>
  </ul>
</li>

<li>Prompt Resilience Analysis:
  <ul>
    <li>Quantifies robustness drop per prompt and ranks prompts based on average performance degradation.</li>
  </ul>
</li>

<li>Large-Scale Evaluation Pipeline:
  <ul>
    <li>Optimized batch processing with parallel data loading for efficient evaluation on large datasets (CIFAR-100).</li>
  </ul>
</li>

<li>Visualization & Reporting:
  <ul>
  <li>Automatically generates:
    <ul>
      <li>Prompt-wise performance plots</li>
      <li>Corruption severity curves</li>
      <li>Prompt × corruption robustness heatmaps</li>
    </ul></li>

  <li>Exports results as CSV and JSON reports.</li>
  </ul>
</li>

<li>Adaptive & Optimized Execution:
  <ul>
    <li>Supports mixed-precision inference (FP16) on GPU.</li>
    <li>Includes memory-efficient batching and adaptive testing strategies.</li>
  </ul>
</li>


</ol>



















