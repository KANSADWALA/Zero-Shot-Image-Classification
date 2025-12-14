import numpy as np
import torch    
from transformers import CLIPModel, CLIPProcessor
from trials.config import MODEL_NAME, DEVICE, BATCH_SIZE

class CLIPWrapper:
    """
    Wrapper class for CLIP model with optimized batch processing.
    
    Encapsulates CLIP model and processor, providing methods for efficient batch
    processing of images and texts. Supports mixed precision inference on GPU
    for improved performance.
    
    Attributes:
        device (str): 'cuda' or 'cpu'
        batch_size (int): Batch size for processing
        model: CLIP model instance
        processor: CLIP processor for image/text preprocessing
    """
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE):
        """
        Initialize CLIP model and processor.
        
        Args:
            model_name (str): HuggingFace model identifier (default: openai/clip-vit-base-patch32)
            device (str): Device to load model on ('cuda' or 'cpu')
            batch_size (int): Batch size for inference operations
        """
        self.device = device
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Enable mixed precision for faster inference
        if device == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference

    def batch_image_embeddings(self, images):
        """
        Compute normalized image embeddings in batches.
        
        Processes images through CLIP's image encoder with automatic batching
        for GPU memory efficiency. Returns L2-normalized embeddings.
        
        Args:
            images (list or PIL.Image): Image(s) to encode. Can be single image or list
        
        Returns:
            np.ndarray: Normalized embeddings of shape (N, embedding_dim)
        """
        if not isinstance(images, list):
            images = [images]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
            
            # Convert to half precision if using CUDA
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(emb.cpu().float().numpy())  # Convert back to float32
        
        return np.vstack(embeddings)

    def batch_text_embeddings(self, texts):
        """
        Compute normalized text embeddings in batches.
        
        Processes text prompts through CLIP's text encoder with automatic batching.
        Returns L2-normalized embeddings suitable for cosine similarity comparisons.
        
        Args:
            texts (list or str): Text string(s) to encode
        
        Returns:
            np.ndarray: Normalized embeddings of shape (N, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Convert to half precision if using CUDA
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(emb.cpu().float().numpy())
        
        return np.vstack(embeddings)