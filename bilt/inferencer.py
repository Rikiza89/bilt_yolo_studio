import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any

from .utils import get_logger, apply_nms

logger = get_logger(__name__)


class Inferencer:
    """CPU-optimized inference engine with proper coordinate handling."""
    
    def __init__(
        self,
        model,
        class_names: List[str],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 640,
        device: torch.device = None
    ):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        self.device = device if device else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Simple transform
        self.transforms = T.Compose([T.ToTensor()])
        
        logger.info(
            f"Inferencer initialized - "
            f"Classes: {len(class_names)}, "
            f"Confidence: {confidence_threshold}, "
            f"NMS: {nms_threshold}"
        )
    
    def preprocess_image(self, image: Image.Image) -> tuple:
        """
        Preprocess image for inference.
        Returns: (tensor, original_size)
        """
        original_size = image.size  # (width, height)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        image_tensor = self.transforms(image)
        
        return image_tensor, original_size
    
    def postprocess_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        original_size: tuple
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Raw model output
            original_size: (width, height) of original image
        
        Returns:
            List of detections with boxes, scores, and class names
        """
        boxes = predictions['boxes'].cpu()
        scores = predictions['scores'].cpu()
        labels = predictions['labels'].cpu()
        
        # Filter by confidence
        keep = scores > self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        if len(boxes) == 0:
            return []
        
        # Apply NMS
        keep_indices = apply_nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Clamp boxes to image bounds
        orig_width, orig_height = original_size
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            
            # Clamp to image bounds
            x1 = max(0, min(int(x1), orig_width))
            y1 = max(0, min(int(y1), orig_height))
            x2 = max(0, min(int(x2), orig_width))
            y2 = max(0, min(int(y2), orig_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            class_id = label.item()
            
            # Handle class names
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        
        return detections
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run detection on a single image.
        
        Args:
            image: PIL Image
        
        Returns:
            List of detections
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # Postprocess
        detections = self.postprocess_predictions(predictions, original_size)
        
        logger.debug(f"Detected {len(detections)} objects in image of size {original_size}")
        
        return detections
    
    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Run detection on multiple images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)
        
        return all_detections
    
    def detect_from_path(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run detection on image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of detections
        """
        try:
            image = Image.open(image_path)
            return self.detect(image)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return []