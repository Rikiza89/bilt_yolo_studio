import torch
from pathlib import Path
from typing import Dict, List, Any

from .dataset import create_dataloader
from .utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluate trained detection model."""
    
    def __init__(
        self,
        model,
        class_names: List[str],
        device: torch.device = None
    ):
        self.model = model
        self.class_names = class_names
        self.device = device if device else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
        batch_size: int = 4,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on a test dataset.
        
        Returns basic detection statistics.
        """
        logger.info("Starting evaluation...")
        
        test_loader, _ = create_dataloader(
            images_dir=images_dir,
            labels_dir=labels_dir,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            training=False
        )
        
        total_images = 0
        total_predictions = 0
        total_ground_truth = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                for pred, target in zip(predictions, targets):
                    total_images += 1
                    
                    # Filter by confidence
                    scores = pred['scores']
                    keep = scores > confidence_threshold
                    
                    total_predictions += keep.sum().item()
                    total_ground_truth += len(target['boxes'])
        
        avg_predictions_per_image = total_predictions / max(total_images, 1)
        avg_ground_truth_per_image = total_ground_truth / max(total_images, 1)
        
        results = {
            'total_images': total_images,
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'avg_predictions_per_image': avg_predictions_per_image,
            'avg_ground_truth_per_image': avg_ground_truth_per_image,
            'confidence_threshold': confidence_threshold
        }
        
        logger.info(f"Evaluation complete: {results}")
        
        return results