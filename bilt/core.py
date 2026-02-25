import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from pathlib import Path
import json
from typing import List, Tuple
from .utils import get_logger

logger = get_logger(__name__)


class DetectionModel:
    """Wrapper for CPU-optimized object detection model."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        self.num_classes = num_classes
        
        logger.info(f"Creating SSD MobileNetV3 model for {num_classes} classes")
        
        weights = (
            SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            if pretrained and num_classes == 91
            else None
        )
        
        self.model = ssdlite320_mobilenet_v3_large(
            weights=weights,
            num_classes=num_classes
        )
        
        self.model.train()
        logger.info("Model initialized successfully")
    
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model
    
    def set_training_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def save(self, save_path: Path, class_names: List[str], class_id_mapping: dict = None):
        """Save model for inference."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': class_names,
            'class_id_mapping': class_id_mapping,
            'architecture': 'ssdlite320_mobilenet_v3_large'
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Save class mapping separately
        class_map_path = save_path.parent / f"{save_path.stem}_classes.json"
        with open(class_map_path, 'w', encoding='utf-8') as f:
            json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(model_path: Path) -> Tuple[nn.Module, List[str]]:
        """Load trained model for inference."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Security: Use weights_only=True to prevent arbitrary code execution
        # via pickle deserialization of untrusted model files.
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        num_classes = checkpoint['num_classes']
        class_names = checkpoint['class_names']
        
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            num_classes=num_classes
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded successfully with {num_classes} classes")
        
        return model, class_names


def get_optimizer(model: nn.Module, learning_rate: float = 5e-4):
    """Get optimizer optimized for CPU object detection."""
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=1e-4
    )
    return optimizer


def get_lr_scheduler(optimizer, num_epochs: int):
    """Cosine decay scheduler for detection."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )