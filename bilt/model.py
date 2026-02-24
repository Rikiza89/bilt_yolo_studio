import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from PIL import Image
import numpy as np

from .core import DetectionModel
from .trainer import Trainer
from .inferencer import Inferencer
from .evaluator import Evaluator
from .utils import get_logger

logger = get_logger(__name__)


class BILT:
    """
    BILT (Because I Like Twice) - Main interface for object detection.
    
    Examples:
        # Load pretrained model
        >>> model = BILT("weights.pth")
        
        # Predict on image
        >>> results = model.predict("image.jpg", conf=0.25)
        
        # Train new model
        >>> model = BILT()
        >>> model.train(dataset="datasets/my_dataset", epochs=50)
        
        # Evaluate model
        >>> metrics = model.evaluate("datasets/val")
    """
    
    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize BILT model.
        
        Args:
            weights: Path to .pth model weights. If None, creates untrained model.
            device: Device to use ('cpu', 'cuda', or None for auto-detect).
        """
        self.device = self._get_device(device)
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.inferencer = None
        
        if weights:
            self.load(weights)
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Determine device to use."""
        if device:
            return torch.device(device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load(self, weights: Union[str, Path]) -> 'BILT':
        """
        Load model weights.
        
        Args:
            weights: Path to .pth model file.
            
        Returns:
            self for chaining.
        """
        weights = Path(weights)
        logger.info(f"Loading model from {weights}")
        
        self.model, self.class_names = DetectionModel.load(weights)
        self.num_classes = len(self.class_names)
        self.model.to(self.device)
        self.model.eval()
        
        # Create inferencer
        self.inferencer = Inferencer(
            model=self.model,
            class_names=self.class_names,
            device=self.device
        )
        
        logger.info(f"Model loaded: {self.num_classes} classes")
        return self
    
    def predict(
        self,
        source: Union[str, Path, Image.Image, np.ndarray, List],
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        return_images: bool = False
    ) -> Union[List[Dict], 'Results']:
        """
        Run inference on images.
        
        Args:
            source: Input source - can be:
                - Path to image file
                - Path to directory of images
                - PIL Image
                - numpy array
                - List of any above
            conf: Confidence threshold (0.0-1.0).
            iou: NMS IoU threshold (0.0-1.0).
            img_size: Input image size for model.
            return_images: If True, return Results object with images.
            
        Returns:
            List of detection dictionaries or Results object.
            
        Example:
            >>> results = model.predict("image.jpg", conf=0.3)
            >>> for det in results:
            >>>     print(f"{det['class_name']}: {det['score']:.2f}")
        """
        if self.inferencer is None:
            raise RuntimeError("No model loaded. Call load() or train() first.")
        
        # Update inferencer thresholds
        self.inferencer.confidence_threshold = conf
        self.inferencer.nms_threshold = iou
        self.inferencer.input_size = img_size
        
        # Handle different input types
        images = self._prepare_source(source)
        
        # Run inference
        all_detections = []
        original_images = []
        
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img)
                original_images.append(pil_img if return_images else None)
                detections = self.inferencer.detect(pil_img)
            elif isinstance(img, Image.Image):
                original_images.append(img if return_images else None)
                detections = self.inferencer.detect(img)
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
                original_images.append(pil_img if return_images else None)
                detections = self.inferencer.detect(pil_img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            all_detections.append(detections)
        
        # Return raw detections if single image and not requesting Results
        if len(all_detections) == 1 and not return_images:
            return all_detections[0]
        
        if return_images:
            return Results(all_detections, original_images, self.class_names)
        
        return all_detections
    
    def _prepare_source(self, source) -> List:
        """Convert source to list of images."""
        if isinstance(source, list):
            return source
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                # Load all images in directory
                images = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.extend(list(path.glob(f"*{ext}")))
                    images.extend(list(path.glob(f"*{ext.upper()}")))
                return sorted(images)
            else:
                return [path]
        
        return [source]
    
    def train(
        self,
        dataset: Union[str, Path],
        epochs: int = 50,
        batch_size: int = 4,
        img_size: int = 640,
        learning_rate: float = 5e-4,
        device: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = "runs/train",
        name: str = "exp",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train object detection model.
        
        Args:
            dataset: Path to dataset directory (YOLO format).
            epochs: Number of training epochs.
            batch_size: Batch size (minimum 2 for batch normalization).
            img_size: Input image size.
            learning_rate: Learning rate.
            device: Device to use (overrides init device).
            save_dir: Directory to save training runs.
            name: Experiment name.
            **kwargs: Additional training arguments.
            
        Returns:
            Dictionary with training metrics.
            
        Example:
            >>> model = BILT()
            >>> results = model.train(
            ...     dataset="datasets/my_data",
            ...     epochs=100,
            ...     batch_size=8
            ... )
        """
        dataset = Path(dataset)
        save_dir = Path(save_dir)
        
        if device:
            self.device = torch.device(device)
        
        # Enforce minimum batch size
        if batch_size < 2:
            logger.warning("Batch size must be >= 2 for batch normalization. Setting to 2.")
            batch_size = 2
        
        # Create trainer
        from .dataset import ObjectDetectionDataset, get_transforms
        
        # Load dataset to get class info
        train_dataset = ObjectDetectionDataset(
            images_dir=dataset / "train" / "images",
            labels_dir=dataset / "train" / "labels",
            transforms=get_transforms(img_size, training=True),
            input_size=img_size
        )
        
        # Get class names
        yaml_path = dataset / "data.yaml"
        if not yaml_path.exists():
            for alt in [dataset / "data.yml", dataset / "dataset.yaml"]:
                if alt.exists():
                    yaml_path = alt
                    break
        
        class_names = train_dataset.get_class_names(yaml_path if yaml_path.exists() else None)
        num_classes = train_dataset.num_classes
        
        logger.info(f"Training on {num_classes} classes: {class_names}")
        
        # Create save directory
        run_dir = save_dir / name
        counter = 1
        while run_dir.exists():
            run_dir = save_dir / f"{name}{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trainer
        trainer = Trainer(
            dataset_path=dataset,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            num_workers=kwargs.get('workers', 0),
            input_size=img_size,
            device=self.device
        )
        
        # Train
        model_path = run_dir / "weights" / "best.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = trainer.train(model_path)
        
        # Load best model
        self.load(model_path)
        
        logger.info(f"Training complete. Model saved to {model_path}")
        
        return results
    
    def evaluate(
        self,
        dataset: Union[str, Path],
        batch_size: int = 4,
        conf: float = 0.25,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation dataset.
        
        Args:
            dataset: Path to validation dataset or subdirectory.
            batch_size: Batch size for evaluation.
            conf: Confidence threshold.
            **kwargs: Additional evaluation arguments.
            
        Returns:
            Dictionary with evaluation metrics.
            
        Example:
            >>> metrics = model.evaluate("datasets/val")
            >>> print(f"mAP: {metrics['map']:.3f}")
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() or train() first.")
        
        dataset = Path(dataset)
        
        # Handle both full dataset path and val subdirectory
        if (dataset / "val" / "images").exists():
            images_dir = dataset / "val" / "images"
            labels_dir = dataset / "val" / "labels"
        elif (dataset / "images").exists():
            images_dir = dataset / "images"
            labels_dir = dataset / "labels"
        else:
            raise ValueError(f"Could not find images in {dataset}")
        
        evaluator = Evaluator(
            model=self.model,
            class_names=self.class_names,
            device=self.device
        )
        
        metrics = evaluator.evaluate_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            batch_size=batch_size,
            confidence_threshold=conf
        )
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model weights.
        
        Args:
            path: Path to save .pth file.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        from .core import DetectionModel as DM
        
        # Create temporary wrapper
        wrapper = DM(num_classes=self.num_classes, pretrained=False)
        wrapper.model = self.model
        
        wrapper.save(path, self.class_names)
        logger.info(f"Model saved to {path}")
    
    @property
    def names(self) -> List[str]:
        """Get class names."""
        return self.class_names if self.class_names else []
    
    def __repr__(self) -> str:
        if self.model:
            return f"BILT(classes={self.num_classes}, device={self.device})"
        return f"BILT(untrained, device={self.device})"


class Results:
    """Results object for predictions with images."""
    
    def __init__(self, detections: List[List[Dict]], images: List[Image.Image], class_names: List[str]):
        self.detections = detections
        self.images = images
        self.class_names = class_names
    
    def __len__(self):
        return len(self.detections)
    
    def __getitem__(self, idx):
        return self.detections[idx]
    
    def save(self, save_dir: Union[str, Path] = "runs/detect") -> None:
        """Save annotated images."""
        from .utils import draw_detections
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (dets, img) in enumerate(zip(self.detections, self.images)):
            if img is None:
                continue
            
            annotated = draw_detections(img, dets)
            output_path = save_dir / f"result_{i}.jpg"
            annotated.save(output_path)
            logger.info(f"Saved {output_path}")
    
    def show(self):
        """Display results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from .utils import draw_detections
            
            for dets, img in zip(self.detections, self.images):
                if img is None:
                    continue
                
                annotated = draw_detections(img, dets)
                plt.figure(figsize=(12, 8))
                plt.imshow(annotated)
                plt.axis('off')
                plt.show()
        except ImportError:
            logger.error("matplotlib required for show(). Use save() instead.")