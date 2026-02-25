import logging
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def set_logging_level(level: str = "INFO"):
    """Set logging level for BILT library."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.getLogger("bilt").setLevel(level_map.get(level.upper(), logging.INFO))


def parse_yolo_label(
    label_path: Path,
    img_width: int,
    img_height: int
) -> List[Dict[str, Any]]:
    """
    Parse YOLO format label file to absolute pixel coordinates.
    
    YOLO format: class_id x_center y_center width height (normalized)
    Output: List of {class_id, bbox: [x_min, y_min, x_max, y_max]}
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate normalized values
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                       0 <= width <= 1 and 0 <= height <= 1):
                    continue
                
                # Convert to absolute pixel coordinates
                x_center_abs = x_center * img_width
                y_center_abs = y_center * img_height
                width_abs = width * img_width
                height_abs = height * img_height
                
                x_min = x_center_abs - width_abs / 2
                y_min = y_center_abs - height_abs / 2
                x_max = x_center_abs + width_abs / 2
                y_max = y_center_abs + height_abs / 2
                
                # Ensure within bounds
                x_min = max(0, min(x_min, img_width))
                y_min = max(0, min(y_min, img_height))
                x_max = max(0, min(x_max, img_width))
                y_max = max(0, min(y_max, img_height))
                
                # Validate box dimensions
                if x_max > x_min and y_max > y_min:
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x_min, y_min, x_max, y_max]
                    })
    
    except Exception as e:
        get_logger(__name__).warning(f"Error parsing label {label_path}: {e}")
    
    return annotations


def load_yaml_classes(yaml_path: Path) -> Optional[List[str]]:
    """Load class names from YOLO data.yaml file."""
    try:
        import yaml
    except ImportError:
        get_logger(__name__).warning("PyYAML not installed. Install with: pip install pyyaml")
        return None
    
    if not yaml_path.exists():
        return None
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data:
            names = data['names']
            if isinstance(names, list):
                return names
            elif isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
        
        return None
    except Exception as e:
        get_logger(__name__).warning(f"Failed to load YAML: {e}")
        return None


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.4
) -> torch.Tensor:
    """Apply Non-Maximum Suppression."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def draw_detections(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """Draw bounding boxes and labels on image."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    colors = [
        'red', 'blue', 'green', 'yellow', 'purple',
        'orange', 'pink', 'cyan', 'magenta', 'lime'
    ]
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        score = detection['score']
        class_id = detection['class_id']
        
        color = colors[class_id % len(colors)]
        
        # Draw box
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()
        
        # Get text size
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background
        draw.rectangle(
            [bbox[0], bbox[1] - text_height - 4, bbox[0] + text_width + 4, bbox[1]],
            fill=color
        )
        
        # Draw text
        draw.text((bbox[0] + 2, bbox[1] - text_height - 2), label, fill='white', font=font)
    
    return result


def validate_dataset_structure(dataset_path: Path) -> tuple:
    """
    Validate YOLO-style dataset structure.
    
    Expected:
        dataset/
            train/images/
            train/labels/
            val/images/
            val/labels/
    """
    required_paths = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "val" / "images",
        dataset_path / "val" / "labels"
    ]
    
    for path in required_paths:
        if not path.exists():
            return False, f"Missing required directory: {path.relative_to(dataset_path)}"
    
    # Check for at least some images
    train_images = list((dataset_path / "train" / "images").glob("*"))
    if len(train_images) == 0:
        return False, "No training images found"
    
    return True, "Dataset structure valid"