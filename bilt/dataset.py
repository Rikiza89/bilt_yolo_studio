import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Dict, Any

from .utils import parse_yolo_label, load_yaml_classes, get_logger

logger = get_logger(__name__)


class ObjectDetectionDataset(Dataset):
    """CPU-optimized dataset with multi-resolution support."""
    
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms=None,
        input_size: int = 640
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.input_size = input_size
        
        # Find all images
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            self.image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        logger.info(f"Loaded {len(self.image_files)} images from {self.images_dir}")
        
        # Collect all class IDs
        self.class_ids = set()
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                self.class_ids.add(int(parts[0]))
                except Exception as e:
                    logger.warning(f"Error reading {label_path}: {e}")
        
        self.class_ids = sorted(list(self.class_ids))
        self.num_classes = len(self.class_ids)
        
        # Create mapping from original class IDs to consecutive IDs
        self.class_id_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        self.idx_to_class_id = {idx: class_id for class_id, idx in self.class_id_to_idx.items()}
        
        logger.info(f"Found {self.num_classes} classes: {self.class_ids}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
            orig_width, orig_height = img.size
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            img = Image.new("RGB", (640, 640))
            orig_width, orig_height = 640, 640
        
        # Parse annotations
        annotations = parse_yolo_label(label_path, orig_width, orig_height)
        
        # Convert to tensors and remap class IDs
        if len(annotations) > 0:
            boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
            labels = torch.tensor(
                [self.class_id_to_idx[ann['class_id']] for ann in annotations],
                dtype=torch.int64
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        # Get final image dimensions
        if isinstance(img, torch.Tensor):
            _, img_height, img_width = img.shape
        else:
            img_width, img_height = img.size
        
        # Scale boxes to transformed image size
        if len(boxes) > 0:
            scale_x = img_width / orig_width
            scale_y = img_height / orig_height
            
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            
            # Clamp to bounds
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, img_width)
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, img_height)
            boxes[:, 2] = torch.clamp(boxes[:, 2], 0, img_width)
            boxes[:, 3] = torch.clamp(boxes[:, 3], 0, img_height)
            
            # Remove invalid boxes
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return img, target
    
    def get_class_names(self, yaml_path: Path = None) -> List[str]:
        """Get class names from YAML or generate defaults."""
        # Try to load from YAML first
        if yaml_path and yaml_path.exists():
            yaml_classes = load_yaml_classes(yaml_path)
            if yaml_classes:
                logger.info(f"Loaded class names from YAML: {yaml_classes}")
                return [
                    yaml_classes[class_id] if class_id < len(yaml_classes) else f"class_{class_id}"
                    for class_id in self.class_ids
                ]
        
        # Generate default names
        logger.warning("Using default class names - no YAML file found")
        return [f"class_{class_id}" for class_id in self.class_ids]


def get_transforms(input_size: int = 640, training: bool = True):
    """Get image transforms with resizing for consistent batch processing."""
    if training:
        transforms_list = [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ]
    else:
        transforms_list = [T.ToTensor()]
    
    return T.Compose(transforms_list)


def collate_fn(batch):
    """Custom collate function."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloader(
    images_dir: Path,
    labels_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    input_size: int = 640,
    training: bool = True
) -> Tuple[DataLoader, int]:
    """Create CPU-safe dataloader with multi-resolution support."""
    
    dataset = ObjectDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=get_transforms(input_size, training),
        input_size=input_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    return dataloader, dataset.num_classes