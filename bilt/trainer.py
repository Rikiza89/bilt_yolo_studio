import torch
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time

from .core import DetectionModel, get_optimizer, get_lr_scheduler
from .dataset import create_dataloader
from .utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """CPU-optimized training engine for object detection."""
    
    def __init__(
        self,
        dataset_path: Path,
        num_classes: int,
        class_names: list,
        batch_size: int = 4,
        learning_rate: float = 5e-4,
        num_epochs: int = 50,
        num_workers: int = 0,
        input_size: int = 640,
        device: torch.device = None
    ):
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.input_size = input_size
        
        self.device = device if device else torch.device('cpu')
        
        # Create dataloaders
        logger.info("Creating training dataloader...")
        self.train_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "train" / "images",
            labels_dir=self.dataset_path / "train" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            input_size=input_size,
            training=True
        )
        
        logger.info("Creating validation dataloader...")
        self.val_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "val" / "images",
            labels_dir=self.dataset_path / "val" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            input_size=input_size,
            training=True
        )
        
        # Create model
        logger.info("Initializing model...")
        self.detection_model = DetectionModel(num_classes=num_classes, pretrained=True)
        self.model = self.detection_model.get_model()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(self.model, learning_rate)
        self.scheduler = get_lr_scheduler(self.optimizer, num_epochs)
        
        # Training state
        self.current_epoch = 0
        self.training_losses = []
        self.validation_losses = []
        
        logger.info("Trainer initialized successfully")
    
    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            epoch_loss += losses.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {self.current_epoch + 1}/{self.num_epochs} - "
                    f"Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {losses.item():.4f}"
                )
        
        avg_loss = epoch_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.train()  # SSD requires train mode for loss calculation
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                epoch_loss += losses.item()
                num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        return avg_loss
    
    def train(
        self,
        save_path: Path,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Args:
            save_path: Path to save the trained model
            callback: Optional callback function for progress updates
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Freeze backbone for warmup
            if epoch == 0:
                logger.info("Freezing backbone for warmup")
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            if epoch == 5:
                logger.info("Unfreezing backbone")
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
            
            # Train
            train_loss = self.train_one_epoch()
            self.training_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.validation_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"LR: {current_lr:.6f}"
            )
            
            # Callback
            if callback:
                callback({
                    'epoch': epoch + 1,
                    'total_epochs': self.num_epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': current_lr
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Get class mappings
                class_id_mapping = {
                    'class_id_to_idx': getattr(self.train_loader.dataset, 'class_id_to_idx', None),
                    'idx_to_class_id': getattr(self.train_loader.dataset, 'idx_to_class_id', None)
                }
                
                self.detection_model.save(save_path, self.class_names, class_id_mapping)
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'num_epochs': self.num_epochs,
            'final_train_loss': self.training_losses[-1],
            'final_val_loss': self.validation_losses[-1],
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'model_path': str(save_path)
        }