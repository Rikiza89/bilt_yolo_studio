"""
BILT (Because I Like Twice) - A PyTorch-based object detection library.

Example usage:
    from bilt import BILT
    
    # Load model
    model = BILT("model.pth")
    
    # Predict
    results = model.predict("image.jpg", conf=0.25)
    
    # Train
    model.train(dataset="datasets/my_dataset", epochs=50)
"""

from .model import BILT
from .utils import set_logging_level

__version__ = "0.1.0"
__all__ = ["BILT", "set_logging_level"]