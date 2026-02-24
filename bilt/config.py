import platform
import os

class Config:
    """Default configuration for BILT."""
    
    # Detect platform
    IS_WINDOWS = platform.system() == 'Windows'
    IS_ARM = platform.machine().lower() in ['aarch64', 'armv7l', 'armv8']
    IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and IS_ARM
    
    # Training defaults - optimized per platform
    if IS_RASPBERRY_PI:
        DEFAULT_BATCH_SIZE = 2
        DEFAULT_EPOCHS = 30
        DEFAULT_LEARNING_RATE = 0.0005
        DEFAULT_NUM_WORKERS = 0
        DEFAULT_INPUT_SIZE = 320
    elif IS_ARM:
        DEFAULT_BATCH_SIZE = 2
        DEFAULT_EPOCHS = 100
        DEFAULT_LEARNING_RATE = 0.0005
        DEFAULT_NUM_WORKERS = 0
        DEFAULT_INPUT_SIZE = 480
    elif IS_WINDOWS:
        DEFAULT_BATCH_SIZE = 4
        DEFAULT_EPOCHS = 100
        DEFAULT_LEARNING_RATE = 0.0005
        DEFAULT_NUM_WORKERS = 0
        DEFAULT_INPUT_SIZE = 640
    else:
        DEFAULT_BATCH_SIZE = 4
        DEFAULT_EPOCHS = 100
        DEFAULT_LEARNING_RATE = 0.001
        DEFAULT_NUM_WORKERS = 2
        DEFAULT_INPUT_SIZE = 640
    
    # Detection defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_NMS_THRESHOLD = 0.4
    
    # Platform-specific optimizations
    if IS_RASPBERRY_PI:
        import torch
        torch.set_num_threads(2)
        os.environ['OMP_NUM_THREADS'] = '2'