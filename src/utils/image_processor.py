"""Image processor module"""

import time
import gc
import psutil
import threading
import logging
import os
from typing import Optional, Tuple, Dict, Callable
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TARGET_WIDTH = 2048  # Reduced for better performance
MAX_INTERMEDIATE_SIZE = 512  # Reduced for better memory usage
PROCESSING_TIMEOUT = 60  # Reduced timeout to 1 minute
SECTION_HEIGHT = 32  # Smaller sections for better memory management
MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold
MODEL_LOAD_TIMEOUT = 30  # 30 seconds timeout for model loading


class DummyModel:
    """Fallback model that performs basic upscaling"""

    def __init__(self):
        self.device = torch.device("cpu")
        self.scale = 4

    def __call__(self, x):
        # Use bicubic upscaling as fallback
        return F.resize(
            x,
            size=[s * self.scale for s in x.shape[-2:]],
            interpolation=F.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class ModelCache:
    """Singleton class to manage model instances"""

    _instance = None
    _models: Dict[str, DummyModel] = {}
    _device = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model cache"""
        # Prefer CPU for testing and small images
        self._device = torch.device("cpu")
        logger.info(f"ModelCache initialized using device: {self._device}")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model"""
        try:
            # Initialize with dummy model for reliable operation
            self._models["edsr_4x"] = DummyModel()
            self._models["edsr_4x"].to(self._device)
            self._models["edsr_4x"].eval()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def get_model(self, model_name: str = "edsr_4x") -> Optional[DummyModel]:
        """Get a model from cache"""
        return self._models.get(model_name)

    def get_device(self) -> torch.device:
        """Get the current device"""
        return self._device


class ImageEnhancer:
    """Class for handling image enhancement"""

    def __init__(self):
        """Initialize the image enhancer"""
        logger.info("Initializing ImageEnhancer...")
        self.model_cache = ModelCache()
        self.device = self.model_cache.get_device()
        self.model = self.model_cache.get_model("edsr_4x")

        if self.model is None:
            raise RuntimeError("Failed to initialize model")

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0))]
        )
        logger.info("ImageEnhancer initialized successfully")

    def _check_memory_usage(self) -> Tuple[float, float]:
        """Check current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
            )
            return gpu_memory, psutil.virtual_memory().percent / 100
        return 0, psutil.virtual_memory().percent / 100

    def enhance_image(
        self,
        image: Image.Image,
        target_width: int = MAX_TARGET_WIDTH,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Image.Image:
        """Enhance a single image"""
        try:

            def update_progress(progress: float, status: str):
                if progress_callback:
                    progress_callback(progress, status)
                logger.info(f"Progress {progress*100:.0f}%: {status}")

            # Input validation
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            if target_width <= 0:
                raise ValueError("Target width must be positive")

            # Enforce maximum target width
            target_width = min(target_width, MAX_TARGET_WIDTH)

            # Start timing
            start_time = time.time()

            # Resize input image
            aspect_ratio = image.size[1] / image.size[0]
            intermediate_width = min(image.size[0], MAX_INTERMEDIATE_SIZE)
            intermediate_height = int(intermediate_width * aspect_ratio)

            image = image.resize(
                (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
            )

            update_progress(0.3, "Processing image...")

            # Convert to tensor
            inputs = self.transform(image).to(self.device)

            # Process image
            with torch.no_grad():
                try:
                    enhanced = self.model(inputs)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        raise RuntimeError("Out of memory. Try reducing target width.")
                    raise

            update_progress(0.7, "Finalizing enhancement...")

            # Convert back to PIL Image
            enhanced = enhanced.cpu().squeeze(0).clamp(0, 1)
            result_img = F.to_pil_image(enhanced)

            # Final resize to target width
            if result_img.size[0] != target_width:
                result_img = result_img.resize(
                    (target_width, int(target_width * aspect_ratio)),
                    Image.Resampling.LANCZOS,
                )

            process_time = time.time() - start_time
            logger.info(f"Enhancement completed in {process_time:.2f}s")

            update_progress(1.0, "Enhancement complete!")

            return result_img

        except Exception as e:
            logger.error(f"Error in enhancement: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            raise
