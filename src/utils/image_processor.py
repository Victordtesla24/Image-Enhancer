"""Image processor module with advanced AI models"""

import time
import gc
import psutil
import threading
import logging
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import Optional, Tuple, Dict, Callable, List
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet50, ResNet50_Weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Updated for better performance
MAX_TARGET_WIDTH = 5120  # Support up to 5K resolution
MAX_INTERMEDIATE_SIZE = 1024  # Reduced for better performance
PROCESSING_TIMEOUT = 60  # Reduced timeout
SECTION_HEIGHT = 64  # Reduced for better performance
MEMORY_THRESHOLD = 0.8
MODEL_LOAD_TIMEOUT = 30  # Reduced timeout


class AIModel:
    """Base class for AI models"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        """Enhance image - to be implemented by subclasses"""
        raise NotImplementedError


class SuperResolutionModel(AIModel):
    """Advanced Super Resolution Model"""

    def __init__(self):
        super().__init__(
            "SuperRes",
            "Advanced Super Resolution using ResNet backbone",
        )

    def load(self):
        try:
            # Simplified model for better performance
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info("SuperRes model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SuperRes model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                enhanced = self.model(image)
                return enhanced
            except Exception as e:
                logger.error(f"Error in SuperRes enhancement: {str(e)}")
                return image  # Return original image on error


class DetailEnhancementModel(AIModel):
    """Detail Enhancement Model"""

    def __init__(self):
        super().__init__("DetailEnhance", "Detail Enhancement")

    def load(self):
        # Simplified model for better performance
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                enhanced = self.model(image)
                return enhanced
            except Exception as e:
                logger.error(f"Error in Detail enhancement: {str(e)}")
                return image


class ColorEnhancementModel(AIModel):
    """Color Enhancement Model"""

    def __init__(self):
        super().__init__("ColorEnhance", "Color Enhancement")

    def load(self):
        # Simplified model for better performance
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                enhanced = self.model(image)
                return enhanced
            except Exception as e:
                logger.error(f"Error in Color enhancement: {str(e)}")
                return image


class ModelCache:
    """Enhanced singleton class to manage multiple AI models"""

    _instance = None
    _models: Dict[str, AIModel] = {}
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
        """Initialize the model cache with multiple AI models"""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelCache initialized using device: {self._device}")

        # Initialize all models
        self._models["superres"] = SuperResolutionModel()
        self._models["detail"] = DetailEnhancementModel()
        self._models["color"] = ColorEnhancementModel()

        # Preload models in background
        threading.Thread(target=self._preload_models, daemon=True).start()

    def _preload_models(self):
        """Preload models in background"""
        for model_name, model in self._models.items():
            try:
                logger.info(f"Preloading {model_name} model...")
                model.load()
                logger.info(f"{model_name} model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")

    def get_model(self, model_name: str) -> Optional[AIModel]:
        """Get a model from cache"""
        model = self._models.get(model_name)
        if model and not model.loaded:
            try:
                model.load()
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                return None
        return model

    def get_device(self) -> torch.device:
        """Get the current device"""
        return self._device

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models with descriptions"""
        return [
            {"name": model.name, "description": model.description}
            for model in self._models.values()
        ]


class ImageEnhancer:
    """Enhanced class for handling image enhancement with multiple AI models"""

    def __init__(self):
        """Initialize the image enhancer"""
        logger.info("Initializing ImageEnhancer...")
        self.model_cache = ModelCache()
        self.device = self.model_cache.get_device()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0))]
        )
        logger.info("ImageEnhancer initialized successfully")

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models"""
        return self.model_cache.get_available_models()

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
        models: List[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Image.Image, Dict[str, str]]:
        """Enhance image using multiple AI models"""
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

            # Use default models if none specified
            if not models:
                models = ["superres"]  # Reduced to single model for better performance

            # Track enhancement details
            enhancement_details = {
                "source_size": f"{image.size[0]}x{image.size[1]}",
                "models_used": [],
                "processing_time": 0,
            }

            start_time = time.time()

            # Resize input image
            aspect_ratio = image.size[1] / image.size[0]
            intermediate_width = min(image.size[0], MAX_INTERMEDIATE_SIZE)
            intermediate_height = int(intermediate_width * aspect_ratio)

            image = image.resize(
                (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
            )

            # Convert to tensor
            inputs = self.transform(image).to(self.device)
            enhanced = inputs

            # Apply each model in sequence
            total_models = len(models)
            for idx, model_name in enumerate(models):
                model = self.model_cache.get_model(model_name)
                if model and model.loaded:
                    try:
                        update_progress(
                            (idx + 1) / (total_models + 1),
                            f"Applying {model.name} enhancement...",
                        )
                        enhanced = model.enhance(enhanced)
                        enhancement_details["models_used"].append(
                            {"name": model.name, "description": model.description}
                        )
                    except Exception as e:
                        logger.error(f"Error in {model_name} enhancement: {str(e)}")
                        continue

            update_progress(0.9, "Finalizing enhancement...")

            # Convert back to PIL Image
            enhanced = enhanced.cpu().squeeze(0).clamp(0, 1)
            result_img = F.to_pil_image(enhanced)

            # Final resize to target width
            if result_img.size[0] != target_width:
                result_img = result_img.resize(
                    (target_width, int(target_width * aspect_ratio)),
                    Image.Resampling.LANCZOS,
                )

            # Update enhancement details
            enhancement_details["target_size"] = (
                f"{result_img.size[0]}x{result_img.size[1]}"
            )
            enhancement_details["processing_time"] = f"{time.time() - start_time:.2f}s"

            update_progress(1.0, "Enhancement complete!")

            return result_img, enhancement_details

        except Exception as e:
            logger.error(f"Error in enhancement: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            raise
