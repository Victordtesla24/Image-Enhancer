"""Main image processor module orchestrating multiple AI enhancement models"""

import time
import gc
import psutil
import logging
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import Optional, Tuple, Dict, Callable, List
from .models.super_resolution import SuperResolutionModel
from .models.detail_enhancement import DetailEnhancementModel
from .models.color_enhancement import ColorEnhancementModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TARGET_WIDTH = 5120
MAX_INTERMEDIATE_SIZE = 2048
MEMORY_THRESHOLD = 0.7


class ModelCache:
    """Singleton class to manage multiple AI models with lazy loading"""

    _instance = None
    _models: Dict[str, any] = {}
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model cache with lazy loading"""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelCache initialized using device: {self._device}")

        # Initialize model references (lazy loading)
        self._models["superres"] = SuperResolutionModel()
        self._models["detail"] = DetailEnhancementModel()
        self._models["color"] = ColorEnhancementModel()

    def get_model(self, model_name: str):
        """Get a model from cache with lazy loading"""
        model = self._models.get(model_name)
        if model and not model.loaded:
            try:
                model.load()
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                return None
        return model

    def cleanup(self):
        """Cleanup all models"""
        for model in self._models.values():
            if model.loaded:
                model.cleanup()
        torch.cuda.empty_cache()
        gc.collect()

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
        """Enhance image using multiple AI models with improved memory management"""
        try:

            def update_progress(progress: float, status: str):
                if progress_callback:
                    progress_callback(progress, status)
                logger.info(f"Progress {progress*100:.0f}%: {status}")

            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            if target_width <= 0:
                raise ValueError("Target width must be positive")

            if not models:
                models = ["detail", "superres", "color"]  # Optimized order

            enhancement_details = {
                "source_size": f"{image.size[0]}x{image.size[1]}",
                "models_used": [],
                "processing_time": 0,
            }

            start_time = time.time()

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Optimize intermediate size based on available memory
            gpu_memory, ram_memory = self._check_memory_usage()
            if gpu_memory > MEMORY_THRESHOLD or ram_memory > MEMORY_THRESHOLD:
                logger.warning("High memory usage detected, reducing intermediate size")
                intermediate_size = 1024
            else:
                intermediate_size = MAX_INTERMEDIATE_SIZE

            # Resize input for processing
            aspect_ratio = image.size[1] / image.size[0]
            intermediate_width = min(image.size[0], intermediate_size)
            intermediate_height = int(intermediate_width * aspect_ratio)

            image = image.resize(
                (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
            )

            # Convert to tensor
            inputs = self.transform(image).to(self.device)
            enhanced = inputs

            # Apply models in sequence
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

                        # Cleanup after each model except the last
                        if idx < total_models - 1:
                            model.cleanup()

                    except Exception as e:
                        logger.error(f"Error in {model_name} enhancement: {str(e)}")
                        continue

            update_progress(0.9, "Finalizing enhancement...")

            # Convert back to image
            enhanced = enhanced.cpu().squeeze(0).clamp(0, 1)
            result_img = F.to_pil_image(enhanced)

            # Final resize to target width
            if result_img.size[0] != target_width:
                result_img = result_img.resize(
                    (target_width, int(target_width * aspect_ratio)),
                    Image.Resampling.LANCZOS,
                )

            enhancement_details["target_size"] = (
                f"{result_img.size[0]}x{result_img.size[1]}"
            )
            enhancement_details["processing_time"] = f"{time.time() - start_time:.2f}s"

            # Final cleanup
            self.model_cache.cleanup()

            update_progress(1.0, "Enhancement complete!")

            return result_img, enhancement_details

        except Exception as e:
            logger.error(f"Error in enhancement: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            raise
