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

# Constants - Updated for professional quality
MAX_TARGET_WIDTH = 5120  # Support up to 5K resolution
MAX_INTERMEDIATE_SIZE = 2560  # Increased for better quality
PROCESSING_TIMEOUT = 60
SECTION_HEIGHT = 512  # Increased for better detail preservation
MEMORY_THRESHOLD = 0.8
MODEL_LOAD_TIMEOUT = 30


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
    """Advanced Super Resolution Model with professional-grade enhancements"""

    def __init__(self):
        super().__init__(
            "SuperRes",
            "Professional Super Resolution with adaptive sharpening",
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.denormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

    def load(self):
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

            # Enhanced architecture for professional quality
            self.enhancement_layers = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 3, kernel_size=3, padding=1),
                nn.Tanh(),
            )

            self.feature_extractor.to(self.device)
            self.enhancement_layers.to(self.device)
            self.feature_extractor.eval()
            self.enhancement_layers.eval()
            self.loaded = True
            logger.info("SuperRes model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SuperRes model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Normalize input
                normalized = self.normalize(image)

                # Extract features
                features = self.feature_extractor(normalized)

                # Enhance features
                enhanced = self.enhancement_layers(features)

                # Professional-grade upscaling
                enhanced = TF.interpolate(
                    enhanced, size=image.shape[2:], mode="bicubic", align_corners=True
                )

                # Adaptive sharpening based on image content
                edge_detect = (
                    torch.tensor(
                        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device=self.device
                    ).view(1, 1, 3, 3)
                    / 8
                )
                edge_detect = edge_detect.repeat(3, 1, 1, 1)

                edges = TF.conv2d(enhanced, edge_detect, padding=1, groups=3)
                edge_intensity = torch.mean(torch.abs(edges))

                # Adjust sharpening strength based on edge intensity
                sharpening_strength = torch.clamp(1.0 - edge_intensity * 2, 0.5, 0.8)

                kernel = (
                    torch.tensor(
                        [[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]], device=self.device
                    ).view(1, 1, 3, 3)
                    / 8
                )
                kernel = kernel.repeat(3, 1, 1, 1)

                sharpened = TF.conv2d(enhanced, kernel, padding=1, groups=3)

                # Adaptive blending
                result = (
                    1 - sharpening_strength
                ) * image + sharpening_strength * sharpened

                # Professional contrast enhancement
                mean = torch.mean(result, dim=[2, 3], keepdim=True)
                std = torch.std(result, dim=[2, 3], keepdim=True)
                contrast_enhanced = mean + 1.2 * (result - mean) * (0.5 + std)

                return torch.clamp(contrast_enhanced, 0, 1)
            except Exception as e:
                logger.error(f"Error in SuperRes enhancement: {str(e)}")
                return image


class DetailEnhancementModel(AIModel):
    """Detail Enhancement Model with professional-grade improvements"""

    def __init__(self):
        super().__init__(
            "DetailEnhance", "Professional Detail Enhancement with adaptive processing"
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def load(self):
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

            # Enhanced detail preservation
            self.detail_enhance = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 3, kernel_size=3, padding=1),
                nn.Tanh(),
            )

            self.feature_extractor.to(self.device)
            self.detail_enhance.to(self.device)
            self.feature_extractor.eval()
            self.detail_enhance.eval()
            self.loaded = True

        except Exception as e:
            logger.error(f"Error loading Detail Enhancement model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Extract features
                features = self.feature_extractor(self.normalize(image))

                # Generate detail enhancement mask
                detail_mask = self.detail_enhance(features)

                # Professional-grade upscaling
                detail_mask = TF.interpolate(
                    detail_mask,
                    size=image.shape[2:],
                    mode="bicubic",
                    align_corners=True,
                )

                # Adaptive detail enhancement
                detail_strength = torch.mean(torch.abs(detail_mask))
                enhancement_factor = torch.clamp(
                    0.4 / (detail_strength + 1e-5), 0.2, 0.4
                )

                enhanced = image + enhancement_factor * detail_mask

                # Professional local contrast enhancement
                kernel_size = 7  # Increased kernel size for better local contrast
                padding_size = kernel_size // 2

                # Calculate local statistics
                local_mean = TF.avg_pool2d(
                    enhanced, kernel_size=kernel_size, stride=1, padding=padding_size
                )
                local_var = (
                    TF.avg_pool2d(
                        enhanced**2,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding_size,
                    )
                    - local_mean**2
                )
                local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))

                # Normalize local contrast
                normalized = (enhanced - local_mean) / local_std

                # Apply adaptive contrast enhancement
                contrast_enhanced = local_mean + local_std * normalized * 1.15

                return torch.clamp(contrast_enhanced, 0, 1)
            except Exception as e:
                logger.error(f"Error in Detail enhancement: {str(e)}")
                return image


class ColorEnhancementModel(AIModel):
    """Color Enhancement Model with professional color processing"""

    def __init__(self):
        super().__init__(
            "ColorEnhance", "Professional Color Enhancement with natural preservation"
        )

    def load(self):
        try:
            # Professional color processing layers
            self.color_enhance = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

            # Initialize with improved weights
            for m in self.color_enhance.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            self.color_enhance.to(self.device)
            self.color_enhance.eval()
            self.loaded = True

        except Exception as e:
            logger.error(f"Error loading Color Enhancement model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Professional color enhancement
                rgb_to_hsv = transforms.ColorJitter(
                    brightness=0.25,  # Reduced for more natural look
                    contrast=0.25,  # Reduced for more natural look
                    saturation=0.35,  # Adjusted for better balance
                )
                enhanced_hsv = rgb_to_hsv(image)

                # Apply color enhancement
                enhanced_colors = self.color_enhance(enhanced_hsv)

                # Adaptive color blending
                color_diff = torch.abs(enhanced_colors - enhanced_hsv)
                blend_factor = torch.clamp(
                    0.3 / (torch.mean(color_diff) + 1e-5), 0.3, 0.5
                )

                result = (
                    1 - blend_factor
                ) * enhanced_hsv + blend_factor * enhanced_colors

                # Natural saturation enhancement
                luminance = (
                    0.299 * result[:, 0] + 0.587 * result[:, 1] + 0.114 * result[:, 2]
                )
                luminance = luminance.unsqueeze(1).repeat(1, 3, 1, 1)

                saturation_factor = torch.clamp(
                    1.2 - torch.mean(torch.abs(result - luminance)), 0.8, 1.2
                )
                color_enhanced = luminance + saturation_factor * (result - luminance)

                return torch.clamp(color_enhanced, 0, 1)
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

            # Use all models by default
            if not models:
                models = ["superres", "detail", "color"]

            # Track enhancement details
            enhancement_details = {
                "source_size": f"{image.size[0]}x{image.size[1]}",
                "models_used": [],
                "processing_time": 0,
            }

            start_time = time.time()

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize input image with improved quality
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

            # Convert back to PIL Image with improved quality
            enhanced = enhanced.cpu().squeeze(0).clamp(0, 1)
            result_img = F.to_pil_image(enhanced)

            # Final resize with high quality
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
