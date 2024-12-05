"""Image processor module"""

import sys
from huggingface_hub import hf_hub_download


# Patch huggingface_hub compatibility before importing super_image
def patched_cached_download(url_or_filename, **kwargs):
    """Compatibility wrapper for cached_download"""
    if url_or_filename.startswith("http"):
        # Extract repo_id and filename from URL
        parts = url_or_filename.split("/")
        repo_id = f"{parts[3]}/{parts[4]}"  # e.g., "eugenesiow/edsr-base"
        filename = parts[-1]  # e.g., "config.json"
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return hf_hub_download(
        repo_id="eugenesiow/edsr-base", filename=url_or_filename, **kwargs
    )


sys.modules["huggingface_hub"].cached_download = patched_cached_download

from PIL import Image
import logging
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Optional, Union
import os
import time
import functools
import math
import signal
from contextlib import contextmanager

try:
    from super_image import EdsrModel
    from super_image.modeling_utils import PreTrainedModel

    # Patch torch.load to always use weights_only=True
    original_load = torch.load

    @functools.wraps(original_load)
    def safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    # Apply the patch to both torch and the model's internal load function
    torch.load = safe_torch_load
    PreTrainedModel._load_state_dict_into_model = staticmethod(safe_torch_load)

except ImportError:
    raise ImportError(
        "Could not import super_image. Please install it with: pip install super-image"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TARGET_WIDTH = 7680  # Updated maximum allowed target width to support 8K resolution
MAX_PROCESSING_TIME = 30  # Increased processing time for larger images
MAX_INTERMEDIATE_SIZE = 640  # Increased intermediate size for better quality


class TimeoutError(Exception):
    """Raised when processing time exceeds limit"""

    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout"""

    def signal_handler(signum, frame):
        raise TimeoutError("Processing timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ImageEnhancer:
    """Class for handling image enhancement using EDSR model"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the image enhancer with model path"""
        logger.info("Initializing ImageEnhancer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Load model
        try:
            if model_path:
                logger.info("Loading model from path: %s", model_path)
                self.model = EdsrModel.from_pretrained(model_path, scale=2)
            else:
                # Use cached model with hf_hub_download
                cache_dir = os.path.expanduser("~/.cache/image_enhancer")
                os.makedirs(cache_dir, exist_ok=True)

                # Download model file
                logger.info("Downloading model if needed...")
                model_file = hf_hub_download(
                    repo_id="eugenesiow/edsr-base",
                    filename="pytorch_model_2x.pt",
                    cache_dir=cache_dir,
                    force_download=True,  # Force fresh download to avoid cache issues
                )

                # Load the model
                logger.info("Loading model from cache...")
                start_time = time.time()

                # Initialize model with minimal parameters
                self.model = EdsrModel.from_pretrained(
                    "eugenesiow/edsr-base", scale=2, cache_dir=cache_dir
                )

                logger.info("Model loaded in %.2fs", time.time() - start_time)

        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Move model to device and set eval mode
        logger.info("Moving model to device and setting eval mode...")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define transforms with normalization
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0))]
        )
        logger.info("ImageEnhancer initialized successfully")

    def get_model_device(self) -> str:
        """Get the device where the model is located"""
        return str(self.device)

    def enhance_image(
        self, image: Image.Image, target_width: int = MAX_TARGET_WIDTH
    ) -> Image.Image:
        """
        Enhance a single image using EDSR model

        Args:
            image: Input PIL Image
            target_width: Desired width of output image (max 7680)

        Returns:
            Enhanced PIL Image at target width

        Raises:
            ValueError: If input image is None or invalid
            ValueError: If target width is invalid
            RuntimeError: For processing errors
            TimeoutError: If processing takes too long
        """
        try:
            # Input validation
            if image is None:
                raise ValueError("Input image cannot be None")
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            if target_width <= 0:
                raise ValueError("Target width must be positive")

            # Enforce maximum target width
            target_width = min(target_width, MAX_TARGET_WIDTH)
            logger.info(f"Using target width: {target_width} (max: {MAX_TARGET_WIDTH})")

            # Start timing
            start_time = time.time()

            # Calculate intermediate size for processing
            aspect_ratio = image.size[1] / image.size[0]
            intermediate_width = min(image.size[0], MAX_INTERMEDIATE_SIZE)
            intermediate_height = int(intermediate_width * aspect_ratio)

            logger.info(
                f"Processing at intermediate size: {(intermediate_width, intermediate_height)}"
            )

            # Initial resize using Lanczos resampling for better quality
            image = image.resize(
                (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
            )

            with time_limit(MAX_PROCESSING_TIME):
                # Convert to tensor
                logger.info("Converting to tensor...")
                inputs = self.transform(image).to(self.device)

                # Process image
                logger.info("Running enhancement model...")
                with torch.no_grad():
                    try:
                        enhanced = self.model(inputs)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            raise RuntimeError(
                                "Out of memory. Try reducing target width."
                            )
                        raise

                # Convert back to PIL Image
                logger.info("Converting result back to PIL Image...")
                enhanced = enhanced.cpu().squeeze(0).clamp(0, 1)
                result_img = F.to_pil_image(enhanced)

                # Final resize to target width using Lanczos resampling for better quality
                if result_img.size[0] != target_width:
                    logger.info(f"Resizing to target width: {target_width}")
                    result_img = result_img.resize(
                        (target_width, int(target_width * aspect_ratio)),
                        Image.Resampling.LANCZOS,
                    )

            process_time = time.time() - start_time
            logger.info(f"Image enhancement completed in {process_time:.2f}s")

            return result_img

        except TimeoutError:
            logger.error("Processing timeout")
            raise
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise
