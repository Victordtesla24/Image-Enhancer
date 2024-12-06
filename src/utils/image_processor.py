"""Main image processor integrating all enhancement systems"""

import logging
import time
import hashlib
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
from .model_management.model_manager import ModelManager
from .session_management.session_manager import SessionManager
from .quality_management.quality_manager import QualityManager
from .enhancers.super_resolution import SuperResolutionEnhancer
from .enhancers.color_enhancement import ColorEnhancer
from .enhancers.detail_enhancement import DetailEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Main image enhancement coordinator with advanced management systems"""

    def __init__(self, session_id: Optional[str] = None):
        logger.info("Initializing ImageEnhancer")

        # Load configuration
        self.config = self._load_config()

        # Initialize management systems
        self.session_manager = SessionManager(session_id)
        self.model_manager = self.session_manager.model_manager
        self.quality_manager = QualityManager(self.config)

        # Initialize enhancement models
        self.super_res = SuperResolutionEnhancer()
        self.color = ColorEnhancer()
        self.detail = DetailEnhancer()

        # Initialize models list with descriptions
        self.models = [
            {
                "name": "Super Resolution",
                "description": "Intelligently upscales image resolution using advanced multi-step processing with Lanczos resampling and adaptive sharpening",
                "internal_name": "super_resolution",
            },
            {
                "name": "Color Enhancement",
                "description": "Optimizes color balance and vibrancy using LAB color space processing and adaptive contrast enhancement",
                "internal_name": "color_enhancement",
            },
            {
                "name": "Detail Enhancement",
                "description": "Enhances image details using multi-scale contrast enhancement and advanced noise reduction techniques",
                "internal_name": "detail_enhancement",
            },
        ]

        logger.info("ImageEnhancer initialized successfully")

    def _load_config(self):
        """Load configuration from session manager"""
        return {
            "resolution": {"width": 5120, "height": 2880},
            "quality": {
                "dpi": 300,
                "min_sharpness": 70,
                "max_noise_level": 120,
                "min_file_size_mb": 1.5,
            },
            "color": {"bit_depth": 24, "dynamic_range": {"min": 220, "max": 255}},
        }

    def get_available_models(self):
        """Return list of available enhancement models"""
        return [
            {"name": model["name"], "description": model["description"]}
            for model in self.models
        ]

    def _compute_image_hash(self, image: Image.Image) -> str:
        """Compute unique hash for image"""
        return hashlib.md5(np.array(image).tobytes()).hexdigest()

    def enhance_image(
        self,
        input_image: Image.Image,
        target_width: int,
        models: List[str],
        progress_callback=None,
        retry_count: int = 0,
    ) -> Tuple[Image.Image, Dict]:
        """Enhance image using selected models with quality validation and retry capability"""
        try:
            logger.info(f"Starting image enhancement with models: {models}")
            start_time = time.time()

            # Compute image hash for tracking
            image_hash = self._compute_image_hash(input_image)

            # Get enhancement suggestions if available
            suggested_params = self.session_manager.get_enhancement_suggestions(
                image_hash
            )

            # Store original size and compute initial quality metrics
            source_size = input_image.size
            initial_metrics = self.quality_manager.compute_quality_metrics(input_image)
            logger.info(f"Source image size: {source_size}")

            # Convert PIL Image to numpy array
            img_array = np.array(input_image)

            # Track which models were used
            models_used = []
            parameters_used = {}

            # Calculate total steps for progress tracking
            total_steps = len(models)
            current_step = 0

            # Process image with selected models
            for model_name in models:
                logger.info(f"Processing with model: {model_name}")
                if progress_callback:
                    progress_callback(
                        current_step / total_steps, f"Applying {model_name}..."
                    )

                # Get internal model name
                internal_name = model_name.lower().replace(" ", "_")

                # Get model parameters
                model_params = suggested_params.get(
                    internal_name,
                    self.model_manager.get_model_parameters(internal_name),
                )

                # Apply appropriate enhancement based on model
                if internal_name == "super_resolution":
                    img_array = self.super_res.enhance(img_array, target_width)
                    models_used.append("Super Resolution")
                    parameters_used["super_resolution"] = model_params

                elif internal_name == "color_enhancement":
                    img_array = self.color.enhance(img_array)
                    models_used.append("Color Enhancement")
                    parameters_used["color_enhancement"] = model_params

                elif internal_name == "detail_enhancement":
                    img_array = self.detail.enhance(img_array)
                    models_used.append("Detail Enhancement")
                    parameters_used["detail_enhancement"] = model_params

                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps, f"Processing...")

            # Convert final result to PIL Image
            enhanced_img = Image.fromarray(img_array)
            enhanced_img.info["dpi"] = (300, 300)  # Set DPI for enhanced image

            # Compute final quality metrics
            final_metrics = self.quality_manager.compute_quality_metrics(
                enhanced_img, original=input_image
            )

            # Validate quality
            quality_passed, quality_results = self.quality_manager.validate_quality(
                final_metrics
            )

            if not quality_passed and retry_count < 3:
                # Get improvement suggestions
                suggestions = self.quality_manager.suggest_improvements(final_metrics)

                # Adjust parameters based on suggestions
                self.model_manager.adapt_parameters(
                    "detail_enhancement",
                    {"sharpness": 1 if "sharpness" in suggestions else -1},
                )

                # Retry enhancement with adjusted parameters
                logger.info(
                    f"Quality check failed. Retrying enhancement (attempt {retry_count + 1})"
                )
                return self.enhance_image(
                    input_image,
                    target_width,
                    models,
                    progress_callback,
                    retry_count + 1,
                )

            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Enhancement completed in {processing_time:.2f} seconds")

            # Record enhancement attempt
            self.session_manager.record_enhancement_attempt(
                input_image_hash=image_hash,
                models_used=models_used,
                parameters=parameters_used,
                quality_metrics=quality_results,
                success=quality_passed,
            )

            # Prepare enhancement details
            enhancement_details = {
                "source_size": f"{source_size[0]}x{source_size[1]}",
                "target_size": f"{enhanced_img.size[0]}x{enhanced_img.size[1]}",
                "models_used": [
                    {
                        "name": model,
                        "parameters": parameters_used.get(
                            model.lower().replace(" ", "_"), {}
                        ),
                    }
                    for model in models_used
                ],
                "processing_time": f"{processing_time:.2f} seconds",
                "quality_results": quality_results,
                "retry_count": retry_count,
            }

            return enhanced_img, enhancement_details

        except Exception as e:
            logger.error(f"Error during image enhancement: {str(e)}")
            raise

    def apply_feedback(self, image_hash: str, feedback: Dict):
        """Apply user feedback to enhance model performance"""
        self.session_manager.apply_feedback(image_hash, feedback)

    def get_quality_preferences(self) -> Dict:
        """Get current quality preferences"""
        return self.session_manager.quality_preferences.__dict__

    def update_quality_preferences(self, preferences: Dict):
        """Update quality preferences"""
        self.session_manager.update_quality_preferences(preferences)

    def get_enhancement_history(self, image_hash: Optional[str] = None) -> Dict:
        """Get enhancement history with quality metrics"""
        history = self.session_manager.get_enhancement_history(image_hash)
        metrics_summary = self.session_manager.get_quality_metrics_summary(image_hash)

        return {"history": history, "metrics_summary": metrics_summary}
