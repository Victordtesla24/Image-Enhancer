"""Image processing module."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .core.gpu_accelerator import GPUAccelerator
from .model_management.model_manager import ModelManager
from .quality_management.quality_manager import QualityManager
from .session_management.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Main image processing class."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize image processor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_manager = ModelManager()
        self.quality_manager = QualityManager()
        self.session_manager = SessionManager()
        self.gpu_accelerator = GPUAccelerator()

        # Initialize processing parameters
        self.processing_params = {
            "quality_threshold": self.config.get("quality_threshold", 0.8),
            "max_iterations": self.config.get("max_iterations", 5),
            "use_gpu": self.config.get("use_gpu", True),
        }

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path.

        Args:
            image_path: Path to image file

        Returns:
            Image array

        Raises:
            FileNotFoundError: If image file not found
            ValueError: If image cannot be loaded
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def save_image(self, image: np.ndarray, output_path: Union[str, Path]):
        """Save image to path.

        Args:
            image: Image array
            output_path: Path to save image
        """
        self._save_image(image, output_path)

    def enhance_image(
        self, image: np.ndarray, enhancement_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Enhance image with given parameters.

        Args:
            image: Input image array
            enhancement_params: Optional enhancement parameters

        Returns:
            Tuple of (enhanced image array, quality metrics)
        """
        # Update processing parameters
        if enhancement_params:
            self.processing_params.update(enhancement_params)

        # Start processing session
        session_id = self.session_manager.start_session()

        try:
            # Initial quality assessment
            initial_metrics = self.quality_manager.calculate_quality_metrics(image)

            # Apply enhancements
            enhanced = self._apply_enhancements(image)

            # Final quality assessment
            final_metrics = self.quality_manager.calculate_quality_metrics(enhanced)

            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                initial_metrics, final_metrics
            )

            # Update session metrics
            self.session_manager.update_session_metrics(
                session_id, initial_metrics, final_metrics, improvement_metrics
            )

            return enhanced, improvement_metrics

        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            self.session_manager.mark_session_error(session_id, str(e))
            return image, {}

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Process an image with enhancement models.

        Args:
            image_path: Path to input image
            output_path: Optional path to save enhanced image
            params: Optional processing parameters

        Returns:
            Tuple of (enhanced image array, quality metrics)
        """
        # Load and validate image
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Update processing parameters
        if params:
            self.processing_params.update(params)

        # Start processing session
        session_id = self.session_manager.start_session()

        try:
            # Initial quality assessment
            initial_metrics = self.quality_manager.calculate_quality_metrics(image)

            # Apply enhancements
            enhanced = self._apply_enhancements(image)

            # Final quality assessment
            final_metrics = self.quality_manager.calculate_quality_metrics(enhanced)

            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                initial_metrics, final_metrics
            )

            # Update session metrics
            self.session_manager.update_session_metrics(
                session_id, initial_metrics, final_metrics, improvement_metrics
            )

            return enhanced, improvement_metrics

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.session_manager.mark_session_error(session_id, str(e))
            return image, {}

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        enhancement_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Process multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Optional output directory
            enhancement_params: Optional enhancement parameters

        Returns:
            List of result dictionaries
        """
        results = []
        output_dir = Path(output_dir) if output_dir else None

        for i, image_path in enumerate(image_paths):
            try:
                # Create output path if needed
                output_path = None
                if output_dir:
                    output_path = output_dir / f"enhanced_{Path(image_path).name}"

                # Process image
                enhanced, metrics = self.process_image(
                    image_path, output_path, enhancement_params
                )

                results.append(
                    {
                        "enhanced_image": enhanced,
                        "metrics": metrics,
                        "output_path": str(output_path) if output_path else None,
                    }
                )

                logger.info(f"Processed image {i+1}/{len(image_paths)}: {image_path}")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({"error": str(e)})

        return results

    def _load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load image from path.

        Args:
            image_path: Path to image file

        Returns:
            Image array or None if loading fails
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _save_image(self, image: np.ndarray, output_path: Union[str, Path]):
        """Save image to path.

        Args:
            image: Image array
            output_path: Path to save image
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to BGR for OpenCV
            if isinstance(image, torch.Tensor):
                image = self.gpu_accelerator.to_cpu(image)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(str(output_path), image_bgr)
            logger.info(f"Saved enhanced image to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise

    def _apply_enhancements(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Apply enhancement models to image.

        Args:
            image: Input image array or tensor

        Returns:
            Enhanced image array
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            enhanced = self.gpu_accelerator.to_gpu(image)
        else:
            enhanced = image.clone()

        iterations = 0

        while iterations < self.processing_params["max_iterations"]:
            # Convert to numpy for quality metrics
            if isinstance(enhanced, torch.Tensor):
                metrics_image = self.gpu_accelerator.to_cpu(enhanced)
            else:
                metrics_image = enhanced

            # Get current quality
            metrics = self.quality_manager.calculate_quality_metrics(metrics_image)
            quality_score = np.mean(list(metrics.values()))

            if quality_score >= self.processing_params["quality_threshold"]:
                logger.info(f"Quality threshold reached after {iterations} iterations")
                break

            # Apply models
            models = self.model_manager.get_active_models()
            for model in models:
                model_output = model.process(
                    {
                        "image": enhanced,
                        "metrics": metrics,
                        "params": self.processing_params,
                    }
                )
                enhanced = model_output["image"]

            iterations += 1

        # Convert back to numpy for return
        if isinstance(enhanced, torch.Tensor):
            enhanced = self.gpu_accelerator.to_cpu(enhanced)

        return enhanced

    def get_supported_models(self) -> List[str]:
        """Get list of supported enhancement models.

        Returns:
            List of model names
        """
        return self.model_manager.get_supported_models()

    def set_model_parameters(self, model_name: str, parameters: Dict):
        """Set parameters for specific model.

        Args:
            model_name: Name of model
            parameters: Parameter dictionary
        """
        self.model_manager.set_model_parameters(model_name, parameters)

    def get_quality_metrics(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """Get quality metrics for image.

        Args:
            image: Image path or array

        Returns:
            Dictionary of quality metrics
        """
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        return self.quality_manager.calculate_quality_metrics(image)

    def calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for image.

        Args:
            image: Input image array

        Returns:
            Dictionary of quality metrics
        """
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Calculate metrics using quality manager
        metrics = self.quality_manager.calculate_quality_metrics(image)

        # Ensure all values are numeric
        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                numeric_metrics[key] = float(value)
            except (TypeError, ValueError):
                numeric_metrics[key] = 0.0

        return numeric_metrics

    def _calculate_improvement_metrics(
        self, initial_metrics: Dict[str, float], final_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement metrics.

        Args:
            initial_metrics: Initial quality metrics
            final_metrics: Final quality metrics

        Returns:
            Dictionary of improvement metrics
        """
        improvement_metrics = {}

        # Calculate absolute improvements
        for metric in initial_metrics:
            if metric in final_metrics:
                improvement_metrics[metric] = (
                    final_metrics[metric] - initial_metrics[metric]
                )

        # Calculate overall improvement
        if improvement_metrics:
            improvement_metrics["overall_improvement"] = sum(
                improvement_metrics.values()
            ) / len(improvement_metrics)
        else:
            improvement_metrics["overall_improvement"] = 0.0

        return improvement_metrics
