"""Detail enhancement module using advanced techniques"""

import cv2
import numpy as np
import logging
import torch
import torch.nn.functional as F
from ..core.processor import CoreProcessor

logger = logging.getLogger(__name__)


class DetailEnhancer:
    """Handles detail enhancement using advanced techniques"""

    def __init__(self):
        self.core = CoreProcessor()
        self.config = self.core.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Detail enhancer initialized")

    def enhance(self, img):
        """Apply detail enhancement"""
        try:
            logger.info("Applying detail enhancement")

            # Convert to BGR for OpenCV processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Advanced noise reduction
            img = self._apply_advanced_denoising(img)

            # Multi-scale detail enhancement
            img = self._apply_multi_scale_enhancement(img)

            # Advanced local contrast enhancement
            img = self._enhance_local_contrast(img)

            # Advanced detail sharpening
            img = self._apply_advanced_sharpening(img)

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Detail enhancement completed")
            return img

        except Exception as e:
            logger.error(f"Error in detail enhancement: {str(e)}")
            raise

    def _apply_advanced_denoising(self, img):
        """Apply advanced noise reduction"""
        # Convert to float32
        img_float = img.astype(np.float32) / 255.0

        # Apply bilateral filter for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(img_float, 5, 0.1, 5)

        # Apply non-local means denoising
        nlm = (
            cv2.fastNlMeansDenoisingColored(
                (img_float * 255).astype(np.uint8),
                None,
                h=3,
                hColor=3,
                templateWindowSize=7,
                searchWindowSize=21,
            ).astype(np.float32)
            / 255.0
        )

        # Blend the results
        result = cv2.addWeighted(bilateral, 0.5, nlm, 0.5, 0)

        return (result * 255).astype(np.uint8)

    def _apply_multi_scale_enhancement(self, img):
        """Apply multi-scale detail enhancement"""
        # Generate Gaussian pyramid
        pyramid = [img]
        num_levels = 4
        for i in range(num_levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))

        # Enhanced detail reconstruction
        for i in range(len(pyramid) - 1, 0, -1):
            size = (pyramid[i - 1].shape[1], pyramid[i - 1].shape[0])
            upscaled = cv2.pyrUp(pyramid[i], dstsize=size)

            # Calculate and enhance details
            details = cv2.subtract(pyramid[i - 1], upscaled)

            # Apply adaptive enhancement based on local variance
            local_var = cv2.GaussianBlur(cv2.multiply(details, details), (5, 5), 0)
            enhancement_factor = np.clip(1.5 - local_var * 0.01, 1.0, 2.0)
            enhanced_details = cv2.multiply(details, enhancement_factor)

            pyramid[i - 1] = cv2.add(upscaled, enhanced_details)

        return pyramid[0]

    def _enhance_local_contrast(self, img):
        """Enhance local contrast using advanced techniques"""
        # Convert to LAB for better contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply adaptive CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Apply guided filter for edge-preserving contrast enhancement
        l_guided = cv2.ximgproc.guidedFilter(l_clahe, l_clahe, radius=8, eps=100)

        # Blend original and enhanced lightness
        l_enhanced = cv2.addWeighted(l_clahe, 0.7, l_guided, 0.3, 0)

        # Merge channels
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _apply_advanced_sharpening(self, img):
        """Apply advanced adaptive sharpening"""
        # Convert to float32
        img_float = img.astype(np.float32) / 255.0

        # Calculate local variance for adaptive sharpening
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        local_var = cv2.GaussianBlur(
            cv2.multiply(gray.astype(float), gray.astype(float)), (5, 5), 0
        ) - cv2.multiply(
            cv2.GaussianBlur(gray.astype(float), (5, 5), 0),
            cv2.GaussianBlur(gray.astype(float), (5, 5), 0),
        )

        # Normalize variance
        local_var = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)
        local_var = local_var.astype(np.float32)

        # Multi-kernel sharpening
        kernels = [
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) / 9.0,
            np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, 2, 8, 2, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1],
                ],
                dtype=np.float32,
            )
            / 8.0,
        ]

        sharpened_images = []
        for kernel in kernels:
            sharpened = cv2.filter2D(img_float, -1, kernel)
            sharpened_images.append(sharpened)

        # Blend sharpened images based on local variance
        result = img_float.copy()
        for sharp_img in sharpened_images:
            blend_factor = cv2.multiply(local_var, 0.8)
            blend_factor = cv2.cvtColor(blend_factor, cv2.COLOR_GRAY2BGR)
            result = result * (1 - blend_factor) + sharp_img * blend_factor

        # Apply final refinement
        result = cv2.detailEnhance(
            (result * 255).astype(np.uint8), sigma_s=10, sigma_r=0.15
        )

        return result
