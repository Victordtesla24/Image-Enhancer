"""Color enhancement module using advanced techniques"""

import cv2
import numpy as np
import logging
from ..core.processor import CoreProcessor

logger = logging.getLogger(__name__)


class ColorEnhancer:
    """Handles color enhancement using advanced techniques"""

    def __init__(self):
        self.core = CoreProcessor()
        self.config = self.core.config
        logger.info("Color enhancer initialized")

    def enhance(self, img):
        """Apply color enhancement"""
        try:
            logger.info("Applying color enhancement")

            # Convert to BGR for OpenCV processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Apply advanced white balance correction
            img = self._apply_white_balance(img)

            # Apply advanced color space enhancement
            img = self._enhance_color_space(img)

            # Apply selective color enhancement
            img = self._enhance_selective_colors(img)

            # Apply final color refinement
            img = self._apply_color_refinement(img)

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Color enhancement completed")
            return img

        except Exception as e:
            logger.error(f"Error in color enhancement: {str(e)}")
            raise

    def _apply_white_balance(self, img):
        """Apply advanced white balance correction"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Calculate color statistics
        a_avg = np.average(a)
        b_avg = np.average(b)

        # Calculate luminance weights
        l_norm = l.astype(float) / 255.0

        # Apply weighted correction
        a = a - ((a_avg - 128) * l_norm * 1.2)
        b = b - ((b_avg - 128) * l_norm * 1.2)

        # Merge channels and convert back
        corrected = cv2.merge([l, a, b])
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    def _enhance_color_space(self, img):
        """Enhance colors using multiple color spaces"""
        # Convert to LAB for lightness enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Enhance lightness using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Enhance color channels
        a = cv2.convertScaleAbs(a, alpha=1.2, beta=0)
        b = cv2.convertScaleAbs(b, alpha=1.2, beta=0)

        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Convert to HSV for saturation enhancement
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Apply adaptive saturation enhancement
        s = cv2.convertScaleAbs(s, alpha=1.3, beta=0)

        # Merge and convert back
        enhanced_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    def _enhance_selective_colors(self, img):
        """Enhance specific color ranges"""
        # Convert to HSV for selective enhancement
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Define color ranges to enhance
        color_ranges = [
            # Red tones
            {"range": (0, 20), "sat_boost": 1.3, "val_boost": 1.2},
            {"range": (160, 180), "sat_boost": 1.3, "val_boost": 1.2},
            # Green tones
            {"range": (40, 80), "sat_boost": 1.2, "val_boost": 1.1},
            # Blue tones
            {"range": (100, 140), "sat_boost": 1.2, "val_boost": 1.1},
            # Yellow tones
            {"range": (20, 40), "sat_boost": 1.25, "val_boost": 1.15},
            # Purple tones
            {"range": (140, 160), "sat_boost": 1.25, "val_boost": 1.15},
        ]

        # Process each color range
        for color_range in color_ranges:
            mask = cv2.inRange(h, color_range["range"][0], color_range["range"][1])

            # Enhance saturation and value for the masked region
            s_enhanced = cv2.multiply(s, color_range["sat_boost"], mask=mask)
            v_enhanced = cv2.multiply(v, color_range["val_boost"], mask=mask)

            # Update only the masked regions
            s = cv2.bitwise_and(s, s, mask=cv2.bitwise_not(mask))
            v = cv2.bitwise_and(v, v, mask=cv2.bitwise_not(mask))
            s = cv2.add(s, s_enhanced)
            v = cv2.add(v, v_enhanced)

        # Merge channels and convert back
        enhanced = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)

    def _apply_color_refinement(self, img):
        """Apply final color refinement"""
        # Convert to float32
        img_float = img.astype(np.float32) / 255.0

        # Apply color balance refinement
        means = np.mean(img_float, axis=(0, 1))
        max_mean = np.max(means)

        # Calculate color correction factors
        correction = max_mean / (means + 1e-6)
        correction = np.clip(correction, 0.8, 1.2)

        # Apply correction
        img_balanced = img_float * correction

        # Apply contrast enhancement
        min_val = np.min(img_balanced)
        max_val = np.max(img_balanced)
        img_balanced = (img_balanced - min_val) / (max_val - min_val)

        # Apply gamma correction
        gamma = 1.1
        img_balanced = np.power(img_balanced, 1 / gamma)

        # Convert back to uint8
        return (img_balanced * 255).astype(np.uint8)
