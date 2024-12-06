"""File uploader component"""

from PIL import Image
import io
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUploader:
    """Handle file uploads and validation"""

    def __init__(self):
        self.supported_formats = ["PNG", "JPEG", "JPG"]
        self.max_size_mb = 200

    def validate_image(self, file) -> Image.Image:
        """
        Validate and load an image file from Streamlit's uploaded file

        Args:
            file: UploadedFile from Streamlit's file_uploader

        Returns:
            PIL Image object

        Raises:
            ValueError: If file is invalid
        """
        try:
            logger.info("Starting image validation")

            if not file:
                logger.error("No file provided")
                raise ValueError("No file provided")

            # Read file content
            logger.info("Reading file content...")
            content = file.getvalue()
            if not content:
                logger.error("Empty file provided")
                raise ValueError("Empty file provided")

            # Check file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > self.max_size_mb:
                logger.error(
                    f"File size ({file_size_mb:.1f}MB) exceeds limit of {self.max_size_mb}MB"
                )
                raise ValueError(f"File size exceeds {self.max_size_mb}MB limit")

            # Try to open as image
            logger.info("Attempting to open file as image...")
            try:
                image = Image.open(io.BytesIO(content))

                # Verify it's actually an image
                image.verify()
                logger.info(f"Image verified successfully - Format: {image.format}")

                if image.format not in self.supported_formats:
                    logger.error(f"Unsupported image format: {image.format}")
                    raise ValueError(
                        f"Unsupported format. Please use {', '.join(self.supported_formats)}"
                    )

            except Exception as e:
                logger.error(f"Image verification failed: {str(e)}")
                raise ValueError(f"Invalid image format: {str(e)}")

            # Re-open because verify() closes the file
            image = Image.open(io.BytesIO(content))

            # Convert to RGB if necessary
            if image.mode in ["RGBA", "LA"]:
                logger.info("Converting RGBA/LA image to RGB")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode not in ["RGB"]:
                logger.info(f"Converting {image.mode} image to RGB")
                image = image.convert("RGB")

            # Basic image quality checks
            width, height = image.size
            if width < 100 or height < 100:
                logger.error(f"Image dimensions too small: {width}x{height}")
                raise ValueError("Image dimensions must be at least 100x100 pixels")

            # Check for corrupted or empty images
            img_array = np.array(image)
            if img_array.size == 0:
                logger.error("Empty image array")
                raise ValueError("Image appears to be corrupted or empty")

            if len(img_array.shape) != 3:
                logger.error(f"Invalid image dimensions: {img_array.shape}")
                raise ValueError("Invalid image format")

            logger.info(
                f"Image loaded successfully - Size: {width}x{height}, Mode: {image.mode}"
            )
            return image

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image validation: {str(e)}")
            raise ValueError(f"Invalid image file: {str(e)}")
